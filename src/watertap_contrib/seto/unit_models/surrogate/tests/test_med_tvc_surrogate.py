import pytest
from pyomo.environ import (
    ConcreteModel,
    value,
    assert_optimal_termination,
    units as pyunits,
)
import re
from pyomo.network import Port
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from watertap_contrib.seto.unit_models.surrogate import MEDTVCSurrogate

from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.property_models.water_prop_pack import WaterParameterBlock
from watertap_contrib.seto.costing import SETOWaterTAPCosting
from idaes.core.util.testing import initialization_tester
from watertap.core.util.initialization import assert_no_degrees_of_freedom
from pyomo.util.check_units import assert_units_consistent

from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
    unused_variables_set,
)
from idaes.core.util.testing import initialization_tester
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    constraint_scaling_transform,
    unscaled_variables_generator,
    unscaled_constraints_generator,
    badly_scaled_var_generator,
)

import idaes.logger as idaeslog

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


class TestMEDTVC:
    @pytest.fixture(scope="class")
    def MED_TVC_frame(self):
        # create model, flowsheet
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.water_prop = SeawaterParameterBlock()
        m.fs.steam_prop = WaterParameterBlock()
        m.fs.med_tvc = MEDTVCSurrogate(
            property_package_water=m.fs.water_prop,
            property_package_steam=m.fs.steam_prop,
        )

        med_tvc = m.fs.med_tvc
        feed = med_tvc.feed_props[0]
        cool = med_tvc.cooling_out_props[0]
        dist = med_tvc.distillate_props[0]
        steam = med_tvc.heating_steam_props[0]
        motive = med_tvc.motive_steam_props[0]

        # System specification
        # Input variable 1: Feed salinity (30-60 g/L = kg/m3)
        feed_salinity = 35 * pyunits.kg / pyunits.m**3

        # Input variable 2: Feed temperature (25-35 deg C)
        feed_temperature = 25

        # Input variable 3: Motive steam pressure (4-45 bar)
        motive_pressure = 24

        # Input variable 4: System capacity (2,000 - 100,000 m3/day)
        sys_capacity = 2000 * pyunits.m**3 / pyunits.day

        # Input variable 5: Recovery ratio (30%- 40%)
        recovery_ratio = 0.3 * pyunits.dimensionless

        feed_flow = pyunits.convert(
            (sys_capacity / recovery_ratio), to_units=pyunits.m**3 / pyunits.s
        )  # feed volumetric flow rate [m3/s]

        """
        Specify feed flow state properties
        """
        # Specify feed flow state properties
        med_tvc.feed_props.calculate_state(
            var_args={
                ("flow_vol_phase", "Liq"): feed_flow,
                ("conc_mass_phase_comp", ("Liq", "TDS")): feed_salinity,
                ("temperature", None): feed_temperature + 273.15,
                # feed flow is at atmospheric pressure
                ("pressure", None): 101325,
            },
            hold_state=True,
        )

        """
        Specify heating steam state properties
        """
        # Flow rate of liquid heating steam is zero
        steam.flow_mass_phase_comp["Liq", "H2O"].fix(0)

        # Heating steam temperature (saturated) is fixed at 70 C in this configuration
        steam.temperature.fix(70 + 273.15)

        # Calculate heating steam pressure (saturated)
        med_tvc.heating_steam_props.calculate_state(
            var_args={
                ("pressure_sat", None): value(steam.pressure),
            },
            hold_state=True,
        )
        # Release vapor mass flow rate
        steam.flow_mass_phase_comp["Vap", "H2O"].unfix()

        """
        Specify motive steam state properties
        """
        # Flow rate of liquid motive steam is zero
        motive.flow_mass_phase_comp["Liq", "H2O"].fix(0)

        # Calculate temperature of the motive steam (saturated)
        med_tvc.motive_steam_props.calculate_state(
            var_args={
                ("pressure", None): motive_pressure * 1e5,
                ("pressure_sat", None): motive_pressure * 1e5,
            },
            hold_state=True,
        )
        # Release vapor mass flow rate
        motive.flow_mass_phase_comp["Vap", "H2O"].unfix()

        """
        Specify distillate flow state properties
        """
        # salinity in distillate is zero
        dist.flow_mass_phase_comp["Liq", "TDS"].fix(0)

        med_tvc.recovery_vol_phase[0, "Liq"].fix(recovery_ratio)

        # Set scaling factors for mass flow rates
        m.fs.water_prop.set_default_scaling(
            "flow_mass_phase_comp", 1e-2, index=("Liq", "H2O")
        )
        m.fs.water_prop.set_default_scaling(
            "flow_mass_phase_comp", 1e3, index=("Liq", "TDS")
        )
        m.fs.steam_prop.set_default_scaling(
            "flow_mass_phase_comp", 1e-2, index=("Liq", "H2O")
        )
        m.fs.steam_prop.set_default_scaling(
            "flow_mass_phase_comp", 1, index=("Vap", "H2O")
        )

        return m

    @pytest.mark.unit
    def test_config(self, MED_TVC_frame):
        m = MED_TVC_frame
        # check unit config arguments
        assert len(m.fs.med_tvc.config) == 5

        assert not m.fs.med_tvc.config.dynamic
        assert not m.fs.med_tvc.config.has_holdup
        assert m.fs.med_tvc.config.property_package_water is m.fs.water_prop
        assert m.fs.med_tvc.config.property_package_steam is m.fs.steam_prop

    @pytest.mark.unit
    def test_num_effects_domain(self, MED_TVC_frame):
        m = MED_TVC_frame
        error_msg = re.escape(
            "Invalid parameter value: fs.med_tvc.number_effects[None] = '100', value type=<class 'int'>.\n\tValue not in parameter domain fs.med_tvc.number_effects_domain"
        )
        with pytest.raises(ValueError, match=error_msg):
            m.fs.med_tvc.number_effects.set_value(100)

    @pytest.mark.unit
    def test_build(self, MED_TVC_frame):
        m = MED_TVC_frame

        # test ports
        port_lst = ["feed", "distillate", "brine", "steam", "motive"]
        for port_str in port_lst:
            port = getattr(m.fs.med_tvc, port_str)
            assert isinstance(port, Port)
            assert len(port.vars) == 3

        # test statistics
        assert number_variables(m) == 201
        assert number_total_constraints(m) == 58
        assert number_unused_variables(m) == 76  # vars from property package parameters

    @pytest.mark.unit
    def test_dof(self, MED_TVC_frame):
        m = MED_TVC_frame
        assert degrees_of_freedom(m) == 0

    @pytest.mark.unit
    def test_calculate_scaling(self, MED_TVC_frame):
        m = MED_TVC_frame
        calculate_scaling_factors(m)

        # check that all variables have scaling factors
        unscaled_var_list = list(unscaled_variables_generator(m))
        assert len(unscaled_var_list) == 0

        # check that all constraints have been scaled
        unscaled_constraint_list = list(unscaled_constraints_generator(m))
        assert len(unscaled_constraint_list) == 0

    @pytest.mark.component
    def test_var_scaling(self, MED_TVC_frame):
        m = MED_TVC_frame
        badly_scaled_var_lst = list(badly_scaled_var_generator(m))
        assert badly_scaled_var_lst == []

    @pytest.mark.component
    def test_initialize(self, MED_TVC_frame):
        m = MED_TVC_frame
        initialization_tester(m, unit=m.fs.med_tvc, outlvl=idaeslog.DEBUG)

    @pytest.mark.component
    def test_solve(self, MED_TVC_frame):
        m = MED_TVC_frame
        results = solver.solve(m)

        # Check for optimal solution
        assert_optimal_termination(results)

    @pytest.mark.component
    def test_mass_balance(self, MED_TVC_frame):
        m = MED_TVC_frame

        med_tvc = m.fs.med_tvc

        feed_flow_m3_hr = 277.777
        dist_flow_m3_hr = 83.333
        brine_flow_m3_hr = 194.444
        cool_flow_m3_hr = 131.593

        feed_mass_flow_tot = 78.93
        cool_mass_flow_tot = 37.31
        feed_mass_flow_tds = 2.70
        brine_mass_flow_tds = 2.70
        recovery = dist_flow_m3_hr / feed_flow_m3_hr

        assert value(med_tvc.recovery_vol_phase[0, "Liq"]) == pytest.approx(
            recovery, rel=1e-3
        )
        assert value(
            pyunits.convert(
                med_tvc.feed_props[0].flow_vol_phase["Liq"]
                - med_tvc.distillate_props[0].flow_vol_phase["Liq"]
                - med_tvc.brine_props[0].flow_vol_phase["Liq"],
                to_units=pyunits.m**3 / pyunits.hr,
            )
        ) == pytest.approx(
            feed_flow_m3_hr - dist_flow_m3_hr - brine_flow_m3_hr, rel=1e-3
        )
        assert value(med_tvc.feed_cool_mass_flow) == pytest.approx(
            feed_mass_flow_tot + cool_mass_flow_tot, rel=1e-2
        )  # mass flow calculated two different ways
        assert value(med_tvc.feed_cool_vol_flow) == pytest.approx(
            (feed_flow_m3_hr + cool_flow_m3_hr), rel=1e-3
        )
        assert value(
            med_tvc.brine_props[0].flow_mass_phase_comp["Liq", "TDS"]
            - med_tvc.feed_props[0].flow_mass_phase_comp["Liq", "TDS"]
        ) == pytest.approx(feed_mass_flow_tds - brine_mass_flow_tds, rel=1e-6)

    @pytest.mark.component
    def test_solution(self, MED_TVC_frame):
        m = MED_TVC_frame

        assert pytest.approx(12.9102, rel=1e-3) == value(m.fs.med_tvc.gain_output_ratio)
        assert pytest.approx(5.1664, rel=1e-3) == value(m.fs.med_tvc.specific_area)
        assert pytest.approx(53.1622, rel=1e-3) == value(
            m.fs.med_tvc.specific_energy_consumption_thermal
        )
        assert pytest.approx(4430.19, rel=1e-3) == value(
            m.fs.med_tvc.thermal_power_requirement
        )
        assert pytest.approx(2.6766, rel=1e-3) == value(
            m.fs.med_tvc.heating_steam_props[0].flow_mass_phase_comp["Vap", "H2O"]
        )
        assert pytest.approx(1.2175, rel=1e-3) == value(
            m.fs.med_tvc.motive_steam_props[0].flow_mass_phase_comp["Vap", "H2O"]
        )
        assert pytest.approx(131.593, rel=1e-3) == value(
            pyunits.convert(
                m.fs.med_tvc.cooling_out_props[0].flow_vol_phase["Liq"],
                to_units=pyunits.m**3 / pyunits.hr,
            )
        )

    @pytest.mark.component
    def test_costing(self, MED_TVC_frame):
        m = MED_TVC_frame
        med_tvc = m.fs.med_tvc
        dist = med_tvc.distillate_props[0]
        m.fs.costing = SETOWaterTAPCosting()
        med_tvc.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

        m.fs.costing.factor_total_investment.fix(1)
        m.fs.costing.factor_maintenance_labor_chemical.fix(0)
        m.fs.costing.factor_capital_annualization.fix(0.08764)

        m.fs.costing.cost_process()
        m.fs.costing.add_annual_water_production(dist.flow_vol_phase["Liq"])
        m.fs.costing.add_LCOW(dist.flow_vol_phase["Liq"])

        assert degrees_of_freedom(m) == 0

        results = solver.solve(m)
        assert_optimal_termination(results)

        assert pytest.approx(2254.658, rel=1e-3) == value(
            m.fs.med_tvc.costing.med_specific_cost
        )
        assert pytest.approx(5126761.859, rel=1e-3) == value(
            m.fs.med_tvc.costing.capital_cost
        )
        assert pytest.approx(2705589.357, rel=1e-3) == value(
            m.fs.med_tvc.costing.membrane_system_cost
        )
        assert pytest.approx(2421172.502, rel=1e-3) == value(
            m.fs.med_tvc.costing.evaporator_system_cost
        )
        assert pytest.approx(239692.046, rel=1e-3) == value(
            m.fs.med_tvc.costing.fixed_operating_cost
        )

        assert pytest.approx(1.6905, rel=1e-3) == value(m.fs.costing.LCOW)

        assert pytest.approx(785633.993, rel=1e-3) == value(
            m.fs.costing.total_operating_cost
        )
        assert pytest.approx(5126761.859, rel=1e-3) == value(
            m.fs.costing.total_capital_cost
        )