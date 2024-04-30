###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

import pytest
from pyomo.environ import (
    ConcreteModel,
    Suffix,
    value,
    Var,
    Set,
    assert_optimal_termination,
)
from pyomo.util.check_units import assert_units_consistent
from idaes.core import FlowsheetBlock

from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import (
    number_variables,
    number_total_constraints,
    number_unused_variables,
)

from idaes.core.util.scaling import calculate_scaling_factors

from watertap.core.util.initialization import check_dof
from watertap.property_models.tests.property_test_harness import PropertyAttributeError

import watertap_contrib.reflo.property_models.fo_draw_solution_properties as ds_props


solver = get_solver()

# class TestFODrawSolutionProperty_idaes(PropertyTestHarness_idaes):
#     def configure(self):
#         self.prop_pack = ds_props.FODrawSolutionParameterBlock
#         self.param_args = {}
#         self.prop_args = {}

#         self.skip_initialization_raises_exception_test = True
#         self.skip_state_block_mole_frac_phase_comp_test = True


@pytest.fixture(scope="module")
def m():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = ds_props.FODrawSolutionParameterBlock()

    return m


@pytest.mark.unit
def test_parameter_block(m):
    assert isinstance(m.fs.properties.component_list, Set)
    for j in m.fs.properties.component_list:
        assert j in ["H2O", "DrawSolution"]
    assert isinstance(m.fs.properties.solute_set, Set)
    assert "H2O" not in m.fs.properties.solute_set
    assert isinstance(m.fs.properties.solvent_set, Set)
    for j in m.fs.properties.solvent_set:
        assert j in ["H2O"]

    assert isinstance(m.fs.properties.phase_list, Set)
    for j in m.fs.properties.phase_list:
        assert j in ["Liq"]

    assert m.fs.properties._state_block_class is ds_props.FODrawSolutionStateBlock


@pytest.mark.component
def test_parameters(m):
    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)

    assert hasattr(m.fs.stream[0], "scaling_factor")
    assert isinstance(m.fs.stream[0].scaling_factor, Suffix)

    state_vars_lst = ["flow_mass_phase_comp", "temperature", "pressure"]
    state_vars_dict = m.fs.stream[0].define_state_vars()
    assert len(state_vars_dict) == len(state_vars_lst)
    for sv in state_vars_lst:
        assert sv in state_vars_dict
        assert hasattr(m.fs.stream[0], sv)
        var = getattr(m.fs.stream[0], sv)
        assert isinstance(var, Var)

    metadata = m.fs.properties.get_metadata().properties

    # check that properties are not built if not demanded
    for v in metadata.list_supported_properties():
        if metadata[v.name].method is not None:
            if m.fs.stream[0].is_property_constructed(v.name):
                raise PropertyAttributeError(
                    "Property {v_name} is an on-demand property, but was found "
                    "on the stateblock without being demanded".format(v_name=v.name)
                )

    # check that properties are built if demanded
    for v in metadata.list_supported_properties():
        if metadata[v.name].method is not None:
            if not hasattr(m.fs.stream[0], v.name):
                raise PropertyAttributeError(
                    "Property {v_name} is an on-demand property, but was not built "
                    "when demanded".format(v_name=v.name)
                )

    assert number_variables(m) == 12
    assert number_total_constraints(m) == 8
    assert number_unused_variables(m) == 2

    m.fs.stream[0].flow_mass_phase_comp["Liq", "H2O"].fix(0.5)
    m.fs.stream[0].flow_mass_phase_comp["Liq", "DrawSolution"].fix(0.5)
    m.fs.stream[0].temperature.fix(25 + 273.15)
    m.fs.stream[0].pressure.fix(101325)
    m.fs.stream[0].dens_mass_phase[...]
    m.fs.stream[0].cp_mass_phase[...]
    m.fs.stream[0].pressure_osm_phase[...]
    m.fs.stream[0].flow_vol_phase[...]
    m.fs.stream[0].conc_mass_phase_comp[...]
    m.fs.stream[0].mass_frac_phase_comp[...]

    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "DrawSolution")
    )

    calculate_scaling_factors(m)

    assert_units_consistent(m)

    check_dof(m, fail_flag=True)

    m.fs.stream.initialize()

    results = solver.solve(m)
    assert_optimal_termination(results)

    assert value(m.fs.stream[0].dens_mass_phase["Liq"]) == pytest.approx(
        1067.9535, rel=1e-3
    )
    assert value(m.fs.stream[0].cp_mass_phase["Liq"]) == pytest.approx(
        2785.2975, rel=1e-3
    )
    assert value(m.fs.stream[0].pressure_osm_phase["Liq"]) == pytest.approx(
        7.3161e7, rel=1e-3
    )
    assert value(m.fs.stream[0].flow_vol_phase["Liq"]) == pytest.approx(
        9.3637e-4, rel=1e-3
    )
    assert value(m.fs.stream[0].conc_mass_phase_comp["Liq", "H2O"]) == pytest.approx(
        533.977, rel=1e-3
    )
    assert value(
        m.fs.stream[0].conc_mass_phase_comp["Liq", "DrawSolution"]
    ) == pytest.approx(533.977, rel=1e-3)
    assert value(m.fs.stream[0].mass_frac_phase_comp["Liq", "H2O"]) == pytest.approx(
        0.5, rel=1e-3
    )
    assert value(
        m.fs.stream[0].mass_frac_phase_comp["Liq", "DrawSolution"]
    ) == pytest.approx(0.5, rel=1e-3)