import os
import numpy as np

from pyomo.environ import (
    ConcreteModel,
    Objective,
    Param,
    Expression,
    Constraint,
    Block,
    log10,
    TransformationFactory,
    assert_optimal_termination,
    value,
    units as pyunits,
)
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.core.solvers.get_solver import get_solver
from idaes.models.unit_models import Product, Feed
from idaes.core.util.model_statistics import *
from idaes.core.util.scaling import (
    set_scaling_factor,
    calculate_scaling_factors,
    constraint_scaling_transform,
)
from idaes.core import UnitModelCostingBlock
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
from watertap.property_models.NaCl_prop_pack import NaClParameterBlock
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice

from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.examples.flowsheets.RO_with_energy_recovery.RO_with_energy_recovery import (
    calculate_operating_pressure,
)
from watertap_contrib.seto.analysis.net_metering.util import (
    display_ro_pv_results,
    display_pv_results,
)
from watertap_contrib.seto.costing import (
    TreatmentCosting,
    EnergyCosting,
    SETOSystemCosting,
)
from watertap_contrib.seto.solar_models.zero_order import Photovoltaic
from watertap_contrib.seto.core import SETODatabase, PySAMWaterTAP
from watertap_contrib.seto.unit_models.surrogate import PVdummy

from dispatches.unit_models.battery import BatteryStorage
__author__ = "Zhuoran Zhang"

_log = idaeslog.getLogger(__name__)
solver = get_solver()
def build_pv_battery_flowsheet(m = None,
                               GHI = 1000,
                               elec_price = 0.07,
                               ro_capacity = 6000,
                               ro_elec_req = 944.3):
    """Builds the structure of the PV-RO-battery system

    Returns:
        object: A Pyomo concrete optimization model and flowsheet
    """
    if m is None:
        m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.pv = PVdummy()
    m.fs.pv.global_horizontal_irrad.fix(GHI)
    battery = add_battery(m)

    m.fs.pv_to_ro = Var(
            initialize = 100,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'PV to RO electricity'
        )
    m.fs.grid_to_ro = Var(
            initialize = 100,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'Grid to RO electricity'
        )
    m.fs.curtailment = Var(
            initialize = 0,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'PV curtailment'
        )

    # Add energy flow balance
    @m.Constraint(doc="PV electricity generation")
    def eq_pv_elec_gen(b):
        return (
        b.fs.pv.elec_generation == b.fs.pv_to_ro + b.fs.battery.elec_in[0] + b.fs.curtailment
        )

    @m.Constraint(doc="RO electricity requirment")
    def eq_ro_elec_req(b):
        return (ro_elec_req == b.fs.pv_to_ro + b.fs.battery.elec_out[0] + b.fs.grid_to_ro
        )

    # Add grid electricity cost
    @m.Expression(doc="grid electricity cost")
    def grid_cost(b):
        return (elec_price * b.fs.grid_to_ro * b.fs.battery.dt)

    return m

def fix_dof_and_initialize(
    m,
    outlvl=idaeslog.WARNING,
):
    """Fix degrees of freedom and initialize the flowsheet

    This function fixes the degrees of freedom of each unit and initializes the entire flowsheet.

    Args:
        m: Pyomo `Block` or `ConcreteModel` containing the flowsheet
        outlvl: Logger (default: idaeslog.WARNING)
    """
    m.fs.battery.initialize(outlvl=outlvl)

    return 


def add_battery(m):
    m.fs.battery = BatteryStorage()
    m.fs.battery.charging_eta.set_value(0.95)
    m.fs.battery.discharging_eta.set_value(0.95)
    m.fs.battery.dt.set_value(1) # hr


    return m.fs.battery

if __name__ == "__main__":
    m = build_pv_battery_flowsheet()
    fix_dof_and_initialize(m)
    results = solver.solve(m)

    print('initial state of charge: ', value(m.fs.battery.initial_state_of_charge))
    print('state of charge: ', value(m.fs.battery.state_of_charge[0]))
    print('energy throughput: ', value(m.fs.battery.energy_throughput[0]))
    print('pv size: ', value(m.fs.pv.size))
    print('battery power: ', value(m.fs.battery.nameplate_power))
    print('battery energy: ', value(m.fs.battery.nameplate_energy))
