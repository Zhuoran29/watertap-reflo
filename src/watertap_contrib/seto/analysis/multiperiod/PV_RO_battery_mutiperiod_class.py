
# General python imports
import numpy as np
import pandas as pd
import logging
from collections import deque

# Pyomo imports
from pyomo.environ import Set, Expression, value, Objective

# IDAES imports
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
# Flowsheet function imports
from watertap_contrib.seto.analysis.multiperiod.PV_RO_battery_flowsheet import (
    build_pv_battery_flowsheet,
    fix_dof_and_initialize,
)


__author__ = "Zhuoran Zhang"

_log = idaeslog.getLogger(__name__)
solver = get_solver()

def get_pv_ro_variable_pairs(t1, t2):
    """
    This function returns paris of variables that need to be connected across two time periods

    Args:
        t1: current time block
        t2: next time block

    Returns:
        None
    """
    return [
        (t1.fs.battery.state_of_charge[0], t2.fs.battery.initial_state_of_charge),
        (t1.fs.battery.energy_throughput[0], t2.fs.battery.initial_energy_throughput),
        (t1.fs.battery.nameplate_power, t2.fs.battery.nameplate_power),
        (t1.fs.battery.nameplate_energy, t2.fs.battery.nameplate_energy),
        (t1.fs.pv.size, t2.fs.pv.size)]

def unfix_dof(m):
    """
    This function unfixes a few degrees of freedom for optimization

    Args:
        m: object containing the integrated nuclear plant flowsheet

    Returns:
        None
    """
    m.fs.battery.nameplate_energy.unfix()
    m.fs.battery.nameplate_power.unfix()
    m.fs.battery.initial_state_of_charge.unfix()
    m.fs.battery.initial_energy_throughput.unfix()


    return

def create_multiperiod_pv_battery_model(
        n_time_points=24,
        ro_capacity = 6000, # m3/day
        ro_elec_req = 944.3, # kW
        cost_battery_power = 75, # $/kW
        cost_battery_energy = 50, # $/kWh
        # 24-hr GHI in Phoenix, AZ on June 18th (W/m2)
        GHI = [0, 0, 0, 0, 0, 23, 170, 386, 596, 784, 939, 1031, 1062, 1031, 938, 790, 599, 383, 166, 31, 0, 0, 0, 0],
        elec_price = [0.07] * 24,
    ):
    """
    This function creates a multi-period pv battery flowsheet object. This object contains 
    a pyomo model with a block for each time instance.

    Args:
        n_time_points: Number of time blocks to create

    Returns:
        Object containing multi-period vagmd batch flowsheet model
    """
    mp = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=build_pv_battery_flowsheet,
        linking_variable_func=get_pv_ro_variable_pairs,
        # initialization_func=fix_dof_and_initialize,
        # unfix_dof_func=unfix_dof,
        outlvl=logging.WARNING,
    )

    flowsheet_options={ t: {"GHI": GHI[t], 
                            "elec_price": elec_price[t],
                            "ro_capacity": ro_capacity, 
                            "ro_elec_req": ro_elec_req} 
                            for t in range(n_time_points)
    }

    # create the multiperiod object
    mp.build_multi_period_model(
        model_data_kwargs=flowsheet_options,
        flowsheet_options={ "ro_capacity": ro_capacity, 
                            "ro_elec_req": ro_elec_req},
        # initialization_options=None,
        # unfix_dof_options=None,
        )

    # initialize the beginning status of the system
    mp.blocks[0].process.fs.battery.initial_state_of_charge.fix(0)
    mp.blocks[0].process.fs.battery.initial_energy_throughput.fix(0)

    # Add battery cost function
    @mp.Expression(doc="battery cost")
    def battery_cost(b):
        return ( 0.096 * # capital recovery factor
            (cost_battery_power * b.blocks[0].process.fs.battery.nameplate_power
            +cost_battery_energy * b.blocks[0].process.fs.battery.nameplate_energy))
        
    # Add PV cost function
    @mp.Expression(doc="PV cost")
    def pv_cost(b):
        return (
            1040 * b.blocks[0].process.fs.pv.size * 0.096 # Annualized CAPEX
            +9 * b.blocks[0].process.fs.pv.size)          # OPEX

    # Total cost
    @mp.Expression(doc='total cost')
    def total_cost(b):
        # The annualized capital cost is evenly distributed to the multiperiod
        return (
            (b.battery_cost + b.pv_cost) / 365 / 24 * n_time_points
            + sum([b.blocks[i].process.grid_cost for i in range(n_time_points)])
        )

    # LCOW
    @mp.Expression(doc='total cost')
    def LCOW(b):
        # LCOW from RO: 0.45
        return (
            b.total_cost / ro_capacity / 24 * n_time_points + 0.45
        )   

    # Set objective
    mp.obj = Objective(expr=mp.LCOW)

    return mp


if __name__ == "__main__":
    mp = create_multiperiod_pv_battery_model()
    results = solver.solve(mp)

    for i in range(24):
        print(f'battery status at hour: {i}', value(mp.blocks[i].process.fs.battery.state_of_charge[0]))    
        print('pv gen(kW): ', value(mp.blocks[i].process.fs.curtailment))
    print('pv size: ', value(mp.blocks[0].process.fs.pv.size))
    print('battery power: ', value(mp.blocks[0].process.fs.battery.nameplate_power))
    print('battery energy: ', value(mp.blocks[0].process.fs.battery.nameplate_energy))
    print('total cost: ', value(mp.LCOW))