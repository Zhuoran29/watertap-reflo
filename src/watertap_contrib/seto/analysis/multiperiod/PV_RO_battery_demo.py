from watertap_contrib.seto.analysis.multiperiod.PV_RO_battery_mutiperiod_class import create_multiperiod_pv_battery_model
from pyomo.environ import Set, Expression, value, Objective
from idaes.core.solvers.get_solver import get_solver

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.interpolate import interp1d, griddata
import numpy as np

#%%
# Create an instance of the multiperiod model
mp = create_multiperiod_pv_battery_model(
        n_time_points=24,
        ro_capacity = 6000, # m3/day
        ro_elec_req = 944.3, # kW
        cost_battery_power = 150, # $/kW
        cost_battery_energy = 100, # $/kWh
        # 24-hr GHI in Phoenix, AZ on June 18th (W/m2)
        GHI = [0, 0, 0, 0, 0, 23, 170, 386, 596, 784, 939, 1031, 1062, 1031, 938, 790, 599, 383, 166, 31, 0, 0, 0, 0],
        elec_price = [0.14] * 24,)

solver = get_solver()
results = solver.solve(mp)
for i in range(24):
    print(f'battery status at hour: {i}', value(mp.blocks[i].process.fs.battery.state_of_charge[0]))    
    print('pv gen(kW): ', value(mp.blocks[i].process.fs.pv.elec_generation))
print('pv size: ', value(mp.blocks[0].process.fs.pv.size))
print('battery power: ', value(mp.blocks[0].process.fs.battery.nameplate_power))
print('battery energy: ', value(mp.blocks[0].process.fs.battery.nameplate_energy))
print('total cost: ', value(mp.LCOW))

#%%
# Create diagrams
plt.clf()
fig,  axes= plt.subplots(2, figsize=(8,6))
(ax1, ax2) = axes
hour = [i for i in range(24)]
battery_state = [value(mp.blocks[i].process.fs.battery.state_of_charge[0]) for i in range(24)]
pv_gen = [value(mp.blocks[i].process.fs.pv.elec_generation) for i in range(24)]
pv_curtail = [value(mp.blocks[i].process.fs.curtailment) for i in range(24)]


ax1.plot(hour, battery_state, 'r', label='Battery state (kWh)')
ax1.plot(hour, pv_gen, 'k', label = 'PV generation (kWh)')
ax1.plot(hour, pv_curtail, 'g', label = 'PV curtailment (kWh)')

ax1.set_xlabel('Hour (June 18th)')
ax1.set_ylabel('Energy (kWh)')
ax1.legend(loc="upper left", frameon = False, fontsize = 'small')


pv_to_ro = [value(mp.blocks[i].process.fs.pv_to_ro) for i in range(24)]
battery_to_ro = [value(mp.blocks[i].process.fs.battery.elec_out[0]) for i in range(24)]
grid_to_ro = [value(mp.blocks[i].process.fs.grid_to_ro) for i in range(24)]
labels=["PV to RO", "Battery to RO", "Grid to RO"]
ax2.set_xlabel('Hour (June 18th)')
ax2.set_ylabel('Energy (kWh)')
ax2.stackplot(hour, pv_to_ro, battery_to_ro, grid_to_ro, labels=labels)
ax2.legend(loc="upper left", frameon = False, fontsize = 'small')


# %%
