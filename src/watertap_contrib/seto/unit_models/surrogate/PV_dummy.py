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

from copy import deepcopy

# Import Pyomo libraries
from pyomo.environ import (
    ConcreteModel,
    value,
    Set,
    Var,
    Param,
    Suffix,
    Constraint,
    check_optimal_termination,
    assert_optimal_termination,
    units as pyunits,
)

import pyomo.environ as pyo
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
)
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.exceptions import ConfigurationError, InitializationError
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    unscaled_variables_generator,
    unscaled_constraints_generator,
    badly_scaled_var_generator,
)


_log = idaeslog.getLogger(__name__)
__author__ = "Zhuoran Zhang"


@declare_process_block_class("PVdummy")
class PVDUMMYData(UnitModelBlockData):
    """
    Dummy PV model
    """

    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False.""",
        ),
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False.""",
        ),
    )

    def build(self):
        super().build()

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        self.elec_generation = Var(
            initialize=650,
            bounds=(0, None),
            units=pyunits.kW * pyunits.h,
            doc="Electricity generation",
        )

        self.size = Var(
            initialize = 1000,
            bounds = (0,None),
            units = pyunits.kW,
            doc = 'PV size'
        )

        self.global_horizontal_irrad = Var(
            initialize = 1000,
            bounds = (0,None),
            units = pyunits.W / pyunits.m**2,
            doc = 'GHI'
        )
        if "USD_2021" not in pyo.units._pint_registry:
            pyo.units.load_definitions_from_strings(
                ["USD_2021 = 500/708.0 * USD_CE500"]
            )
        self.unit_capital_cost = Param(
            initialize=1040,
            mutable=True,
            units=pyo.units.USD_2021 / pyunits.kW,
            doc="Unit capital cost",
        )

        self.opex = Param(
            initialize=9,
            mutable=True,
            units=pyunits.USD_2021 / pyunits.kW,
            doc="Unit capital cost",
        )

        @self.Constraint(doc="calculate electricity generation")
        def eq_elec_gen(b):
            return (
                b.elec_generation == 650 * b.global_horizontal_irrad / (1000 * pyunits.W / pyunits.m**2)
                                   * b.size / (1000 * pyunits.kW)
            )        
        
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        if iscale.get_scaling_factor(self.elec_generation) is None:
            iscale.set_scaling_factor(self.elec_generation, 1e-3)

        if iscale.get_scaling_factor(self.size) is None:
            iscale.set_scaling_factor(self.size, 1e-3)

        if iscale.get_scaling_factor(self.global_horizontal_irrad) is None:
            iscale.set_scaling_factor(self.global_horizontal_irrad, 1e-3)

        sf = iscale.get_scaling_factor(self.elec_generation)
        iscale.constraint_scaling_transform(self.eq_elec_gen, sf)

if __name__=="__main__":
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.pv = PVdummy()

    pv = m.fs.pv
    pv.size.fix(1000)
    pv.global_horizontal_irrad.fix(1000)

    calculate_scaling_factors(m)

    solver = get_solver()
    results = solver.solve(m)
    assert_optimal_termination(results)

    print('electricity generation (kW): ', value(pv.elec_generation))