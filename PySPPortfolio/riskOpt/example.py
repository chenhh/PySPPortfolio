# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
# import numpy as np
# import openpyxl as pyxl
# import math

i = range(1,4)
j = range(1,3)
ij = [(x, y) for x in j for y in i]
a= [5,9,4]
sp = dict(zip(i,a))
b= [4,2,8,7,1,5]
r = dict(zip(ij,b))

# sets
instance = ConcreteModel ()
instance.i = Set(initialize = i)
instance.j = Set(initialize = j)
instance.ij = Set(initialize = ij)
instance.A = RangeSet(1,10)

# parameters
instance.z  = Var (instance.i ,  within= Binary)
instance.x  = Var (instance.i ,  within= Integers)
instance.sp = Param(instance.i , initialize = sp)
instance.r  = Param (instance.ij , initialize = r)

# constraint
def zc_constraint_rule(model, jdx):
    element_sum = -50-sum(model.x[idx]*model.sp[idx]*model.r[(jdx,idx)] for
                          idx in
                          model.i)
    return (model.z[jdx] >= 0.01 * 0.01*element_sum)

instance.zc_constraint = Constraint(instance.j, rule=zc_constraint_rule)


