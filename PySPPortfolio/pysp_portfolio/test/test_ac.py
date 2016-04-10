# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
import numpy as np
from pyomo.environ import *


def AC_lp():
    # concrete model
    instance = ConcreteModel(name="AC_LP")

    instance.periods = np.arange(3)
    instance.scenarios2 = np.arange(2)
    instance.scenarios3 = np.arange(4)

    # decision variables
    instance.x1 = Var(bounds=(0, 2), within=NonNegativeReals)
    instance.w1 = Var(within=NonNegativeReals)
    instance.y1 = Var(within=NonNegativeReals)

    instance.x2 = Var(instance.scenarios2, bounds=(0, 2),
                      within=NonNegativeReals)
    instance.w2 = Var(instance.scenarios2, within=NonNegativeReals)
    instance.y2 = Var(instance.scenarios2, within=NonNegativeReals)

    instance.x3 = Var(instance.scenarios3, bounds=(0, 2),
                      within=NonNegativeReals)
    instance.w3 = Var(instance.scenarios3, within=NonNegativeReals)
    instance.y3 = Var(instance.scenarios3, within=NonNegativeReals)

    # parameters
    instance.probs2 = [0.5, 0.5]
    instance.probs3 = [0.25, 0.25, 0.25, 0.25]
    instance.d2 = [1, 3]
    instance.d3 = [1, 3, 1, 3]

    # 1st constraint
    def demand1_rule(model):
        return (model.x1 + model.w1 + model.y1 == 1)

    instance.demand1_constraint = Constraint(rule=demand1_rule)

    # 2nd constraint
    def demand2_rule(model, sdx2):
        return (model.y1 + model.x2[sdx2] + model.w2[sdx2] - model.y2[sdx2]
                == model.d2[sdx2])

    instance.demand2_constraint = Constraint(
        instance.scenarios2, rule=demand2_rule)

    # 3rd constraint
    def demand3_rule(model, sdx3):
        adx = int(sdx3/2)
        return (model.y2[adx] + model.x3[sdx3] + model.w3[sdx3] - model.y3[sdx3]
                == model.d3[sdx3])

    instance.demand3_constraint = Constraint(
        instance.scenarios3, rule=demand3_rule)

    # objective
    def min_cost_rule(model):
        s1_cost = model.x1 + 3 * model.w1 + 0.5 * model.y1

        s2_cost = 0
        for sdx2 in instance.scenarios2:
            s2_cost += model.probs2[sdx2] * (
                model.x2[sdx2] + 3 * model.w2[sdx2] + 0.5 * model.y2[sdx2]
            )

        s3_cost = 0
        for sdx3 in instance.scenarios3:
            s3_cost += model.probs3[sdx3] * (
                model.x3[sdx3] + 3 * model.w3[sdx3]
            )

        return s1_cost + s2_cost + s3_cost

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                            sense=minimize)

    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)

    print "x1:", instance.x1.value
    print "y1:", instance.y1.value


def L_shape_example():
    # concrete model
    instance = ConcreteModel(name="L_shape_example")

    instance.periods = np.arange(2)
    instance.scenarios2 = np.arange(2)

    # decision variables
    instance.x1 = Var(bounds=(40, None), within=NonNegativeReals)
    instance.x2 = Var(bounds=(20, None), within=NonNegativeReals)
    instance.y1 = Var(within=NonNegativeReals)
    instance.y2 = Var(within=NonNegativeReals)

    instance.d1 = [500, 300]
    instance.d2 = [100, 300]
    instance.q1 = [-24, -28]
    instance.q2 = [-28, -32]
    instance.probs = [.4, .6]

    def rule1(model):
        return (model.x1 + model.x2 <=120)

    instance.rule1_constraint= Constraint(rule=rule1)

    def rule2(model):
        return (6 * model.y1 + 10 * model.y2 <= 60 * model.x1)

    instance.rule2_constraint = Constraint(rule=rule2)

    def rule3(model):
        return (8 * model.y1 + 5 * model.y2 <= 80 * model.x2)

    instance.rule3_constraint = Constraint(rule=rule3)

    def rule4(model, sdx):
        return instance.y1 <= model.d1[sdx]

    instance.rule4_constraint = Constraint(range(2), rule=rule4)

    def rule5(model, sdx):
        return instance.y2 <= model.d2[sdx]

    instance.rule5_constraint = Constraint(range(2), rule=rule5)

    def min_cost_rule(model):
        s1 = 100 * model.x1 + 150 * model.x2
        s2 = 0
        for sdx in range(2):
            s2 += model.probs[sdx]* (model.q1[sdx] * model.y1 +
                                     model.q2[sdx] * model.y2)

        return s1 + s2

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                                sense=minimize)


    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)



if __name__ == '__main__':
    # AC_lp()
    L_shape_example()



