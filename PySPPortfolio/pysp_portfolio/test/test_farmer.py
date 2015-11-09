# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from __future__ import division
import numpy as np
from pyomo.environ import *

def farmer_lp():
    # concrete model
    instance = ConcreteModel(name="Farmer_LP")

    #set
    instance.plants = ["wheat", "corn", "beet"]
    instance.action =["buy", "sell"]
    instance.price =["high", "low"]

    #decision variables
    instance.area = Var(instance.plants, within=NonNegativeReals)
    instance.wheat_act = Var(instance.action, within=NonNegativeReals)
    instance.corn_act = Var(instance.action, within=NonNegativeReals)
    instance.beet_price = Var(instance.price, bounds=(0, 6000),
                              within=NonNegativeReals)

    # constraint
    def area_rule(model):
        return sum(instance.area[pdx] for pdx in instance.plants) <= 500

    instance.area_constraint = Constraint(rule=area_rule)

    # constraint
    def min_wheat_rule(model):
        return (2.5* model.area['wheat'] + model.wheat_act['buy'] -
                model.wheat_act['sell'] >= 200)

    instance.min_wheat_constraint = Constraint(rule=min_wheat_rule)

    # constraint
    def min_corn_rule(model):
        return (3*model.area['corn'] + model.corn_act['buy'] -
                model.corn_act['sell'] >= 240)

    instance.min_corn_constraint = Constraint(rule=min_corn_rule)

    # constraint
    def beet_price_rule(model):
        return (model.beet_price['high'] + model.beet_price['low']
                <= 20*model.area['beet'])

    instance.beat_price_constraint = Constraint(rule=beet_price_rule)

    #objective
    def min_cost_rule(model):
        grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                     260*model.area['beet'])
        wheat_cost = (238 * model.wheat_act['buy'] -
                      170* model.wheat_act['sell'])
        corn_cost = (210*model.corn_act['buy'] -
                     150* model.corn_act['sell'])
        beet_cost = -(36 * model.beet_price['high'] +
                      10 * model.beet_price['low'])
        return grow_cost + wheat_cost + corn_cost + beet_cost

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                        sense=minimize)
    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)


def farmer_sp():
    # concrete model
    instance = ConcreteModel(name="Farmer_SP")

    yields = np.array([
        [3, 3.6, 24],
        [2.5, 3, 20],
        [2, 2.4, 16],
    ])

    #set
    instance.plants = ["wheat", "corn", "beet"]
    instance.action =["buy", "sell"]
    instance.price =["high", "low"]
    instance.scenarios = range(3)

    #decision variables
    instance.area = Var(instance.plants, within=NonNegativeReals)
    instance.wheat_act = Var(instance.action, instance.scenarios,
                             within=NonNegativeReals)
    instance.corn_act = Var(instance.action, instance.scenarios,
                            within=NonNegativeReals)
    instance.beet_price = Var(instance.price, instance.scenarios,
                              bounds=(0, 6000))

    # constraint
    def area_rule(model):
        return sum(instance.area[pdx] for pdx in instance.plants) <= 500

    instance.area_constraint = Constraint(rule=area_rule)

    # constraint
    def min_wheat_rule(model, sdx):
        return (yields[sdx,0] * model.area['wheat'] +
                model.wheat_act["buy",sdx] -
                model.wheat_act["sell",sdx] >= 200)

    instance.min_wheat_constraint = Constraint(
        instance.scenarios, rule=min_wheat_rule)

    # constraint
    def min_corn_rule(model, sdx):
        return (yields[sdx, 1] * model.area['corn'] +
                model.corn_act['buy', sdx] -
                model.corn_act['sell', sdx] >= 240)

    instance.min_corn_constraint = Constraint(
        instance.scenarios, rule=min_corn_rule)

    # constraint
    def beet_price_rule(model, sdx):
        return (model.beet_price['high', sdx] + model.beet_price['low', sdx]
                <= yields[sdx, 2] * model.area['beet'])

    instance.beat_price_constraint = Constraint(
        instance.scenarios, rule=beet_price_rule)

    #objective
    def min_cost_rule(model):
        grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                     260*model.area['beet'])

        wheat_cost, corn_cost, beet_cost = 0, 0, 0
        for sdx in instance.scenarios:
            wheat_cost += (238 * model.wheat_act['buy', sdx] -
                      170* model.wheat_act['sell', sdx])
            corn_cost += (210*model.corn_act['buy', sdx] -
                     150* model.corn_act['sell', sdx])
            beet_cost += -(36 * model.beet_price['high', sdx] +
                      10 * model.beet_price['low', sdx])
        return grow_cost + wheat_cost/3 + corn_cost/3 + beet_cost/3

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                        sense=minimize)
    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)

if __name__ == '__main__':
    farmer_sp()