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

    # set
    instance.plants = ["wheat", "corn", "beet"]
    instance.action = ["buy", "sell"]
    instance.price = ["high", "low"]

    # decision variables
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
        return (2.5 * model.area['wheat'] + model.wheat_act['buy'] -
                model.wheat_act['sell'] >= 200)

    instance.min_wheat_constraint = Constraint(rule=min_wheat_rule)

    # constraint
    def min_corn_rule(model):
        return (3 * model.area['corn'] + model.corn_act['buy'] -
                model.corn_act['sell'] >= 240)

    instance.min_corn_constraint = Constraint(rule=min_corn_rule)

    # constraint
    def beet_price_rule(model):
        return (model.beet_price['high'] + model.beet_price['low']
                <= 20 * model.area['beet'])

    instance.beat_price_constraint = Constraint(rule=beet_price_rule)

    # objective
    def min_cost_rule(model):
        grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                     260 * model.area['beet'])
        wheat_cost = (238 * model.wheat_act['buy'] -
                      170 * model.wheat_act['sell'])
        corn_cost = (210 * model.corn_act['buy'] -
                     150 * model.corn_act['sell'])
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
    print ("LP objective: {}".format(-instance.min_cost_objective()))


def farmer_sp(yields=None):
    # concrete model
    instance = ConcreteModel(name="Farmer_SP")

    if yields is None:
        yields = np.array([
            [3, 3.6, 24],
            [2.5, 3, 20],
            [2, 2.4, 16],
        ])

    # set
    instance.plants = ["wheat", "corn", "beet"]
    instance.action = ["buy", "sell"]
    instance.price = ["high", "low"]
    instance.scenarios = range(3)

    # decision variables
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
        return (yields[sdx, 0] * model.area['wheat'] +
                model.wheat_act["buy", sdx] -
                model.wheat_act["sell", sdx] >= 200)

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
        return (yields[sdx, 2] * model.area['beet'] -
                model.beet_price['high', sdx] -
                model.beet_price['low', sdx] >= 0
                )

    instance.beat_price_constraint = Constraint(
        instance.scenarios, rule=beet_price_rule)

    # objective
    def min_cost_rule(model):
        grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                     260 * model.area['beet'])

        wheat_cost, corn_cost, beet_cost = 0, 0, 0
        for sdx in instance.scenarios:
            wheat_cost += (238 * model.wheat_act['buy', sdx] -
                           170 * model.wheat_act['sell', sdx])
            corn_cost += (210 * model.corn_act['buy', sdx] -
                          150 * model.corn_act['sell', sdx])
            beet_cost += -(36 * model.beet_price['high', sdx] +
                           10 * model.beet_price['low', sdx])
        return grow_cost + wheat_cost / 3 + corn_cost / 3 + beet_cost / 3

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                            sense=minimize)
    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)
    print ("SP: {}".format(-instance.min_cost_objective()))
    return -instance.min_cost_objective()


def farmer_wait_and_see(yields=None):
    # concrete model
    if yields is None:
        yields = np.array([
            [3, 3.6, 24],
            [2.5, 3, 20],
            [2, 2.4, 16],
        ])

    obj_values = np.zeros(3)
    for sdx in xrange(3):
        # concrete model
        instance = ConcreteModel(name="Farmer_wait_and_see_{}".format(sdx))

        # set
        instance.plants = ["wheat", "corn", "beet"]
        instance.action = ["buy", "sell"]
        instance.price = ["high", "low"]

        # decision variables
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
            return (yields[sdx, 0] * model.area['wheat'] +
                    model.wheat_act['buy'] -
                    model.wheat_act['sell'] >= 200)

        instance.min_wheat_constraint = Constraint(rule=min_wheat_rule)

        # constraint
        def min_corn_rule(model):
            return (yields[sdx, 1] * model.area['corn'] +
                    model.corn_act['buy'] -
                    model.corn_act['sell'] >= 240)

        instance.min_corn_constraint = Constraint(rule=min_corn_rule)

        # constraint
        def beet_price_rule(model):
            return (model.beet_price['high'] + model.beet_price['low']
                    <= yields[sdx, 2] * model.area['beet'])

        instance.beat_price_constraint = Constraint(rule=beet_price_rule)

        # objective
        def min_cost_rule(model):
            grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                         260 * model.area['beet'])
            wheat_cost = (238 * model.wheat_act['buy'] -
                          170 * model.wheat_act['sell'])
            corn_cost = (210 * model.corn_act['buy'] -
                         150 * model.corn_act['sell'])
            beet_cost = -(36 * model.beet_price['high'] +
                          10 * model.beet_price['low'])
            return grow_cost + wheat_cost + corn_cost + beet_cost

        instance.min_cost_objective = Objective(rule=min_cost_rule,
                                                sense=minimize)
        # solve
        opt = SolverFactory("cplex")
        results = opt.solve(instance)
        instance.solutions.load_from(results)
        # display(instance)
        # print (" WS objective: {}".format(-instance.min_cost_objective()))
        obj_values[sdx] = -instance.min_cost_objective()
    print "WS:", obj_values.mean()
    return obj_values.mean()

def farmer_eev(yields=None):
    """ value of stochastic solution """
    if yields is None:
        yields = np.array([
            [3, 3.6, 24],
            [2.5, 3, 20],
            [2, 2.4, 16],
        ])

    yields_mean = yields.mean(axis=0)

    # concrete model
    instance = ConcreteModel(name="Farmer_EEV")

    # set
    instance.plants = ["wheat", "corn", "beet"]
    instance.action = ["buy", "sell"]
    instance.price = ["high", "low"]
    instance.scenarios = range(3)

    # decision variables
    instance.area = Var(instance.plants, within=NonNegativeReals)
    instance.wheat_act = Var(instance.action, within=NonNegativeReals)
    instance.corn_act = Var(instance.action, within=NonNegativeReals)
    instance.beet_price = Var(instance.price, bounds=(0, 6000))

    # stage 1
    # constraint
    def area_rule(model):
        return sum(instance.area[pdx] for pdx in instance.plants) <= 500

    instance.area_constraint = Constraint(rule=area_rule)

    # constraint
    def min_wheat_rule(model):
        return (yields_mean[0] * model.area['wheat'] +
                model.wheat_act["buy"] -
                model.wheat_act["sell"] >= 200)

    instance.min_wheat_constraint = Constraint(rule=min_wheat_rule)

    # constraint
    def min_corn_rule(model):
        return (yields_mean[1] * model.area['corn'] +
                model.corn_act['buy'] - model.corn_act['sell'] >= 240)

    instance.min_corn_constraint = Constraint(rule=min_corn_rule)

    # constraint
    def beet_price_rule(model):
        return (model.beet_price['high'] + model.beet_price['low']
                <= yields_mean[2] * model.area['beet'])

    instance.beat_price_constraint = Constraint(rule=beet_price_rule)

    # objective
    def min_cost_rule(model):
        grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                     260 * model.area['beet'])

        wheat_cost = (238 * model.wheat_act['buy'] -
                      170 * model.wheat_act['sell'])
        corn_cost = (210 * model.corn_act['buy'] -
                     150 * model.corn_act['sell'])
        beet_cost = -(36 * model.beet_price['high'] +
                      10 * model.beet_price['low'])
        return grow_cost + wheat_cost + corn_cost + beet_cost

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                            sense=minimize)
    # solve stage-1
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    # display(instance)
    print ("EV: {}".format(-instance.min_cost_objective()))
    EV = -instance.min_cost_objective()
    # print ("area: wheat:{} corn:{} beet:{}".format(
    #     instance.area['wheat'].value, instance.area['corn'].value,
    #     instance.area['beet'].value))

    # fixed stage-1 variables
    for plant in instance.plants:
        instance.area[plant].fixed = True
    # instance.area['wheat'].fixed = True
    # instance.area['corn'].fixed = True
    # instance.area['beet'].fixed = True

    obj_values = np.zeros(3)
    for sdx in xrange(3):
        instance.del_component("min_wheat_constraint")
        instance.del_component("min_corn_constraint")
        instance.del_component("beat_price_constraint")

        def min_wheat_rule(model):
            return (yields[sdx,0] * model.area['wheat'] +
                    model.wheat_act["buy"] -
                    model.wheat_act["sell"] >= 200)

        instance.min_wheat_constraint = Constraint(rule=min_wheat_rule)

        def min_corn_rule(model):
            return (yields[sdx,1] * model.area['corn'] +
                    model.corn_act['buy'] - model.corn_act['sell'] >= 240)

        instance.min_corn_constraint = Constraint(rule=min_corn_rule)

        # constraint
        def beet_price_rule(model):
            return (model.beet_price['high'] + model.beet_price['low']
                    <= yields[sdx,2] * model.area['beet'])

        instance.beat_price_constraint = Constraint(rule=beet_price_rule)

        # solve stage-2
        opt = SolverFactory("cplex")
        results = opt.solve(instance)
        instance.solutions.load_from(results)
        # display(instance)
        # print ("scenario:{}, profit:{}".format(
        #         sdx+1, -instance.min_cost_objective()))
        obj_values[sdx] = -instance.min_cost_objective()
    print "EEV:",obj_values.mean()

    return EV, obj_values.mean()

def farmer_3stage_independent_sp():
    # concrete model
    instance = ConcreteModel(name="Farmer_3stage_independent_SP")


    yields = np.array([
        [3, 3.6, 24],
        [2.5, 3, 20],
        [2, 2.4, 16],
    ])

    yields2 = np.array([
        [2.8, 3.1, 24],
        [3.2, 2.9, 22],
        [2.7, 3.2, 19],
        [2.4, 3.1, 26],
        [2, 2, 20],
        [1.7, 3, 24],
    ])

    # set
    instance.plants = ["wheat", "corn", "beet"]
    instance.action = ["buy", "sell"]
    instance.price = ["high", "low"]
    instance.scenarios = range(len(yields))
    instance.scenarios2 = range(len(yields2))

    # stage 1 decision variables
    instance.area = Var(instance.plants, within=NonNegativeReals)
    instance.wheat_act = Var(instance.action, instance.scenarios,
                             within=NonNegativeReals)
    instance.corn_act = Var(instance.action, instance.scenarios,
                            within=NonNegativeReals)
    instance.beet_price = Var(instance.price, instance.scenarios,
                              bounds=(0, 6000))

    # stage 2
    instance.area2 = Var(instance.plants, instance.scenarios,
                         within=NonNegativeReals)
    instance.wheat_act2 = Var(instance.action, instance.scenarios2,
                             within=NonNegativeReals)

    instance.corn_act2 = Var(instance.action, instance.scenarios2,
                        within=NonNegativeReals)
    instance.beet_price2 = Var(instance.price, instance.scenarios2,
                           bounds=(0, 6000))

    # 1st constraint
    def area_rule(model):
        return sum(instance.area[pdx] for pdx in instance.plants) <= 500

    instance.area_constraint = Constraint(rule=area_rule)

    # 1st constraint
    def min_wheat_rule(model, sdx):
        return (yields[sdx, 0] * model.area['wheat'] +
                model.wheat_act["buy", sdx] -
                model.wheat_act["sell", sdx] -200 >= 0 )

    instance.min_wheat_constraint = Constraint(
        instance.scenarios, rule=min_wheat_rule)

    # 1st constraint
    def min_corn_rule(model, sdx):
        return (yields[sdx, 1] * model.area['corn'] +
                model.corn_act['buy', sdx] -
                model.corn_act['sell', sdx] -240 >= 0)

    instance.min_corn_constraint = Constraint(
        instance.scenarios, rule=min_corn_rule)

    # 1st constraint
    def beet_price_rule(model, sdx):
        return (yields[sdx, 2] * model.area['beet'] -
                model.beet_price['high', sdx] -
                model.beet_price['low', sdx] >= 0
                )

    instance.beat_price_constraint = Constraint(
        instance.scenarios, rule=beet_price_rule)

    # 2nd constraint
    def area2_rule(model, sdx):
        return sum(instance.area2[pdx, sdx] for pdx in instance.plants) <= 500

    instance.area2_constraint = Constraint(instance.scenarios,
                                           rule=area2_rule)

    # 2nd constraint
    def min_wheat2_rule(model, sdx):
        adx = int(sdx/2)
        return (yields2[sdx, 0] * model.area2['wheat', adx] +
                model.wheat_act2["buy", sdx] -
                model.wheat_act2["sell", sdx] - 200 >= 0)

    instance.min_wheat2_constraint = Constraint(
        instance.scenarios2, rule=min_wheat2_rule)

    # 2nd constraint
    def min_corn2_rule(model, sdx):
        adx = int(sdx / 2)
        return (yields2[sdx, 1] * model.area2['corn', adx] +
                model.corn_act2['buy', sdx] -
                model.corn_act2['sell', sdx] - 240 >= 0)

    instance.min_corn2_constraint = Constraint(
        instance.scenarios2, rule=min_corn2_rule)

    # 2nd constraint
    def beet_price2_rule(model, sdx):
        adx = int(sdx / 2)
        return (yields2[sdx, 2] * model.area2['beet', adx] -
                model.beet_price2['high', sdx] -
                model.beet_price2['low', sdx] >= 0)

    instance.beat_price2_constraint = Constraint(
        instance.scenarios2, rule=beet_price2_rule)

    # objective
    def min_cost_rule(model):
        grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                     260 * model.area['beet'])

        probs = np.ones(3)/3.
        scenario_cost = 0
        for sdx in instance.scenarios:
            wheat_cost =  probs[sdx] * (
                            238 * model.wheat_act['buy', sdx] -
                            170 * model.wheat_act['sell', sdx])
            corn_cost = probs[sdx] * (
                            210 * model.corn_act['buy', sdx] -
                            150 * model.corn_act['sell', sdx])
            beet_cost = - probs[sdx] * (
                            36 * model.beet_price['high', sdx] +
                            10 * model.beet_price['low', sdx])
            scenario_cost += (wheat_cost + corn_cost + beet_cost)

        s1_cost =  grow_cost + scenario_cost

        grow_cost2 = 0
        for sdx in instance.scenarios:
            grow_cost2 = grow_cost2 + probs[sdx]*(
                            150 * model.area2['wheat', sdx] +
                            230 * model.area2['corn', sdx] +
                            260 * model.area2['beet', sdx])

        scenario_cost2 = 0
        probs2 = [0.4, 0.6, 0.7, 0.3, 0.5, 0.5]

        for sdx in instance.scenarios2:
            adx = int(sdx/2)
            wheat_cost2 = probs[adx] * probs2[sdx] * (
                            238 * model.wheat_act2['buy', sdx] -
                            170 * model.wheat_act2['sell', sdx])
            corn_cost2 =  probs[adx] * probs2[sdx] * (
                            210 * model.corn_act2['buy', sdx] -
                            150 * model.corn_act2['sell', sdx])
            beet_cost2 = - probs[adx] * probs2[sdx] * (
                            36 * model.beet_price2['high', sdx] +
                            10 * model.beet_price2['low', sdx])
            scenario_cost2 +=  (wheat_cost2 + corn_cost2 + beet_cost2)
        s2_cost = grow_cost2 + scenario_cost2

        return s1_cost + s2_cost

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                            sense=minimize)
    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)
    print ("3stage_independent_SP: {}".format(-instance.min_cost_objective()))
    return -instance.min_cost_objective()


def farmer_sp2():
    # concrete model
    instance = ConcreteModel(name="Farmer_SP2")


    # yields = np.array([
    #     [2.8, 3.1, 24],
    #     [3.2, 2.9, 22],
    # ])
    # yields = np.array([
    #     [2.7, 3.2, 19],
    #     [2.4, 3.1, 26],
    # ])
    yields = np.array([
        [2, 2, 20],
        [1.7, 3, 24],
    ])

    # probs = np.array([0.4, 0.6])
    # probs = np.array([0.7, 0.3])
    probs = np.array([0.5, 0.5])

    # set
    instance.plants = ["wheat", "corn", "beet"]
    instance.action = ["buy", "sell"]
    instance.price = ["high", "low"]
    instance.scenarios = range(2)

    # decision variables
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
        return (yields[sdx, 0] * model.area['wheat'] +
                model.wheat_act["buy", sdx] -
                model.wheat_act["sell", sdx] >= 200)

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

    # objective
    def min_cost_rule(model):
        grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                     260 * model.area['beet'])

        wheat_cost, corn_cost, beet_cost = 0, 0, 0
        for sdx in instance.scenarios:
            wheat_cost += probs[sdx]* (238 * model.wheat_act['buy', sdx] -
                           170 * model.wheat_act['sell', sdx])
            corn_cost += probs[sdx]* (210 * model.corn_act['buy', sdx] -
                          150 * model.corn_act['sell', sdx])
            beet_cost += -probs[sdx]* (36 * model.beet_price['high', sdx] +
                           10 * model.beet_price['low', sdx])
        return grow_cost + wheat_cost + corn_cost + beet_cost

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                            sense=minimize)
    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)
    print ("SP2: {}".format(-instance.min_cost_objective()))
    return -instance.min_cost_objective()

def farmer_3stage_dependent_sp():
    # concrete model
    instance = ConcreteModel(name="Farmer_3stage_dependent_SP")


    yields = np.array([
        [3, 3.6, 24],
        [2.5, 3, 20],
        [2, 2.4, 16],
    ])

    yields2 = np.array([
        [2.8, 3.1, 24],
        [3.2, 2.9, 22],
        [2.7, 3.2, 19],
        [2.4, 3.1, 26],
        [2, 2, 20],
        [1.7, 3, 24],
    ])

    # set
    instance.plants = ["wheat", "corn", "beet"]
    instance.action = ["buy", "sell"]
    instance.price = ["high", "low"]
    instance.scenarios = range(len(yields))
    instance.scenarios2 = range(len(yields2))

    #parameters
    instance.reminder = Param(instance.plants, default=0)

    # stage 1 decision variables
    instance.area = Var(instance.plants, within=NonNegativeReals)
    instance.wheat_act = Var(instance.action, instance.scenarios,
                             within=NonNegativeReals)
    instance.corn_act = Var(instance.action, instance.scenarios,
                            within=NonNegativeReals)
    instance.beet_price = Var(instance.price, instance.scenarios,
                              bounds=(0, 6000))
    instance.reminder2 = Var(instance.plants, instance.scenarios,
                             within=NonNegativeReals)
    # stage 2
    instance.area2 = Var(instance.plants, instance.scenarios,
                         within=NonNegativeReals)
    instance.wheat_act2 = Var(instance.action, instance.scenarios2,
                             within=NonNegativeReals)

    instance.corn_act2 = Var(instance.action, instance.scenarios2,
                        within=NonNegativeReals)
    instance.beet_price2 = Var(instance.price, instance.scenarios2,
                           bounds=(0, 6000))

    # 1st constraint
    def area_rule(model):
        return sum(instance.area[pdx] for pdx in instance.plants) <= 500

    instance.area_constraint = Constraint(rule=area_rule)

    # 1st constraint
    def min_wheat_rule(model, sdx):
        return (model.reminder2['wheat', sdx] ==
                yields[sdx, 0] * model.area['wheat'] +
                model.wheat_act["buy", sdx] -
                model.wheat_act["sell", sdx] +
                model.reminder['wheat'] - 200)

    instance.min_wheat_constraint = Constraint(
        instance.scenarios, rule=min_wheat_rule)

    # 1st constraint
    def min_corn_rule(model, sdx):
        return (model.reminder2['corn', sdx] ==
                yields[sdx, 1] * model.area['corn'] +
                model.corn_act['buy', sdx] -
                model.corn_act['sell', sdx] +
                model.reminder['corn'] - 240)

    instance.min_corn_constraint = Constraint(
        instance.scenarios, rule=min_corn_rule)

    # 1st constraint
    def beet_price_rule(model, sdx):
        return (model.reminder2['beet', sdx] ==
                yields[sdx, 2] * model.area['beet'] +
                model.reminder['beet'] -
                model.beet_price['high',sdx] -
                model.beet_price['low', sdx] )

    instance.beat_price_constraint = Constraint(
        instance.scenarios, rule=beet_price_rule)


    # 2nd constraint
    def area2_rule(model, sdx):
        return sum(instance.area2[pdx, sdx] for pdx in instance.plants) <= 500

    instance.area2_constraint = Constraint(instance.scenarios,
                                           rule=area2_rule)

    # 2nd constraint
    def min_wheat2_rule(model, sdx):
        adx = int(sdx/2)
        return (yields2[sdx, 0] * model.area2['wheat', adx] +
                model.wheat_act2["buy", sdx] -
                model.wheat_act2["sell", sdx] +
                model.reminder2['wheat', adx] - 200 >= 0)

    instance.min_wheat2_constraint = Constraint(
        instance.scenarios2, rule=min_wheat2_rule)

    # 2nd constraint
    def min_corn2_rule(model, sdx):
        adx = int(sdx / 2)
        return (yields2[sdx, 1] * model.area2['corn', adx] +
                model.corn_act2['buy', sdx] -
                model.corn_act2['sell', sdx] +
                model.reminder2['corn', adx] - 240 >= 0)

    instance.min_corn2_constraint = Constraint(
        instance.scenarios2, rule=min_corn2_rule)

    # 2nd constraint
    def beet_price2_rule(model, sdx):
        adx = int(sdx / 2)
        return (yields2[sdx, 2] * model.area2['beet', adx] +
                model.reminder2['beet', adx] -
                model.beet_price2['high', sdx] -
                model.beet_price2['low', sdx] >= 0)

    instance.beat_price2_constraint = Constraint(
        instance.scenarios2, rule=beet_price2_rule)

    # objective
    def min_cost_rule(model):
        grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                     260 * model.area['beet'])

        probs = np.ones(3) / 3.
        scenario_cost = 0
        for sdx in instance.scenarios:
            wheat_cost = (238 * model.wheat_act['buy', sdx] -
                          170 * model.wheat_act['sell', sdx])
            corn_cost = (210 * model.corn_act['buy', sdx] -
                         150 * model.corn_act['sell', sdx])
            beet_cost = - (36 * model.beet_price['high', sdx] +
                           10 * model.beet_price['low', sdx])
            scenario_cost += probs[sdx] * (wheat_cost + corn_cost + beet_cost)

        s1_cost = grow_cost + scenario_cost

        grow_cost2 = 0
        for sdx in instance.scenarios:
            grow_cost2 += probs[sdx] * (
                150 * model.area2['wheat', sdx] +
                230 * model.area2['corn', sdx] +
                260 * model.area2['beet', sdx])

        scenario_cost2 = 0
        probs2 = [0.4, 0.6, 0.7, 0.3, 0.5, 0.5]

        for sdx in instance.scenarios2:
            adx = int(sdx / 2)
            wheat_cost2 = (238 * model.wheat_act2['buy', sdx] -
                           170 * model.wheat_act2['sell', sdx])
            corn_cost2 = (210 * model.corn_act2['buy', sdx] -
                          150 * model.corn_act2['sell', sdx])
            beet_cost2 = -(36 * model.beet_price2['high', sdx] +
                           10 * model.beet_price2['low', sdx])
            scenario_cost2 += probs[adx] * probs2[sdx] * (
                wheat_cost2 + corn_cost2 + beet_cost2)

        s2_cost = grow_cost2 + scenario_cost2

        return s1_cost + s2_cost

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                            sense=minimize)
    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)

    # 2nd cost
    grow_cost = (150 * instance.area['wheat'].value +
                 230 * instance.area['corn'].value +
                 260 * instance.area['beet'].value)

    probs = np.ones(3) / 3.
    scenario_cost = 0
    for sdx in instance.scenarios:
        wheat_cost = (238 * instance.wheat_act['buy', sdx].value -
                      170 * instance.wheat_act['sell', sdx].value)
        corn_cost = (210 * instance.corn_act['buy', sdx].value -
                     150 * instance.corn_act['sell', sdx].value)
        beet_cost = - (36 * instance.beet_price['high', sdx].value +
                       10 * instance.beet_price['low', sdx].value)
        scenario_cost += probs[sdx] * (wheat_cost + corn_cost + beet_cost)

    print ("2stage_SP: {}".format(-scenario_cost))
    print ("3stage_SP: {}".format(-instance.min_cost_objective()))
    return -instance.min_cost_objective()


def farmer_3stage_dependent_stage_sp():
    # concrete model
    instance = ConcreteModel(name="Farmer_3stage_dependent_stage_SP")


    yields = np.array([
        [3, 3.6, 24],
        [2.5, 3, 20],
        [2, 2.4, 16],
    ])

    yields2 = np.array([
        # [2.8, 3.1, 24],
        # [3.2, 2.9, 22],
        [2.7, 3.2, 19],
        [2.4, 3.1, 26],
        # [2, 2, 20],
        # [1.7, 3, 24],
    ])

    # set
    instance.plants = ["wheat", "corn", "beet"]
    instance.action = ["buy", "sell"]
    instance.price = ["high", "low"]
    instance.scenarios = range(len(yields))
    instance.scenarios2 = range(len(yields2))

    #parameters
    instance.reminder = Param(instance.plants, default=0)


    # stage 1 decision variables
    instance.area = Var(instance.plants, within=NonNegativeReals)
    instance.wheat_act = Var(instance.action, instance.scenarios,
                             within=NonNegativeReals)
    instance.corn_act = Var(instance.action, instance.scenarios,
                            within=NonNegativeReals)
    instance.beet_price = Var(instance.price, instance.scenarios,
                              bounds=(0, 6000))
    instance.reminder2 = Var(instance.plants, instance.scenarios,
                             within=NonNegativeReals)
    # stage 2
    instance.area2 = Var(instance.plants,
                         within=NonNegativeReals)
    instance.wheat_act2 = Var(instance.action, instance.scenarios2,
                             within=NonNegativeReals)

    instance.corn_act2 = Var(instance.action, instance.scenarios2,
                        within=NonNegativeReals)
    instance.beet_price2 = Var(instance.price, instance.scenarios2,
                           bounds=(0, 6000))

    # 1st constraint
    def area_rule(model):
        return sum(instance.area[pdx] for pdx in instance.plants) <= 500

    instance.area_constraint = Constraint(rule=area_rule)

    # 1st constraint
    def min_wheat_rule(model, sdx):
        return (model.reminder2['wheat', sdx] ==
                yields[sdx, 0] * model.area['wheat'] +
                model.wheat_act["buy", sdx] -
                model.wheat_act["sell", sdx] +
                model.reminder['wheat'] - 200)

    instance.min_wheat_constraint = Constraint(
        instance.scenarios, rule=min_wheat_rule)

    # 1st constraint
    def min_corn_rule(model, sdx):
        return (model.reminder2['corn', sdx] ==
                yields[sdx, 1] * model.area['corn'] +
                model.corn_act['buy', sdx] -
                model.corn_act['sell', sdx] +
                model.reminder['corn'] - 240)

    instance.min_corn_constraint = Constraint(
        instance.scenarios, rule=min_corn_rule)

    # 1st constraint
    def beet_price_rule(model, sdx):
        return (model.reminder2['beet', sdx] ==
                yields[sdx, 2] * model.area['beet'] +
                model.reminder['beet'] -
                model.beet_price['high',sdx] -
                model.beet_price['low', sdx] )

    instance.beat_price_constraint = Constraint(
        instance.scenarios, rule=beet_price_rule)


    # 2nd constraint
    def area2_rule(model):
        return sum(instance.area2[pdx] for pdx in instance.plants) <= 500

    instance.area2_constraint = Constraint(instance.scenarios,
                                           rule=area2_rule)

    # 2nd constraint
    def min_wheat2_rule(model, sdx):
        adx = model.adx
        return (yields2[sdx, 0] * model.area2['wheat'] +
                model.wheat_act2["buy", sdx] -
                model.wheat_act2["sell", sdx] +
                model.reminder2['wheat', adx] >= 200)

    instance.min_wheat2_constraint = Constraint(
        instance.scenarios2, rule=min_wheat2_rule)

    # 2nd constraint
    def min_corn2_rule(model, sdx):
        adx = model.adx
        return (yields2[sdx, 1] * model.area2['corn'] +
                model.corn_act2['buy', sdx] -
                model.corn_act2['sell', sdx] +
                model.reminder2['corn', adx] >= 240)

    instance.min_corn2_constraint = Constraint(
        instance.scenarios2, rule=min_corn2_rule)

    # 2nd constraint
    def beet_price2_rule(model, sdx):
        adx = model.adx
        return (model.beet_price2['high', sdx] + model.beet_price2['low', sdx]
                <= yields2[sdx, 2] * model.area2['beet'] +
                   model.reminder2['beet', adx]
                )

    instance.beat_price2_constraint = Constraint(
        instance.scenarios2, rule=beet_price2_rule)

    # objective
    def min_cost_rule(model):
        grow_cost = (150 * model.area['wheat'] + 230 * model.area['corn'] +
                     260 * model.area['beet'])

        probs = np.ones(3) / 3.
        scenario_cost = 0
        for sdx in instance.scenarios:
            wheat_cost = (238 * model.wheat_act['buy', sdx] -
                          170 * model.wheat_act['sell', sdx])
            corn_cost = (210 * model.corn_act['buy', sdx] -
                         150 * model.corn_act['sell', sdx])
            beet_cost = - (36 * model.beet_price['high', sdx] +
                           10 * model.beet_price['low', sdx])
            scenario_cost += probs[sdx] * (wheat_cost + corn_cost + beet_cost)

        s1_cost = grow_cost + scenario_cost

        grow_cost2 = 0
        for sdx in instance.scenarios:
            grow_cost2 += probs[sdx] * (
                150 * model.area2['wheat', sdx] +
                230 * model.area2['corn', sdx] +
                260 * model.area2['beet', sdx])

        scenario_cost2 = 0
        probs2 = [0.4, 0.6, 0.7, 0.3, 0.5, 0.5]

        for sdx in instance.scenarios2:
            adx = int(sdx / 2)
            wheat_cost2 = (238 * model.wheat_act2['buy', sdx] -
                           170 * model.wheat_act2['sell', sdx])
            corn_cost2 = (210 * model.corn_act2['buy', sdx] -
                          150 * model.corn_act2['sell', sdx])
            beet_cost2 = -(36 * model.beet_price2['high', sdx] +
                           10 * model.beet_price2['low', sdx])
            scenario_cost2 += probs[adx] * probs2[sdx] * (
                wheat_cost2 + corn_cost2 + beet_cost2)

        s2_cost = grow_cost2 + scenario_cost2

        return s1_cost + s2_cost

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                            sense=minimize)
    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)


    print ("3stage_SP: {}".format(-instance.min_cost_objective()))
    return -instance.min_cost_objective()

if __name__ == '__main__':
    # farmer_sp()
    # farmer_3stage_independent_sp()
    # farmer_sp2()
    farmer_3stage_dependent_sp()
    # farmer_3stage_dependent_stage_sp()
    # for _ in xrange(1000):
    #     yields = np.random.rand(3,3)*5
    #     yields[:, 2] += 20
    #     print yields
    #     sp = farmer_sp(yields)
    #     ws = farmer_wait_and_see(yields)
    #     ev, eev = farmer_eev(yields)
    #
    #     assert ws>=sp>=eev
