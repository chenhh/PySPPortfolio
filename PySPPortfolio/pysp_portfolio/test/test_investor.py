# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
import numpy as np
from pyomo.environ import *

def Multistage_investor_RP():

    # concrete model
    instance = ConcreteModel(name="Multistage_investor_RP")

    # parameters
    instance.initial_money = 55
    instance.target_money = 80
    instance.income_interest = 1
    instance.borrow_interest = 4
    instance.targets = ['stock', 'bond']

    # decision variables
    instance.surplus = Var(range(8), within=NonNegativeReals)
    instance.deficit = Var(range(8), within=NonNegativeReals)
    instance.x = Var(range(7), instance.targets,
                     within=NonNegativeReals)

    # scenarios
    instance.ret = [(1.25, 1.14), (1.06, 1.12)]

    # constraints
    def initial_rule(model):
        return (model.x[0, 'stock'] + model.x[0, 'bond'] ==
                model.initial_money)

    instance.initial_constraint = Constraint(rule=initial_rule)

    def roi_rule(model, sdx):
        if sdx in (1, 2):
            adx = 0
        elif sdx in (3, 4):
            adx = 1
        elif sdx in (5, 6):
            adx = 2
        ret = model.ret[(sdx+1)%2]

        cur = model.x[sdx, 'stock'] + model.x[sdx, 'bond']
        prev = model.x[adx,'stock'] * ret[0] + model.x[adx, 'bond'] * ret[1]
        return cur - prev == 0

    instance.roi_constraint = Constraint(
        range(1, 7), rule=roi_rule)

    def goal_rule(model, sdx):
        if sdx in (0, 1):
            adx = 3
        elif sdx in (2, 3):
            adx = 4
        elif sdx in (4, 5):
            adx = 5
        elif sdx in (6, 7):
            adx = 6
        ret = model.ret[sdx%2]

        prev =  model.x[adx,'stock'] * ret[0] + model.x[adx, 'bond'] * ret[1]
        return ( prev  - model.surplus[sdx] +
                 model.deficit[sdx] == model.target_money)

    instance.goal_constraint = Constraint(
        range(8), rule=goal_rule)

    def profit_rule(model):
        profit = sum(instance.income_interest * model.surplus[sdx] -
                       instance.borrow_interest * model.deficit[sdx]
                     for sdx in xrange(8))
        return profit/8.

    instance.max_profit_objective = Objective(rule=profit_rule,
                                            sense=maximize)

    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)

def Multistage_investor_EV():

    # concrete model
    instance = ConcreteModel(name="Multistage_investor_EV")

    # parameters
    instance.initial_money = 55
    instance.target_money = 80
    instance.income_interest = 1
    instance.borrow_interest = 4
    instance.targets = ['stock', 'bond']

    # decision variables
    instance.surplus = Var(within=NonNegativeReals)
    instance.deficit = Var(within=NonNegativeReals)
    instance.x = Var(range(3), instance.targets,
                     within=NonNegativeReals)

    # scenarios
    instance.ret = [1.155, 1.13]

    # constraints
    def initial_rule(model):
        return (model.x[0, 'stock'] + model.x[0, 'bond'] ==
                model.initial_money)

    instance.initial_constraint = Constraint(rule=initial_rule)

    def roi_rule(model, tdx):
        ret = model.ret
        cur = model.x[tdx, 'stock'] + model.x[tdx, 'bond']
        prev = model.x[tdx-1,'stock'] * ret[0] + model.x[tdx-1, 'bond'] * ret[1]
        return cur - prev == 0

    instance.roi_constraint = Constraint(
        [1,2], rule=roi_rule)

    def goal_rule(model):
        ret = model.ret
        prev =  model.x[2,'stock'] * ret[0] + model.x[2, 'bond'] * ret[1]
        return ( prev  - model.surplus +
                 model.deficit == model.target_money)

    instance.goal_constraint = Constraint(rule=goal_rule)

    def profit_rule(model):
        profit = (instance.income_interest * model.surplus -
                       instance.borrow_interest * model.deficit)
        return profit

    instance.max_profit_objective = Objective(rule=profit_rule,
                                            sense=maximize)

    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)


def Multistage_investor_Event():
    # each node of the tree in the wood has two-branch

    # concrete model
    instance = ConcreteModel(name="Multistage_investor_Event")

    # parameters
    instance.initial_money = 55
    instance.target_money = 80
    instance.income_interest = 1
    instance.borrow_interest = 4
    instance.targets = ['stock', 'bond']

    # decision variables
    instance.surplus = Var(range(2), within=NonNegativeReals)
    instance.deficit = Var(range(2), within=NonNegativeReals)
    instance.y = Var(range(3), instance.targets, range(2),
                     within=NonNegativeReals)
    instance.x = Var(range(3), instance.targets, within=NonNegativeReals)

    # scenarios
    instance.ret = [(1.25, 1.14), (1.06, 1.12)]

    # constraints
    def initial_rule(model):
        return (model.y[0, 'stock', 0] + model.y[0, 'bond', 0] ==
                model.initial_money)

    instance.initial_constraint = Constraint(rule=initial_rule)


    def roi_rule(model, tdx, sdx):
        """
        tdx in (1, 2)
        sdx in (0, 1)
        """
        ret = model.ret[sdx]
        curr = model.y[tdx, 'stock', sdx] + model.y[tdx, 'bond', sdx]
        prev = (model.x[tdx-1, 'stock'] * ret[0] +
                model.x[tdx-1, 'bond'] * ret[1])
        return curr - prev == 0

    instance.roi_constraint = Constraint(
        [1, 2], [0, 1], rule=roi_rule)

    def goal_rule(model, sdx):
        """
        sdx in (0, 1)
        """
        tdx = 2
        ret = model.ret[sdx]
        prev = (model.x[tdx, 'stock'] * ret[0] +
                model.x[tdx, 'bond'] * ret[1])
        return ( prev  - model.surplus[sdx] +
                 model.deficit[sdx] == model.target_money)

    instance.goal_constraint = Constraint(
        range(2), rule=goal_rule)

    def avg_decision_rule(model, tdx, pdx):
        """
        tdx in (0, 1, 2)
        pdx in ['stock', 'bond']
        """
        if tdx == 0:
            return (model.x[tdx, pdx] == model.y[tdx, pdx, 0])
        else:
            return (model.x[tdx, pdx] == (0.5 * model.y[tdx, pdx, 0] +
                                 0.5 * model.y[tdx, pdx, 1]))

    instance.avg_decision_constraint = Constraint(
        range(3), ['stock', 'bond'], rule=avg_decision_rule)


    def profit_rule(model):
        profit = sum(instance.income_interest * model.surplus[sdx] -
                       instance.borrow_interest * model.deficit[sdx]
                     for sdx in xrange(2))
        return profit/2.

    instance.max_profit_objective = Objective(rule=profit_rule,
                                            sense=maximize)

    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)

def Multistage_investor2():
    # only 2-time decisions

    # concrete model
    instance = ConcreteModel(name="Multistage_investor")

    # parameters
    instance.initial_money = 55
    instance.target_money = 80
    instance.income_interest = 1
    instance.borrow_interest = 4
    instance.targets = ['stock', 'bond']

    # decision variables
    instance.surplus = Var(range(4), within=NonNegativeReals)
    instance.deficit = Var(range(4), within=NonNegativeReals)
    instance.x = Var(range(3), instance.targets,
                     within=NonNegativeReals)

    # scenarios
    instance.ret = [(1.25, 1.14), (1.06, 1.12)]

    # constraints
    def initial_rule(model):
        return (model.x[0, 'stock'] + model.x[0, 'bond'] ==
                model.initial_money)

    instance.initial_constraint = Constraint(rule=initial_rule)

    def roi_rule(model, sdx):
        if sdx in (1, 2):
            adx = 0

        ret = model.ret[(sdx+1)%2]

        cur = model.x[sdx, 'stock'] + model.x[sdx, 'bond']
        prev = model.x[adx,'stock'] * ret[0] + model.x[adx, 'bond'] * ret[1]
        return cur - prev == 0

    instance.roi_constraint = Constraint(
        range(1, 3), rule=roi_rule)

    def goal_rule(model, sdx):
        if sdx in (0, 1):
            adx = 1
        elif sdx in (2, 3):
            adx = 2

        ret = model.ret[sdx%2]

        prev =  model.x[adx,'stock'] * ret[0] + model.x[adx, 'bond'] * ret[1]
        return ( prev  - model.surplus[sdx] +
                 model.deficit[sdx] == model.target_money)

    instance.goal_constraint = Constraint(
        range(4), rule=goal_rule)

    def profit_rule(model):
        profit = sum(instance.income_interest * model.surplus[sdx] -
                       instance.borrow_interest * model.deficit[sdx]
                     for sdx in xrange(4))
        return profit/4.

    instance.max_profit_objective = Objective(rule=profit_rule,
                                            sense=maximize)

    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)

def Multistage_investor_RP2():

    # concrete model
    instance = ConcreteModel(name="Multistage_investor_RP2")

    # parameters
    instance.initial_money = 55
    # instance.target_money = 80
    instance.income_interest = 1
    instance.borrow_interest = 4
    instance.targets = ['stock', 'bond']

    # decision variables
    instance.surplus = Var(range(8), within=NonNegativeReals)
    # instance.deficit = Var(range(8), within=NonNegativeReals)
    instance.x = Var(range(7), instance.targets,
                     within=NonNegativeReals)

    # scenarios
    instance.ret = [(1.25, 1.14), (1.06, 1.12)]

    # constraints
    def initial_rule(model):
        return (model.x[0, 'stock'] + model.x[0, 'bond'] ==
                model.initial_money)

    instance.initial_constraint = Constraint(rule=initial_rule)

    def roi_rule(model, sdx):
        if sdx in (1, 2):
            adx = 0
        elif sdx in (3, 4):
            adx = 1
        elif sdx in (5, 6):
            adx = 2
        ret = model.ret[(sdx+1)%2]

        cur = model.x[sdx, 'stock'] + model.x[sdx, 'bond']
        prev = model.x[adx,'stock'] * ret[0] + model.x[adx, 'bond'] * ret[1]
        return cur - prev == 0

    instance.roi_constraint = Constraint(
        range(1, 7), rule=roi_rule)

    def goal_rule(model, sdx):
        if sdx in (0, 1):
            adx = 3
        elif sdx in (2, 3):
            adx = 4
        elif sdx in (4, 5):
            adx = 5
        elif sdx in (6, 7):
            adx = 6
        ret = model.ret[sdx%2]

        prev =  model.x[adx,'stock'] * ret[0] + model.x[adx, 'bond'] * ret[1]
        return ( prev  == model.surplus[sdx])

    instance.goal_constraint = Constraint(
        range(8), rule=goal_rule)

    def profit_rule(model):
        profit = sum( model.surplus[sdx] for sdx in xrange(8))
        return profit/8.

    instance.max_profit_objective = Objective(rule=profit_rule,
                                            sense=maximize)

    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)

def Multistage_expected_ret():
    # concrete model
    instance = ConcreteModel(name="Multistage_expected_ret")

    # parameters
    instance.initial_money = 10

    # decision variables
    instance.wealth = Var(range(3),  within=NonNegativeReals)

    # scenarios
    instance.ret = [0.3, -0.1]
    instance.ret2 = [-0.1 , -0.3, 0.4, 0.2]




    def roi_rule(model, sdx):
        if sdx in (1, 2):
            adx = 0
        elif sdx in (3, 4):
            adx = 1
        elif sdx in (5, 6):
            adx = 2
        ret = model.ret[(sdx + 1) % 2]

        cur = model.x[sdx, 'stock'] + model.x[sdx, 'bond']
        prev = model.x[adx, 'stock'] * ret[0] + model.x[adx, 'bond'] * ret[1]
        return cur - prev == 0

    instance.roi_constraint = Constraint(
        range(1, 7), rule=roi_rule)

    def goal_rule(model, sdx):
        if sdx in (0, 1):
            adx = 3
        elif sdx in (2, 3):
            adx = 4
        elif sdx in (4, 5):
            adx = 5
        elif sdx in (6, 7):
            adx = 6
        ret = model.ret[sdx % 2]

        prev = model.x[adx, 'stock'] * ret[0] + model.x[adx, 'bond'] * ret[1]
        return (prev - model.surplus[sdx] +
                model.deficit[sdx] == model.target_money)

    instance.goal_constraint = Constraint(
        range(8), rule=goal_rule)

    def profit_rule(model):
        profit = sum(instance.income_interest * model.surplus[sdx] -
                     instance.borrow_interest * model.deficit[sdx]
                     for sdx in xrange(8))
        return profit / 8.

    instance.max_profit_objective = Objective(rule=profit_rule,
                                              sense=maximize)

    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)

if __name__ == '__main__':
    # Multistage_investor_RP()
    # Multistage_investor_RP2()
    # Multistage_investor_EV()
    Multistage_investor_Event()
    # Multistage_investor2()










