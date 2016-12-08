# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition


def product_line(n_consumer,  n_product, n_point, price, solver='gurobi') :
    '''
    只有限制式5的規劃
    決策變數為x_{ijn}與pi_{jn}
    :return:
    '''

    # Model
    instance = ConcreteModel()

    # Set
    instance.consumers = range(n_consumer)
    instance.products = range(n_product)
    instance.points = range(n_point)


    # decision variables
    instance.x = Var(instance.consumers, instance.products, instance.points,
                   within=Binary)

    instance.pi = Var(instance.products, instance.points, within=Binary)

    # constraint
    def PiConstraint_rule(model, j):
        return sum( model.pi[j, ndx] for ndx in model.points)

    instance.PiConstraint = Constraint(instance.products)

    # objective function
    def objective_function(model):
        return = sum(model.x[idx, jdx ,ndx] * price[jdx, ndx]
               for idx in model.consumers
               for jdx in model.products
               for ndx in model.points)

    model.RObjective = Objective(rule=objective_function, sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)