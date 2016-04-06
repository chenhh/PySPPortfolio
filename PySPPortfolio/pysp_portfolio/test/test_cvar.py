# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
from time import time
import os
import numpy as np
import pandas as pd
import scipy.stats as spstats
from pyomo.environ import *

def min_cvar_sp_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario, scenario_probs=None,
                          solver="cplex", verbose=False):
    """
    2nd-stage minimize conditional value at risk stochastic programming
    portfolio.
    It will be called in get_current_buy_sell_amounts function

    symbols: list of string
    risk_rois: pandas.Series, shape: (n_stock, )
    risk_free_roi: float,
    allocated_risk_wealth: pandas.Series, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, 1-alpha is the significant level
    predict_risk_ret: pandas.DataFrame, shape: (n_stock, n_scenario)
    predict_risk_free_roi: float
    n_scenario: integer
    scenario_probs: numpy.array, shape: (n_scenario,)
    solver: str, supported by Pyomo
    """
    t0 = time()
    if scenario_probs is None:
        scenario_probs = np.ones(n_scenario, dtype=np.float) / n_scenario

    # Model
    instance = ConcreteModel()

    instance.scenario_probs = scenario_probs
    instance.risk_rois = risk_rois
    instance.risk_free_roi = risk_free_roi
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alpha = alpha
    instance.predict_risk_rois = predict_risk_rois
    instance.predict_risk_free_roi = predict_risk_free_roi

    n_stock = len(symbols)
    # Set
    instance.symbols = np.arange(n_stock)
    instance.scenarios = np.arange(n_scenario)

    # decision variables
    # first stage
    instance.buy_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.sell_amounts = Var(instance.symbols, within=NonNegativeReals)

    # second stage
    instance.risk_wealth = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_free_wealth = Var(within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    instance.Z = Var()

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Ys = Var(instance.scenarios, within=NonNegativeReals)

    # constraint
    def risk_wealth_constraint_rule(model, mdx):
        """
        risk_wealth is a decision variable which depends on both buy_amount
        and sell_amount.
        i.e. the risk_wealth depends on scenario.

        buy_amount and sell_amount are first stage variable,
        risk_wealth is second stage variable.
        """
        return (model.risk_wealth[mdx] ==
                (1. + model.risk_rois[mdx]) * model.allocated_risk_wealth[mdx] +
                model.buy_amounts[mdx] - model.sell_amounts[mdx])

    instance.risk_wealth_constraint = Constraint(
        instance.symbols, rule=risk_wealth_constraint_rule)

    # constraint
    def risk_free_wealth_constraint_rule(model):
        total_sell = sum((1. - model.sell_trans_fee) * model.sell_amounts[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) * model.buy_amounts[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth ==
                (1. + risk_free_roi) * allocated_risk_free_wealth +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # constraint
    def cvar_constraint_rule(model, sdx):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        wealth = sum((1. + model.predict_risk_rois[mdx, sdx]) *
                     model.risk_wealth[mdx]
                     for mdx in model.symbols) + model.risk_free_wealth

        return model.Ys[sdx] >= (model.Z - wealth)

    instance.cvar_constraint = Constraint(instance.scenarios,
                                       rule=cvar_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        scenario_expectation = sum(model.Ys[sdx] * model.scenario_probs[sdx]
                                    for sdx in xrange(n_scenario))
        return (model.Z - 1. / (1. - model.alpha) * scenario_expectation -
         model.risk_free_wealth)

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    # if verbose:
    # display(instance)

    # buy and sell amounts
    buy_amounts = pd.Series([instance.buy_amounts[mdx].value
                             for mdx in xrange(n_stock)], index=symbols)
    sell_amounts = pd.Series([instance.sell_amounts[mdx].value
                              for mdx in xrange(n_stock)], index=symbols)

    # value at risk (estimated)
    estimated_var = instance.Z.value

    if verbose:
        print "min_cvar_sp_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "risk_wealth": sum(instance.risk_wealth[mdx].value for mdx in xrange(
            n_stock)),
        "risk_free_wealth": instance.risk_free_wealth.value,
        "estimated_var": estimated_var,
        "estimated_cvar": instance.cvar_objective()
    }


def optimal_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee,
                          predict_risk_rois, predict_risk_free_roi,
                          solver="cplex", verbose=False):
    """
    2nd-stage minimize conditional value at risk stochastic programming
    portfolio.
    It will be called in get_current_buy_sell_amounts function

    symbols: list of string
    risk_rois: numpy.array, shape: (n_stock, )
    risk_free_roi: float,
    allocated_risk_wealth: numpy.array, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, 1-alpha is the significant level
    predict_risk_ret: numpy.array, shape: (n_stock,)
    predict_risk_free_roi: float
    solver: str, supported by Pyomo
    """
    t0 = time()

    # Model
    instance = ConcreteModel()

    instance.risk_rois = risk_rois
    instance.risk_free_roi = risk_free_roi
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.predict_risk_rois = predict_risk_rois
    instance.predict_risk_free_roi = predict_risk_free_roi

    n_stock = len(symbols)
    # Set
    instance.symbols = np.arange(n_stock)

    # decision variables
    # first stage
    instance.buy_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.sell_amounts = Var(instance.symbols, within=NonNegativeReals)

    # second stage
    instance.risk_wealth = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_free_wealth = Var(within=NonNegativeReals)

    # constraint
    def risk_wealth_constraint_rule(model, mdx):
        """
        risk_wealth is a decision variable which depends on both buy_amount
        and sell_amount.
        i.e. the risk_wealth depends on scenario.

        buy_amount and sell_amount are first stage variable,
        risk_wealth is second stage variable.
        """
        return (model.risk_wealth[mdx] ==
                (1. + model.risk_rois[mdx]) * model.allocated_risk_wealth[mdx] +
                model.buy_amounts[mdx] - model.sell_amounts[mdx])

    instance.risk_wealth_constraint = Constraint(
        instance.symbols, rule=risk_wealth_constraint_rule)

    # constraint
    def risk_free_wealth_constraint_rule(model):
        total_sell = sum((1. - model.sell_trans_fee) * model.sell_amounts[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) * model.buy_amounts[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth ==
                (1. + risk_free_roi) * allocated_risk_free_wealth +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # objective
    def objective_rule(model):
        return  sum(model.risk_wealth[mdx] *(1+model.predict_risk_rois[mdx])
                    for mdx in model.symbols) + (model.risk_free_wealth *  (
                    1+model.predict_risk_free_roi))

    instance.wealth_objective = Objective(rule=objective_rule, sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    # if verbose:
    # display(instance)

    # buy and sell amounts
    buy_amounts = pd.Series([instance.buy_amounts[mdx].value
                             for mdx in xrange(n_stock)], index=symbols)
    sell_amounts = pd.Series([instance.sell_amounts[mdx].value
                              for mdx in xrange(n_stock)], index=symbols)

    if verbose:
        print "optimize_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "opt_wealth": instance.wealth_objective()
    }



def test_min_cvar_sp():
    n_stock = 2
    n_scenario = 100
    symbols = np.arange(n_stock)
    risk_rois = np.random.randn(n_stock)
    risk_free_roi = 0
    allocated_risk_wealth = np.zeros(n_stock)
    allocated_risk_free_wealth = 1e6
    buy_trans_fee = 0.001425
    sell_trans_fee = 0.004425
    alpha = 0.99
    predict_risk_rois =  np.random.randn(n_stock, n_scenario)
    predict_risk_free_roi = 0.
    results =min_cvar_sp_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario)
    print results
    risk_wealth = ((1+risk_rois) * allocated_risk_wealth +
                   results['buy_amounts'] -results['sell_amounts'])
    risk_free_wealth = ((1+risk_free_roi) * allocated_risk_free_wealth -
                        sum((1+buy_trans_fee)*results['buy_amounts']) +
                        sum((1-sell_trans_fee)*results['sell_amounts']))

    scenario_wealth = [(risk_wealth*(1+predict_risk_rois[:, sdx])).sum() +
                        risk_free_wealth
                       for sdx in xrange(n_scenario)]
    print results[ "estimated_var"]
    print sorted(scenario_wealth)




def min_cvar_3stage_dependent_sp():
    """
    stage 2: 10 scenarios,
    stage 3: 10 x 10 scenarios
    """
    n_scenario2 = 10
    n_scenario3 = 100
    n_stock = 1
    n_stage = 3
    # scenario_arch = (1, 10, (10, 10))

    # Model
    instance = ConcreteModel("min_cvar_3stage_dependent_sp")

    # conditional probability
    instance.probs2 = np.ones(n_scenario2, dtype=np.float) / n_scenario2
    instance.probs3 = (np.ones((n_scenario2, n_scenario2), dtype=np.float) /
                       n_scenario2)
    # transaction tax
    instance.buy_trans_fee = 0
    instance.sell_trans_fee = 0

    # risk preference, 1 is the most risk-averse
    instance.alpha = 0.75

    instance.risk_rois1 = np.zeros(n_stock)
    # shape: (n_stock, n_scenario2)
    instance.risk_rois2 = (np.arange(n_scenario2, dtype=np.float) /
                           n_scenario2)[np.newaxis]
    instance.risk_rois3 = np.tile(np.arange(10) / 10., (10,)
                                  ).reshape(10,10)
    # instance.risk_rois3 = np.random.randn(10, 10)

    instance.risk_free_roi = 0

    # initial conditions
    instance.risk_wealth0 = np.zeros(n_stock)
    instance.risk_free_wealth0 = 10
    instance.risk_free_roi = 0

    # Set
    instance.stages = np.arange(n_stage)
    instance.symbols = np.arange(n_stock)
    instance.scenarios2 = np.arange(n_scenario2)
    # instance.scenarios3 = np.arange(n_scenario3)

    # decision variables
    instance.buy_amounts1 = Var(instance.symbols, within=NonNegativeReals)
    instance.buy_amounts2 = Var(instance.symbols, instance.scenarios2,
                                within=NonNegativeReals)
    instance.sell_amounts1 = Var(instance.symbols, within=NonNegativeReals)
    instance.sell_amounts2 = Var(instance.symbols, instance.scenarios2,
                                 within=NonNegativeReals)

    instance.risk_wealth1 = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_wealth2 = Var(instance.symbols, instance.scenarios2,
                                within=NonNegativeReals)
    instance.risk_free_wealth1 = Var(within=NonNegativeReals)
    instance.risk_free_wealth2 = Var(instance.scenarios2,
                                     within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    instance.Z2 = Var(within=Reals)
    instance.Z3 = Var(instance.scenarios2, within=Reals)

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Ys2 = Var(instance.scenarios2, within=NonNegativeReals)
    instance.Ys3 = Var(instance.scenarios2, instance.scenarios2,
                       within=NonNegativeReals)

    # 2nd constraint
    def risk_wealth_constraint_rule(model, mdx):
        return (model.risk_wealth1[mdx] == (1. + model.risk_rois1[mdx]) *
                model.risk_wealth0[mdx] + model.buy_amounts1[mdx] -
                model.sell_amounts1[mdx])

    instance.risk_wealth_constraint = Constraint(
        instance.symbols, rule=risk_wealth_constraint_rule)

    # 2nd constraint
    def risk_free_wealth_constraint_rule(model):
        total_sell = sum((1. - model.sell_trans_fee) * model.sell_amounts1[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) * model.buy_amounts1[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth1 ==
                (1. + model.risk_free_roi) * model.risk_free_wealth0 +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # 2nd constraint
    def cvar_constraint_rule(model, sdx):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        wealth = sum((1. + model.risk_rois2[mdx, sdx]) *
                     model.risk_wealth1[mdx]
                     for mdx in model.symbols) + model.risk_free_wealth1
        return model.Ys2[sdx] >= (model.Z2 - wealth)

    instance.cvar_constraint = Constraint(instance.scenarios2,
                                          rule=cvar_constraint_rule)

    # 3rd constraint
    def risk_wealth_constraint_rule2(model, mdx, sdx):
        return (model.risk_wealth2[mdx, sdx] ==
                (1. + model.risk_rois2[mdx, sdx]) *
                model.risk_wealth1[mdx] + model.buy_amounts2[mdx, sdx] -
                model.sell_amounts2[mdx, sdx])

    instance.risk_wealth_constraint2 = Constraint(
        instance.symbols, instance.scenarios2,
        rule=risk_wealth_constraint_rule2)

    # 3rd constraint
    def risk_free_wealth_constraint_rule2(model, sdx):
        total_sell = sum((1. - model.sell_trans_fee) *
                         model.sell_amounts2[mdx, sdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) *
                        model.buy_amounts2[mdx, sdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth2[sdx] ==
                (1. + model.risk_free_roi) * model.risk_free_wealth1 +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint2 = Constraint(
        instance.scenarios2,
        rule=risk_free_wealth_constraint_rule2)

    # 3rd constraint
    def cvar_constraint_rule2(model, adx, sdx):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        wealth = sum((1. + model.risk_rois3[adx, sdx]) *
                     model.risk_wealth2[mdx, adx]
                     for mdx in model.symbols) + model.risk_free_wealth2[adx]
        return model.Ys3[adx, sdx] >= (model.Z3[adx] - wealth)

    instance.cvar_constraint2 = Constraint(
        instance.scenarios2, instance.scenarios2,
        rule=cvar_constraint_rule2)

    # objective
    def cvar_objective_rule(model):
        # stage 2
        s2_exp = sum(model.Ys2[sdx] * model.probs2[sdx]
                                   for sdx in xrange(n_scenario2))
        s2_sum =  (model.Z2 - 1. / (1. - model.alpha) * s2_exp -
                model.risk_free_wealth1)

        # stage 3
        s3_sum = 0
        for adx in model.scenarios2:
            s3_exp = sum(model.Ys3[adx, sdx] * model.probs3[adx, sdx]
                             for sdx in model.scenarios2)
            s3_sum += model.probs2[adx] * (model.Z3[adx] -
                                          1. / (1. - model.alpha) * s3_exp -
                                          model.risk_free_wealth2[adx]
                                          )
        # print s2_sum + s3_sum
        return s2_sum + s3_sum

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve
    solver = "cplex"
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    CVaR1 = instance.Z2.value - 1./(1-instance.alpha)*sum(
        instance.probs2[sdx] * instance.Ys2[sdx].value
        for sdx in instance.scenarios2
    ) - instance.risk_free_wealth1.value
    CVaR2 = np.zeros(10)
    for adx in instance.scenarios2:
        CVaR2[adx] = instance.Z3[adx].value - 1. / (1 - instance.alpha) * sum(
            instance.probs3[adx, sdx] * instance.Ys3[adx, sdx].value
            for sdx in instance.scenarios2
        ) - instance.risk_free_wealth2[adx].value

    print "CVaR1: {}".format(CVaR1)
    for adx in instance.scenarios2:
        print "CVaR2[{}]: {}".format(adx, CVaR2[adx])

    print "CVaR1 + E(CVaR2)= {}".format(CVaR1+
                                sum(instance.probs2[sdx] * CVaR2[sdx]
                                    for sdx in instance.scenarios2))
    print "Objective: {}".format(instance.cvar_objective())

    print ("solver status: {}".format(results.solver.status))
    print ("solver termination cond: {}".format(
        results.solver.termination_condition))
    print (results.solver)
    display(instance)




def min_cvar_3stage_stage_sp():
    n_scenario = 10
    n_stock = 1
    n_stage = 3
    # scenario_arch = (1, 10, (10, 10))

    # Model
    instance = ConcreteModel("min_cvar_3stage_stage_sp")

    # conditional probability
    instance.probs2 = np.ones(n_scenario, dtype=np.float) / n_scenario
    instance.probs3 = np.ones(n_scenario, dtype=np.float) / n_scenario
    # transaction tax
    instance.buy_trans_fee = 0
    instance.sell_trans_fee = 0

    # risk preference, 1 is the most risk-averse
    instance.alpha = 0.75

    instance.risk_rois1 = np.zeros(n_stock)
    # shape: (n_stock, n_scenario2)
    instance.risk_rois2 = (np.arange(n_scenario, dtype=np.float) /
                           n_scenario)[np.newaxis]
    instance.risk_rois3 = (np.arange(n_scenario, dtype=np.float) /
                           n_scenario)[np.newaxis]
    # instance.risk_rois3 = np.random.randn(10)

    instance.risk_free_roi = 0

    # 2nd realized scenario id
    instance.realized_2nd_sdx = 6

    # initial conditions
    instance.risk_wealth0 = np.zeros(n_stock)
    instance.risk_free_wealth0 = 10
    # instance.risk_wealth0 = [10,]
    # instance.risk_free_wealth0 = 0
    instance.risk_free_roi = 0

    # Set
    instance.stages = np.arange(n_stage)
    instance.symbols = np.arange(n_stock)
    instance.scenarios2 = np.arange(n_scenario)
    # instance.scenarios3 = np.arange(n_scenario3)


    # decision variables
    instance.buy_amounts1 = Var(instance.symbols, within=NonNegativeReals)
    instance.buy_amounts2 = Var(instance.symbols, within=NonNegativeReals)
    instance.sell_amounts1 = Var(instance.symbols, within=NonNegativeReals)
    instance.sell_amounts2 = Var(instance.symbols, within=NonNegativeReals)

    instance.risk_wealth1 = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_wealth2 = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_free_wealth1 = Var(within=NonNegativeReals)
    instance.risk_free_wealth2 = Var(within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    instance.Z2 = Var(within=Reals)
    instance.Z3 = Var(within=Reals)

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Ys2 = Var(instance.scenarios2, within=NonNegativeReals)
    instance.Ys3 = Var(instance.scenarios2, within=NonNegativeReals)

    # 2nd constraint
    def risk_wealth_constraint_rule(model, mdx):
        return (model.risk_wealth1[mdx] == (1. + model.risk_rois1[mdx]) *
                model.risk_wealth0[mdx] + model.buy_amounts1[mdx] -
                model.sell_amounts1[mdx])

    instance.risk_wealth_constraint = Constraint(
        instance.symbols, rule=risk_wealth_constraint_rule)

    # 2nd constraint
    def risk_free_wealth_constraint_rule(model):
        total_sell = sum((1. - model.sell_trans_fee) * model.sell_amounts1[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) * model.buy_amounts1[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth1 ==
                (1. + model.risk_free_roi) * model.risk_free_wealth0 +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # 2nd constraint
    def cvar_constraint_rule(model, sdx):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        wealth = sum((1. + model.risk_rois2[mdx, sdx]) *
                     model.risk_wealth1[mdx]
                     for mdx in model.symbols) + model.risk_free_wealth1
        return model.Ys2[sdx] >= (model.Z2 - wealth)

    instance.cvar_constraint = Constraint(instance.scenarios2,
                                          rule=cvar_constraint_rule)

    # 3rd constraint
    def risk_wealth_constraint_rule2(model, mdx):
        return (model.risk_wealth2[mdx] ==
                (1. + model.risk_rois2[mdx, model.realized_2nd_sdx]) *
                model.risk_wealth1[mdx] + model.buy_amounts2[mdx] -
                model.sell_amounts2[mdx])

    instance.risk_wealth_constraint2 = Constraint(
        instance.symbols, rule=risk_wealth_constraint_rule2)

    # 3rd constraint
    def risk_free_wealth_constraint_rule2(model):
        total_sell = sum((1. - model.sell_trans_fee) * model.sell_amounts2[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) * model.buy_amounts2[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth2 ==
                (1. + model.risk_free_roi) * model.risk_free_wealth1 +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint2 = Constraint(
        rule=risk_free_wealth_constraint_rule2)

    # 3rd constraint
    def cvar_constraint_rule2(model, sdx):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        wealth = sum((1. + model.risk_rois3[mdx, sdx]) *
                     model.risk_wealth2[mdx]
                     for mdx in model.symbols) + model.risk_free_wealth2
        return model.Ys3[sdx] >= (model.Z3 - wealth)

    instance.cvar_constraint2 = Constraint(
        instance.scenarios2, rule=cvar_constraint_rule2)

    # objective
    def cvar_objective_rule(model):
        # stage 2
        s2_exp = sum(model.Ys2[sdx] * model.probs2[sdx]
                     for sdx in xrange(n_scenario))
        s2_sum = (model.Z2 - s2_exp / (1. - model.alpha)  -
                  model.risk_free_wealth1)
        # stage 3
        s3_exp = sum(model.Ys3[sdx] * model.probs3[sdx]
                         for sdx in xrange(n_scenario))
        s3_sum = (model.Z3 - s3_exp / (1. - model.alpha) -
            model.risk_free_wealth2)
        # print s2_sum + s3_sum
        return s2_sum + s3_sum

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve
    solver = "cplex"
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    CVaR1 = instance.Z2.value - 1. / (1 - instance.alpha) * sum(
        instance.probs2[sdx] * instance.Ys2[sdx].value
        for sdx in instance.scenarios2
    ) - instance.risk_free_wealth1.value

    CVaR2 =instance.Z3.value - 1. / (1 - instance.alpha) * sum(
            instance.probs3[sdx] * instance.Ys3[sdx].value
            for sdx in instance.scenarios2
        ) - instance.risk_free_wealth2.value

    print "CVaR1: {}".format(CVaR1)
    print "CVaR2: {}".format(CVaR2)

    print "Objective: {}".format(instance.cvar_objective())

    print ("solver status: {}".format(results.solver.status))
    print ("solver termination cond: {}".format(
        results.solver.termination_condition))
    print (results.solver)
    display(instance)


if __name__ == '__main__':
    # test_min_cvar_sp()
    # min_cvar_3stage_dependent_sp()
    min_cvar_3stage_stage_sp()