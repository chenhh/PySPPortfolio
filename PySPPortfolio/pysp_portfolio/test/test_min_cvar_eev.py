# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from time import time
import numpy as np
import pandas as pd
from pyomo.environ import *


def min_cvar_ws_portfolio(symbols, risk_rois, risk_free_roi,
                           allocated_risk_wealth, allocated_risk_free_wealth,
                           buy_trans_fee, sell_trans_fee, alpha,
                           predict_risk_rois, predict_risk_free_roi,
                           n_scenario, solver="cplex", verbose=False):
    """
    given mean scenario vector, solve the first-stage buy and sell amounts

    symbols: list of string
    risk_rois: numpy.array, shape: (n_stock, )
    risk_free_roi: float,
    allocated_risk_wealth: numpy.array, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, 1-alpha is the significant level

    ** only one scenario, the mean scenario vector
    predict_risk_ret: numpy.array, shape: (n_stock, n_scenario)
    predict_risk_free_roi: float
    n_scenario: 1
    solver: str, supported by Pyomo
    """
    t0 = time()

    # Model
    instance = ConcreteModel()

    # data
    instance.risk_rois = risk_rois
    instance.risk_free_roi = risk_free_roi
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alpha = alpha

    # scenario vectors
    # shape:(n_stock, n_scenario)
    instance.all_predict_risk_rois = predict_risk_rois
    # float
    instance.predict_risk_free_roi = predict_risk_free_roi

    n_stock = len(symbols)
    # Set
    instance.symbols = np.arange(n_stock)

    # decision variables
    # first stage
    instance.buy_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.sell_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_wealth = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_free_wealth = Var(within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    instance.Z = Var()

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Y = Var(within=NonNegativeReals)

    # constraint
    def risk_wealth_constraint_rule(model, mdx):
        """
        risk_wealth is a decision variable which depends on both buy_amount
        and sell_amount.
        i.e. the risk_wealth depends on scenario.

        buy_amount and sell_amount are first stage variable,
        risk_wealth is second stage variable.
        """
        return (model.risk_wealth[mdx] == (1. + model.risk_rois[mdx]) *
                model.allocated_risk_wealth[mdx] + model.buy_amounts[mdx] -
                model.sell_amounts[mdx])

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
    def cvar_objective_rule(model):
        return model.Z - 1. / (1. - model.alpha) * model.Y

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)


    estimated_ws_var_arr = np.zeros(n_scenario)
    estimated_ws_cvar_arr = np.zeros(n_scenario)
    for sdx in xrange(n_scenario):
        # constraint
        def cvar_constraint_rule(model):
            """ auxiliary variable Y depends on scenario. CVaR <= VaR """
            wealth = sum((1. + model.all_predict_risk_rois[mdx, sdx]) *
                         model.risk_wealth[mdx]
                         for mdx in model.symbols)

            return model.Y >= (model.Z - wealth)

        instance.cvar_constraint = Constraint(rule=cvar_constraint_rule)

        # 1st-stage solve
        opt = SolverFactory(solver)
        results = opt.solve(instance)
        instance.solutions.load_from(results)

        if verbose:
            display(instance)

        # extract results
        estimated_ws_var_arr[sdx] = instance.Z.value
        estimated_ws_cvar_arr[sdx] = instance.cvar_objective()

        # print "scenario:{}, Y:{}, VaR:{}, CVaR:{}".format(
        #     sdx+1, instance.Y.value,
        #     estimated_ws_var_arr[sdx], estimated_ws_cvar_arr[sdx])
        #
        # print "risk:{}, risk_free:{}, fwealth:{}".format(
        #     [instance.risk_wealth[mdx].value for mdx in instance.symbols],
        #     instance.risk_free_wealth.value,
        #     sum(instance.risk_wealth[mdx].value *
        #         (1+instance.all_predict_risk_rois[mdx, sdx]) for
        #         mdx in instance.symbols)
        # )

        # delete old CVaR constraint
        instance.del_component("cvar_constraint")

    # print "sorted:"
    # estimated_ws_cvar_arr.sort()
    # print estimated_ws_cvar_arr

    if verbose:
        print "min_cvar_ws_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "estimated_ws_var": estimated_ws_var_arr.mean(),
        "estimated_ws_cvar": estimated_ws_cvar_arr.mean()
    }


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
    risk_rois: numpy.array, shape: (n_stock, )
    risk_free_roi: float,
    allocated_risk_wealth: numpy.array, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, 1-alpha is the significant level
    predict_risk_ret: numpy.array, shape: (n_stock, n_scenario)
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
    instance.risk_wealth = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_free_wealth = Var(within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    instance.Z = Var()

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Ys = Var(instance.scenarios, within=NonNegativeReals)

    # constraint
    def risk_wealth_constraint_rule(model,  mdx):
        """
        risk_wealth is a decision variable which depends on both buy_amount
        and sell_amount.
        i.e. the risk_wealth depends on scenario.

        buy_amount and sell_amount are first stage variable,
        risk_wealth is second stage variable.
        """
        return (model.risk_wealth[mdx] == (1. + model.risk_rois[mdx]) *
                model.allocated_risk_wealth[mdx] +
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
                     for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - wealth)

    instance.cvar_constraint = Constraint(instance.scenarios,
                                       rule=cvar_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        scenario_expectation = sum(model.Ys[sdx] * model.scenario_probs[sdx]
                                    for sdx in xrange(n_scenario))
        return model.Z - 1. / (1. - model.alpha) * scenario_expectation

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    print ("VaR:{}, CVaR:{}".format(
        instance.Z.value, instance.cvar_objective()))

    if verbose:
        display(instance)

    # value at risk (estimated)
    estimated_var = instance.Z.value

    if verbose:
        print "min_cvar_sp_portfolio OK, {:.3f} secs".format(time() - t0)


    risk_arr = [instance.risk_wealth[mdx].value for mdx in instance.symbols]
    risk_free = instance.risk_free_wealth.value
    scen_cvars = np.array([sum(risk_arr[mdx] * (1+predict_risk_rois[mdx, sdx])
                               for mdx in instance.symbols)
                           for sdx in xrange(n_scenario)])
    scen_cvars += risk_free
    scen_cvars.sort()
    # print "sorted:"
    # print scen_cvars
    # edx = n_scenario - int(n_scenario * alpha)
    # print scen_cvars[:edx].mean()

    return {
        "estimated_var": estimated_var,
        "estimated_cvar": instance.cvar_objective()
    }


def min_cvar_sp_portfolio2(symbols, risk_rois, risk_free_roi,
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
    risk_rois: numpy.array, shape: (n_stock, )
    risk_free_roi: float,
    allocated_risk_wealth: numpy.array, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, 1-alpha is the significant level
    predict_risk_ret: numpy.array, shape: (n_stock, n_scenario)
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
    instance.risk_wealth = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_free_wealth = Var(within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    instance.Zs = Var(instance.scenarios)

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Ys = Var(instance.scenarios, within=NonNegativeReals)
    instance.slacks = Var(instance.scenarios, within=NonNegativeReals)

    # constraint
    def risk_wealth_constraint_rule(model,  mdx):
        """
        risk_wealth is a decision variable which depends on both buy_amount
        and sell_amount.
        i.e. the risk_wealth depends on scenario.

        buy_amount and sell_amount are first stage variable,
        risk_wealth is second stage variable.
        """
        return (model.risk_wealth[mdx] == (1. + model.risk_rois[mdx]) *
                model.allocated_risk_wealth[mdx] +
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
                     for mdx in model.symbols)
        # return model.Ys[sdx] >= (model.Z - wealth)
        return model.Ys[sdx] == (model.slacks[sdx] + model.Zs[sdx]/n_scenario -
                                 wealth)

    instance.cvar_constraint = Constraint(instance.scenarios,
                                       rule=cvar_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        value = sum(model.Zs[sdx]/n_scenario - 1. /
                    (1. - model.alpha) * model.Ys[sdx]
                                    for sdx in xrange(n_scenario))/n_scenario
        print value
        return value

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    print ("VaR:{}, CVaR:{}".format(
        sum(instance.Zs[sdx].value for sdx in xrange(n_scenario)),
            instance.cvar_objective()))

    if verbose:
        display(instance)

    # value at risk (estimated)
    estimated_var =  sum(instance.Zs[sdx].value for sdx in xrange(n_scenario))

    if verbose:
        print "min_cvar_sp_portfolio OK, {:.3f} secs".format(time() - t0)

    print "slacks:", [instance.slacks[sdx].value for sdx in xrange(n_scenario)]
    print "Ys:", [instance.Ys[sdx].value for sdx in xrange(n_scenario)]
    risk_arr = [instance.risk_wealth[mdx].value for mdx in instance.symbols]
    risk_free = instance.risk_free_wealth.value
    scen_cvars = np.array([sum(risk_arr[mdx] * (1+predict_risk_rois[mdx, sdx])
                               for mdx in instance.symbols)
                           for sdx in xrange(n_scenario)])
    print scen_cvars
    scen_cvars += risk_free
    scen_cvars.sort()
    # print "sorted:"
    # print scen_cvars
    # edx = n_scenario - int(n_scenario * alpha)
    # print scen_cvars[:edx].mean()

    return {
        "estimated_var": estimated_var,
        "estimated_cvar": instance.cvar_objective()
    }


def min_cvar_eev_portfolio(symbols, risk_rois, risk_free_roi,
                           allocated_risk_wealth, allocated_risk_free_wealth,
                           buy_trans_fee, sell_trans_fee, alpha,
                           predict_risk_rois, predict_risk_free_roi,
                           n_scenario, solver="cplex", verbose=False):
    """
    given mean scenario vector, solve the first-stage buy and sell amounts

    symbols: list of string
    risk_rois: numpy.array, shape: (n_stock, )
    risk_free_roi: float,
    allocated_risk_wealth: numpy.array, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, 1-alpha is the significant level

    ** only one scenario, the mean scenario vector
    predict_risk_ret: numpy.array, shape: (n_stock, n_scenario)
    predict_risk_free_roi: float
    n_scenario: 1
    solver: str, supported by Pyomo
    """
    t0 = time()

    # Model
    instance = ConcreteModel()

    # data
    instance.risk_rois = risk_rois
    instance.risk_free_roi = risk_free_roi
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alpha = alpha

    # scenario vectors
    # shape:(n_stock, n_scenario)
    instance.all_predict_risk_rois = predict_risk_rois
    # shape: (n_stock,)
    instance.mean_predict_risk_rois = predict_risk_rois.mean(axis=1)
    # float
    instance.predict_risk_free_roi = predict_risk_free_roi

    n_stock = len(symbols)
    # Set
    instance.symbols = np.arange(n_stock)

    # decision variables
    # first stage
    instance.buy_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.sell_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_wealth = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_free_wealth = Var(within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    instance.Z = Var()

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Y = Var(within=NonNegativeReals)

    # constraint
    def risk_wealth_constraint_rule(model, mdx):
        """
        risk_wealth is a decision variable which depends on both buy_amount
        and sell_amount.
        i.e. the risk_wealth depends on scenario.

        buy_amount and sell_amount are first stage variable,
        risk_wealth is second stage variable.
        """
        return (model.risk_wealth[mdx] == (1. + model.risk_rois[mdx]) *
                model.allocated_risk_wealth[mdx] + model.buy_amounts[mdx] -
                model.sell_amounts[mdx])

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
    def cvar_constraint_rule(model):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        wealth = sum((1. + model.mean_predict_risk_rois[mdx]) *
                     model.risk_wealth[mdx]
                     for mdx in model.symbols)

        return model.Y >= (model.Z - wealth)

    instance.cvar_constraint = Constraint(rule=cvar_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        return model.Z - 1. / (1. - model.alpha) * model.Y

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)
    # 1st-stage solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    if verbose:
        display(instance)

    # extract results
    buy_amounts = pd.Series([instance.buy_amounts[mdx].value
                             for mdx in xrange(n_stock)], index=symbols)
    sell_amounts = pd.Series([instance.sell_amounts[mdx].value
                              for mdx in xrange(n_stock)], index=symbols)

    # value at risk (estimated)
    estimated_var = instance.Z.value
    estimated_cvar = instance.cvar_objective()

    # fixed the first-stage variables
    for mdx in instance.symbols:
        instance.buy_amounts[mdx].fixed = True
        instance.sell_amounts[mdx].fixed = True
        instance.risk_wealth[mdx].fixed = True
    instance.Z.fixed = True
    instance.risk_free_wealth.fixed = True

    estimated_eev_var_arr = np.zeros(n_scenario)
    estimated_eev_cvar_arr = np.zeros(n_scenario)

    for sdx in xrange(n_scenario):
        # delete old CVaR constraint
        instance.del_component("cvar_constraint")

        # update CVaR constraint
        def cvar_constraint_rule(model):
            """ auxiliary variable Y depends on scenario. CVaR <= VaR """
            wealth = sum((1. + model.all_predict_risk_rois[mdx, sdx]) *
                         model.risk_wealth[mdx]
                         for mdx in model.symbols)
            return model.Y >= (model.Z - wealth)

        instance.cvar_constraint = Constraint(rule=cvar_constraint_rule)

        # 2nd-stage solve
        opt = SolverFactory(solver)
        results = opt.solve(instance)
        instance.solutions.load_from(results)

        # extract results
        estimated_eev_var_arr[sdx] = instance.Z.value
        estimated_eev_cvar_arr[sdx] = instance.cvar_objective()

        # print "scenario:{}, Y:{}, VaR:{}, CVaR:{}, [{}, {}]".format(
        #     sdx+1, instance.Y.value,
        #     estimated_eev_var_arr[sdx], estimated_eev_cvar_arr[sdx],
        # instance.Z.value,
        # instance.cvar_objective())
        # print "Y:", instance.Y.value
        #
        # print "risk:{}, risk_free:{}, fwealth:{}".format(
        #     [instance.risk_wealth[mdx].value for mdx in instance.symbols],
        #     instance.risk_free_wealth.value,
        #     sum(instance.risk_wealth[mdx].value *
        #         (1+instance.all_predict_risk_rois[mdx, sdx]) for
        #         mdx in instance.symbols)
        # )
    # print "sorted:"
    # estimated_eev_cvar_arr.sort()
    # print estimated_eev_cvar_arr

    if verbose:
        print "min_cvar_eev_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "estimated_var": estimated_var,
        "estimated_cvar": estimated_cvar,
        "estimated_eev_var": estimated_eev_var_arr.mean(),
        "estimated_eev_cvar": estimated_eev_cvar_arr.mean()
    }


def test_min_cvar_eev_sp():
    n_stock = 5
    n_scenario = 200
    symbols = np.arange(n_stock)
    risk_rois = np.random.randn(n_stock)
    risk_free_roi = 0
    allocated_risk_wealth = np.zeros(n_stock)
    allocated_risk_free_wealth = 1e6
    buy_trans_fee = 0.0
    sell_trans_fee = 0.0
    alpha = 0.5
    predict_risk_rois =  np.random.randn(n_stock, n_scenario)
    predict_risk_free_roi = 0
    # print predict_risk_rois[:,0]

    results =min_cvar_sp_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario)
    # print results
    print "SP:"
    # print "VaR:",results['estimated_var']
    print "CVaR:",results['estimated_cvar']
    print "*"*50
    # results_sp2 =min_cvar_sp_portfolio2(symbols, risk_rois, risk_free_roi,
    #                       allocated_risk_wealth, allocated_risk_free_wealth,
    #                       buy_trans_fee, sell_trans_fee, alpha,
    #                       predict_risk_rois, predict_risk_free_roi,
    #                       n_scenario)
    # # print results
    # print "SP2:"
    # print "VaR:",results_sp2['estimated_var']
    # print "CVaR:",results_sp2['estimated_cvar']
    # print "*"*50

    results2 =min_cvar_eev_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario)
    # print results2
    print "EEV:"
    # print "EV VaR:", results2['estimated_var']
    # print "EV CVaR:", results2['estimated_cvar']
    # print "buy:", results2['buy_amounts'].sum()
    # print "sell:", results['sell_amounts'].sum()
    # print "EEV VaR:", results2['estimated_eev_var']
    print "EEV CVaR:", results2['estimated_eev_cvar']
    print "*"*50

    results3 =min_cvar_ws_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario)
    # print "WS VaR:", results3['estimated_ws_var']
    print "WS CVaR:", results3['estimated_ws_cvar']

    assert (results3['estimated_ws_cvar'] >= results['estimated_cvar']
            >= results2['estimated_eev_cvar'])
    print "EVSI:",results3['estimated_ws_cvar'] -  results['estimated_cvar']
    print "VSS:",  results['estimated_cvar'] - results2['estimated_eev_cvar']
if __name__ == '__main__':
    for _ in xrange(100):
        test_min_cvar_eev_sp()