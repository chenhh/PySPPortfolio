# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from time import time
import numpy as np
import pandas as pd
from pyomo.environ import *

def min_cvar_eevip_portfolio(
        symbols, risk_rois, risk_free_roi,
        allocated_risk_wealth, allocated_risk_free_wealth,
        buy_trans_fee, sell_trans_fee, alpha, predict_risk_rois,
        predict_risk_free_roi, n_scenario, max_portfolio_size,
        solver="cplex", verbose=False):
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
    instance.max_portfolio_size = max_portfolio_size

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

    # aux variable, switching stock variable
    instance.chosen = Var(instance.symbols, within=Binary)

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

    # constraint
    def chosen_constraint_rule(model, mdx):
        total_wealth = (sum(model.allocated_risk_wealth) +
                        model.allocated_risk_free_wealth)
        return model.risk_wealth[mdx] <= model.chosen[mdx] * total_wealth

    instance.chosen_constraint = Constraint(instance.symbols,
                                            rule=chosen_constraint_rule)

    # constraint
    def portfolio_size_constraint_rule(model):
        return (sum(model.chosen[mdx] for mdx in model.symbols) <=
                model.max_portfolio_size)

    instance.portfolio_size_constraint = Constraint(
        rule=portfolio_size_constraint_rule)

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
    # chosen stock
    chosen_symbols = pd.Series([instance.chosen[mdx].value
                                for mdx in xrange(n_stock)], index=symbols)

    # value at risk (estimated)
    estimated_var = instance.Z.value
    estimated_cvar = instance.cvar_objective()

    # fixed the first-stage variables
    for mdx in instance.symbols:
        instance.buy_amounts[mdx].fixed = True
        instance.sell_amounts[mdx].fixed = True
        instance.risk_wealth[mdx].fixed = True
        instance.chosen[mdx].fixed = True
    instance.risk_free_wealth.fixed = True

    # Z is viewed as the first-stage variable
    instance.Z.fixed = True

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
        estimated_eev_cvar_arr[sdx] = instance.cvar_objective()

    if verbose:
        print "min_cvar_eevip_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "estimated_var": estimated_var,
        "estimated_cvar": estimated_cvar,
        "estimated_eev_cvar": estimated_eev_cvar_arr.mean(),
        "chosen_symbols": chosen_symbols,
    }


def min_cvar_eevip2_portfolio(
        symbols, risk_rois, risk_free_roi,
        allocated_risk_wealth, allocated_risk_free_wealth,
        buy_trans_fee, sell_trans_fee, alpha, predict_risk_rois,
        predict_risk_free_roi, n_scenario, max_portfolio_size,
        solver="cplex", verbose=False):
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
    instance.max_portfolio_size = max_portfolio_size

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

    # aux variable, switching stock variable
    instance.chosen = Var(instance.symbols, within=Binary)

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

    # constraint
    def chosen_constraint_rule(model, mdx):
        total_wealth = (sum(model.allocated_risk_wealth) +
                        model.allocated_risk_free_wealth)
        return model.risk_wealth[mdx] <= model.chosen[mdx] * total_wealth

    instance.chosen_constraint = Constraint(instance.symbols,
                                            rule=chosen_constraint_rule)

    # constraint
    def portfolio_size_constraint_rule(model):
        return (sum(model.chosen[mdx] for mdx in model.symbols) <=
                model.max_portfolio_size)

    instance.portfolio_size_constraint = Constraint(
        rule=portfolio_size_constraint_rule)

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
    # chosen stock
    chosen_symbols = pd.Series([instance.chosen[mdx].value
                                for mdx in xrange(n_stock)], index=symbols)

    # value at risk (estimated)
    estimated_var = instance.Z.value
    estimated_cvar = instance.cvar_objective()

    # fixed the first-stage variables
    for mdx in instance.symbols:
        instance.buy_amounts[mdx].fixed = True
        instance.sell_amounts[mdx].fixed = True
        instance.risk_wealth[mdx].fixed = True
        instance.chosen[mdx].fixed = True
    instance.risk_free_wealth.fixed = True

    # Z is viewed as the first-stage variable
    instance.Z.fixed = True
    estimated_eev_var = instance.Z.value
    estimated_eev_cvar_arr = np.zeros(n_scenario)

    for sdx in xrange(n_scenario):
        # delete old CVaR constraint
        scen_roi = predict_risk_rois[:, sdx]
        portfolio_value = (
            sum((1+scen_roi[mdx])* instance.risk_wealth[mdx].value
                for mdx in np.arange(n_stock)) +
            instance.risk_free_wealth.value)

        if estimated_eev_var <= portfolio_value:
            estimated_eev_cvar_arr[sdx] = estimated_eev_var
        else:
            diff = (estimated_eev_var - portfolio_value)
            estimated_eev_cvar_arr[sdx] = estimated_eev_var - 1/(1-alpha) * diff


    if verbose:
        print "min_cvar_eevip_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "estimated_var": estimated_var,
        "estimated_cvar": estimated_cvar,
        "estimated_eev_cvar": estimated_eev_cvar_arr.mean(),
        "chosen_symbols": chosen_symbols,
    }

def test_min_cvar_eevip():
    n_stock = 50
    max_portfolio_size = 5
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
    t = time()
    results =min_cvar_eevip_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario, max_portfolio_size)
    print results['estimated_eev_cvar']
    # print results["chosen_symbols"]
    print "min_eevip {:.4f} secs".format(time()-t)

    t2 = time()
    results2 =min_cvar_eevip2_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario, max_portfolio_size)
    print results2['estimated_eev_cvar']
    # print results2["chosen_symbols"]
    print "min_eevip2 {:.4f} secs".format(time()-t2)


if __name__ == '__main__':
    # for _ in xrange(100):
    #     test_min_cvar_eev_sp()
    test_min_cvar_eevip()