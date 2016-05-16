# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
from time import time
from datetime import datetime
import os
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition

from PySPPortfolio.pysp_portfolio import *
from base_model import (SPTradingPortfolio, )


def min_ms_cvar_eventsip_portfolio(symbols, trans_dates, risk_rois,
                                risk_free_rois, allocated_risk_wealth,
                                allocated_risk_free_wealth, buy_trans_fee,
                                sell_trans_fee, alpha, predict_risk_rois,
                                predict_risk_free_roi, n_scenario=200,
                                max_portfolio_size=5,
                                solver = DEFAULT_SOLVER, verbose=False,
                                solver_io="lp", keepfiles=False):
    """
    in each period, when the decision variables have branch, using the
    expected decisions

    symbols: list of string
    risk_rois: numpy.array, shape: (n_exp_period, n_stock)
    risk_free_rois: numpy.array,, shape: (n_exp_period,)
    allocated_risk_wealth: numpy.array, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alphas: float
    predict_risk_rois: numpy.array,
    shape: (n_exp_period, n_stock, n_scenario)
    predict_risk_free_rois: numpy.array, shape: (n_exp_period,)
    n_scenario: integer
    solver: str, supported by Pyomo
    solve_io: {"lp", "nl", "os", "python"}
    """
    print ("start time: {}".format(datetime.now()))
    t0 = time()
    n_exp_period = risk_rois.shape[0]
    n_stock = len(symbols)
    assert n_stock == 50
    assert max_portfolio_size <= n_stock

    # concrete model
    instance = ConcreteModel(name="ms_min_cvar_eventsip_portfolio")
    param = "{}_{}_m{}_p{}_s{}_a{:.2f}".format(
        trans_dates[0].strftime("%Y%m%d"), trans_dates[-1].strftime("%Y%m%d"),
        n_stock, n_exp_period, n_scenario, alpha)

    # data
    instance.risk_rois = risk_rois
    instance.risk_free_rois = risk_free_rois
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alpha = alpha
    instance.max_portfolio_size = max_portfolio_size
    # shape: (n_exp_period, n_stock, n_scenario)
    instance.predict_risk_rois = predict_risk_rois
    # shape: (n_exp_period, )
    instance.predict_risk_free_roi = predict_risk_free_roi

    # Set
    instance.n_exp_period = n_exp_period
    instance.exp_periods = np.arange(n_exp_period)
    instance.symbols = np.arange(n_stock)
    instance.scenarios = np.arange(n_scenario)
    instance.n_scenario = n_scenario

    # decision variables
    # the expected buy or sell actions, shape: (n_exp_period, n_stock)
    # decision from period 1 to T.
    instance.buy_amounts = Var(instance.exp_periods, instance.symbols,
                               within=NonNegativeReals)
    instance.sell_amounts = Var(instance.exp_periods, instance.symbols,
                                within=NonNegativeReals)
    instance.proxy_buy_amounts = Var(instance.exp_periods, instance.symbols,
                                 instance.scenarios, within=NonNegativeReals)
    instance.proxy_sell_amounts = Var(instance.exp_periods, instance.symbols,
                                  instance.scenarios, within=NonNegativeReals)

    # shape: (n_exp_period, n_stock)
    # decision from period 1 to T
    instance.risk_wealth = Var(instance.exp_periods, instance.symbols,
                               within=NonNegativeReals)
    instance.proxy_risk_wealth = Var(instance.exp_periods, instance.symbols,
                                instance.scenarios, within=NonNegativeReals)
    # shape: (n_exp_period, )
    # decision from period 1 to T
    instance.risk_free_wealth = Var(instance.exp_periods,
                                    within=NonNegativeReals)
    instance.proxy_risk_free_wealth = Var(instance.exp_periods,
                                  instance.scenarios, within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    # shape: (n_exp_period, )
    # decision from period 2 to T+1
    instance.Z = Var(instance.exp_periods, within=Reals)

    # aux variable, portfolio wealth less than than VaR (Z)
    # in each stage, there is only one scenario,
    # shape: (n_exp_period, ),
    # decision from period 2 to T+1
    instance.Ys = Var(instance.exp_periods, instance.scenarios,
                      within=NonNegativeReals)

    # aux variable, switching stock variable
    instance.chosen = Var(instance.exp_periods,
                          instance.symbols, within=Binary)

    print ("dimensions (exp_period, stock, scenarios)=({}, {}, {})".format(
        n_exp_period, n_stock, n_scenario))
    print ("*"*50)
    print ("constructing risk wealth constraints")

    # constraint
    def risk_wealth_constraint_rule(model, tdx, mdx, sdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        sdx: integer, index of scenario
        """
        if tdx == 0:
            prev_risk_wealth = model.allocated_risk_wealth[mdx]
            risk_roi = model.risk_rois[tdx, mdx]
        else:
            prev_risk_wealth = model.risk_wealth[tdx-1, mdx]
            # t-th day realization
            risk_roi = model.predict_risk_rois[tdx-1 , mdx, sdx]

        return (model.proxy_risk_wealth[tdx, mdx, sdx] ==
                (1. + risk_roi) * prev_risk_wealth +
                model.proxy_buy_amounts[tdx, mdx, sdx] -
                model.proxy_sell_amounts[tdx, mdx, sdx]
                )

    instance.risk_wealth_constraint = Constraint(
        instance.exp_periods, instance.symbols, instance.scenarios,
        rule=risk_wealth_constraint_rule)

    # explicit constraint
    def risk_wealth_root_rule(model, mdx, sdx):
        """
        because the risk_roi has the same value in all scenarios in the root
        node, it should have the same decision value at the period.
        The risk wealth is the final value of buy and sell amounts,
        then we can give the constraint on risk_wealth which will imply to
        buy and sell amounts.
        """
        return (model.proxy_risk_wealth[0, mdx, sdx-1] ==
                model.proxy_risk_wealth[0, mdx, sdx])

    instance.risk_wealth_root_constraint = Constraint(
        instance.symbols, range(1, n_scenario),
        rule=risk_wealth_root_rule)


    # risk wealth constraint
    def risk_wealth_expected_decision_rule(model, tdx, mdx):
        """
        expectation of decision variables

        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        exp_wealth = sum(model.proxy_risk_wealth[tdx, mdx, sdx]
                         for sdx in model.scenarios) / model.n_scenario
        return model.risk_wealth[tdx, mdx] == exp_wealth

    instance.risk_wealth_decision_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=risk_wealth_expected_decision_rule
    )

    # buy amount constraint
    def buy_expected_decision_rule(model, tdx, mdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        exp_buy = sum(model.proxy_buy_amounts[tdx, mdx, sdx]
                     for sdx in model.scenarios) / model.n_scenario
        return model.buy_amounts[tdx, mdx] == exp_buy

    instance.buy_decision_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=buy_expected_decision_rule
    )

    # buy amount constraint
    def sell_expected_decision_rule(model, tdx, mdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        exp_sell = sum(model.proxy_sell_amounts[tdx, mdx, sdx]
                      for sdx in model.scenarios) / model.n_scenario
        return model.sell_amounts[tdx, mdx] == exp_sell

    instance.sell_decision_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=sell_expected_decision_rule
    )
    print ("min_ms_cvar_eventsp {} risk wealth decisions constraints OK, "
           "{:.3f} secs".format(param, time() - t0))
    print ("*" * 50)
    print ("constructing risk free wealth constraints")
    t1 = time()

    # constraint
    def risk_free_wealth_constraint_rule(model, tdx, sdx):
        """
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, index of scenario
        """
        total_sell = sum((1. - model.sell_trans_fee) *
                         model.proxy_sell_amounts[tdx, mdx, sdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) *
                        model.proxy_buy_amounts[tdx, mdx, sdx]
                        for mdx in model.symbols)
        if tdx == 0:
            prev_risk_free_wealth = model.allocated_risk_free_wealth
        else:
            prev_risk_free_wealth = model.risk_free_wealth[tdx - 1]

        return (model.proxy_risk_free_wealth[tdx, sdx] ==
                (1. + model.risk_free_rois[tdx]) *
                prev_risk_free_wealth + total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        instance.exp_periods, instance.scenarios,
        rule=risk_free_wealth_constraint_rule)

    # risk wealth constraint
    def risk_free_wealth_decision_rule(model, tdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        """
        exp_wealth = sum(model.proxy_risk_free_wealth[tdx, sdx]
                         for sdx in model.scenarios) / model.n_scenario
        return model.risk_free_wealth[tdx] == exp_wealth

    instance.risk_free_wealth_decision_constraint = Constraint(
        instance.exp_periods, rule=risk_free_wealth_decision_rule
    )
    print ("min_ms_cvar_eventsp {} risk free wealth decisions constraints OK, "
           "{:.3f} secs".format(param, time() - t1))
    print ("*"*50)
    print ("constructing CVaR constraints")
    t2 = time()

    # constraint
    def cvar_constraint_rule(model, tdx, sdx):
        """
        auxiliary variable Y depends on scenario. CVaR <= VaR
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, scenario index
        sdx2: integer, the descent scenario index of sdx
        """
        wealth = (sum((1. + model.predict_risk_rois[tdx, mdx, sdx]) *
                          model.risk_wealth[tdx, mdx]
                      for mdx in model.symbols) +
                  model.risk_free_wealth[tdx])
        return model.Ys[tdx, sdx] >= (model.Z[tdx] - wealth)

    instance.cvar_constraint = Constraint(
        instance.exp_periods, instance.scenarios,
        rule=cvar_constraint_rule)

    print ("min_ms_cvar_eventsp {} CVaR constraints OK, "
           "{:.3f} secs".format(param, time() - t2))
    print ("*"*50)
    print ("constructing objective rule.")
    t3 = time()

    # objective
    def cvar_objective_rule(model):
        cvar_expr_sum = 0
        for tdx in xrange(n_exp_period):
            scenario_expectation = sum(model.Ys[tdx, sdx] for sdx in
                                       model.scenarios)/model.n_scenario
            cvar_expr = (model.Z[tdx] - scenario_expectation /
                         (1. - model.alpha) -  model.risk_free_wealth[tdx])
            cvar_expr_sum = cvar_expr_sum + cvar_expr
        return cvar_expr_sum/model.n_exp_period

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)
    # solve
    print ("min_ms_cvar_eventsp {} objective OK {:.3f} secs, "
           "start solving:".format(param, time()-t3))

    t4 = time()
    opt = SolverFactory(solver, solver_io=solver_io)
    results = opt.solve(instance, keepfiles=keepfiles)
    instance.solutions.load_from(results)
    if verbose:
        display(instance)

    print ("solve min_ms_cvar_eventsp {} OK {:.2f} secs".format(
        param, time() - t4))
    print ("solver status: {}".format(results.solver.status))
    print ("solver termination cond: {}".format(
        results.solver.termination_condition))
    print (results.solver)

     # extract results
    proxy_buy_pnl = np.zeros((n_exp_period, n_stock, n_scenario))
    proxy_sell_pnl = np.zeros((n_exp_period, n_stock, n_scenario))
    proxy_risk_pnl= np.zeros((n_exp_period, n_stock, n_scenario))
    proxy_risk_free_df = np.zeros((n_exp_period, n_scenario))
    buy_df = np.zeros((n_exp_period, n_stock))
    sell_df = np.zeros((n_exp_period, n_stock))
    risk_df = np.zeros((n_exp_period, n_stock))
    risk_free_arr = np.zeros(n_exp_period)
    var_arr = np.zeros(n_exp_period)

    for tdx in xrange(n_exp_period):
        # shape: (n_exp_period,)
        risk_free_arr[tdx] = instance.risk_free_wealth[tdx].value
        var_arr[tdx] = instance.Z[tdx].value

        for mdx in xrange(n_stock):
            # shape: (n_exp_period, n_stock)
            buy_df[tdx, mdx] = instance.buy_amounts[tdx, mdx].value
            sell_df[tdx, mdx] = instance.sell_amounts[tdx, mdx].value
            risk_df[tdx, mdx] = instance.risk_wealth[tdx, mdx].value

            for sdx in xrange(n_scenario):
                proxy_buy_pnl[tdx, mdx, sdx] = instance.proxy_buy_amounts[
                    tdx, mdx, sdx].value
                proxy_sell_pnl[tdx, mdx, sdx] = instance.proxy_sell_amounts[
                    tdx, mdx, sdx].value
                proxy_risk_pnl[tdx, mdx, sdx] = instance.proxy_risk_wealth[
                    tdx, mdx, sdx].value

        for sdx in xrange(n_scenario):
            proxy_risk_free_df[tdx, sdx] = \
                instance.proxy_risk_free_wealth[tdx, sdx].value
            # proxy_var_df[tdx, sdx] = instance.proxy_Z[tdx, sdx].value

    # shape: (n_exp_period, n_stock, n_scenario)
    proxy_buy_amounts_pnl = pd.Panel(proxy_buy_pnl, items=trans_dates,
                                     major_axis=symbols)
    proxy_sell_amounts_pnl = pd.Panel(proxy_sell_pnl, items=trans_dates,
                                     major_axis=symbols)
    proxy_risk_wealth_pnl = pd.Panel(proxy_risk_pnl, items=trans_dates,
                                     major_axis=symbols)

    # shape: (n_exp_period, n_scenario)
    proxy_risk_free_wealth_df = pd.DataFrame(proxy_risk_free_df,
                                           index=trans_dates)
    # proxy_estimated_var_df =   pd.DataFrame(proxy_var_df,
    #                                        index=trans_dates)

    # shape: (n_exp_period, n_stock)
    buy_amounts_df = pd.DataFrame(buy_df, index=trans_dates,
                                  columns=symbols)
    sell_amounts_df = pd.DataFrame(sell_df, index=trans_dates,
                                   columns=symbols)
    risk_wealth_df = pd.DataFrame(risk_df, index=trans_dates,
                                  columns=symbols)
    # shape: (n_exp_period, )
    risk_free_wealth_arr = pd.Series(risk_free_arr, index=trans_dates)
    estimated_var_arr = pd.Series(var_arr, index=trans_dates)

    Tdx = instance.n_exp_period - 1
    exp_final_wealth = risk_df[Tdx].sum() + risk_free_arr[Tdx]
    print ("{} expected_final_total_wealth: {:.2f}".format(
        param, exp_final_wealth ))

    results = {
        # shape: (n_exp_period, n_stock, n_scenario)
        "proxy_buy_amounts_pnl": proxy_buy_amounts_pnl,
        "proxy_sell_amounts_pnl": proxy_sell_amounts_pnl,
        "proxy_risk_wealth_pnl": proxy_risk_wealth_pnl,

        # shape: (n_exp_period, n_scenario)
        "proxy_risk_free_wealth_df": proxy_risk_free_wealth_df,
        # "proxy_estimated_var_df": proxy_estimated_var_df,

        # shape: (n_exp_period, n_stock)
        "buy_amounts_df": buy_amounts_df,
        "sell_amounts_df": sell_amounts_df,
        "risk_wealth_df": risk_wealth_df,
        # shape: (n_exp_period, )
        "risk_free_wealth_arr": risk_free_wealth_arr,
        "estimated_var_arr": estimated_var_arr,
        # float
        "estimated_cvar": instance.cvar_objective(),
        "expected_final_wealth":exp_final_wealth,
    }
    return results
