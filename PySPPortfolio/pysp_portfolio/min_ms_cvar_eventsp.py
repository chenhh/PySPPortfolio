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
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from PySPPortfolio.pysp_portfolio import *
from base_model import (SPTradingPortfolio, )

def min_ms_cvar_eventsp_portfolio(symbols, trans_dates, risk_rois,
                                risk_free_rois, allocated_risk_wealth,
                                allocated_risk_free_wealth, buy_trans_fee,
                                sell_trans_fee, alpha, predict_risk_rois,
                                predict_risk_free_roi, n_scenario=200,
                                solver = DEFAULT_SOLVER, verbose=False):
    """
    in each period, when the decision branchs, using the expected decisions.

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
    """

    t0 = time()
    n_exp_period = risk_rois.shape[0]
    n_stock = len(symbols)
    print ("transaction dates: {}-{}".format(trans_dates[0], trans_dates[-1]))
    # concrete model
    instance = ConcreteModel(name="ms_min_cvar_eventsp_portfolio")
    param = "{}_{}_m{}_p{}_s{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock, n_exp_period, n_scenario, alpha)

    # data
    instance.risk_rois = risk_rois
    instance.risk_free_rois = risk_free_rois
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alpha = alpha
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
    instance.buy_amounts = Var(instance.exp_periods, instance.symbols,
                               within=NonNegativeReals)
    instance.sell_amounts = Var(instance.exp_periods, instance.symbols,
                                within=NonNegativeReals)

    instance.proxy_buy_amounts = Var(instance.exp_periods, instance.symbols,
                                 instance.scenarios, within=NonNegativeReals)
    instance.proxy_sell_amounts = Var(instance.exp_periods, instance.symbols,
                                  instance.scenarios, within=NonNegativeReals)

    # shape: (n_exp_period, n_stock)
    instance.risk_wealth = Var(instance.exp_periods, instance.symbols,
                               within=NonNegativeReals)
    instance.proxy_risk_wealth = Var(instance.exp_periods, instance.symbols,
                                instance.scenarios, within=NonNegativeReals)
    # shape: (n_exp_period, )
    instance.risk_free_wealth = Var(instance.exp_periods,
                                    within=NonNegativeReals)
    instance.proxy_risk_free_wealth = Var(instance.exp_periods,
                                  instance.scenarios, within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    # shape: (n_exp_period, )
    instance.Z = Var(instance.exp_periods, within=Reals)
    instance.proxy_Z = Var(instance.exp_periods,
                           instance.scenarios, within=Reals)

    # aux variable, portfolio wealth less than than VaR (Z)
    # in each stage, there is only one scneario,
    # shape: (n_exp_period, ),
    instance.Ys = Var(instance.exp_periods, instance.scenarios,
                      within=NonNegativeReals)
    instance.proxy_Ys = Var(instance.exp_periods, instance.scenarios,
                            instance.scenarios, within=NonNegativeReals)

    print ("combinations (exp_period, stock, scenarios)=({}, {}, {})".format(
        n_exp_period, n_stock, n_scenario))
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
            prev_risk_wealth = model.risk_wealth[tdx - 1, mdx]
            risk_roi = model.predict_risk_rois[tdx, mdx, sdx]

        return (model.proxy_risk_wealth[tdx, mdx, sdx] ==
                (1. + risk_roi) * prev_risk_wealth +
                model.proxy_buy_amounts[tdx, mdx, sdx] -
                model.proxy_sell_amounts[tdx, mdx, sdx]
                )

    instance.risk_wealth_constraint = Constraint(
        instance.exp_periods, instance.symbols, instance.scenarios,
        rule=risk_wealth_constraint_rule)

    # risk wealth constraint
    def risk_wealth_decision_rule(model, tdx, mdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        exp_wealth = sum(model.proxy_risk_wealth[tdx, mdx, sdx]
                         for sdx in model.scenarios) / model.n_scenario
        return model.risk_wealth[tdx, mdx] == exp_wealth

    instnace.risk_wealth_decision_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=risk_wealth_decision_rule
    )

    # buy amount constraint
    def buy_decision_rule(model, tdx, mdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        exp_buy = sum(model.proxy_buy_amounts[tdx, mdx, sdx]
                         for sdx in model.scenarios) / model.n_scenario
        return model.buy_amounts[tdx, mdx] == exp_buy

    instnace.buy_decision_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=buy_decision_rule
    )

    # buy amount constraint
    def sell_decision_rule(model, tdx, mdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        exp_sell = sum(model.proxy_sell_amounts[tdx, mdx, sdx]
                      for sdx in model.scenarios) / model.n_scenario
        return model.sell_amounts[tdx, mdx] == exp_sell

    instnace.sell_decision_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=sell_decision_rule
    )
    print ("min_ms_cvar_eventsp {} risk wealth decisions constraints OK, "
           "{:.3f} secs".format(param, time() - t0))
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
                (1. + model.predict_risk_free_roi[tdx]) *
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

    instnace.risk_free_wealth_decision_constraint = Constraint(
        instance.exp_periods, rule=risk_free_wealth_decision_rule
    )
    print ("min_ms_cvar_eventsp {} risk free wealth decisions constraints OK, "
           "{:.3f} secs".format(param, time() - t1))
    print ("constructing cvar constraints")
    t2 = time()

    # constraint
    def cvar_constraint_rule(model, tdx, sdx, sdx2):
        """
        auxiliary variable Y depends on scenario. CVaR <= VaR
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, scenario index
        """
        risk_wealth = sum((1. + model.predict_risk_rois[tdx, mdx, sdx]) *
                          model.proxy_risk_wealth[tdx, mdx, sdx]
                          for mdx in model.symbols)
        return model.proxy_Ys[tdx, sdx, sdx2] >= (model.proxy_Z[tdx, sdx] -
                                            risk_wealth)

    instance.cvar_constraint = Constraint(
        instance.exp_periods, rule=cvar_constraint_rule)

    def z_decision_rule(model, tdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        """
        exp_Z = (sum(model.proxy_Z[tdx, sdx]
                     for sdx in model.scenarios) / model.n_scenario)
        return model.Z[tdx] == exp_Z

    instance.z_decision_constraint = Constraint(
        instance.exp_periods, within=z_decision_rule
    )

    def y_decision_rule(model, tdx, sdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        sdx: integer, index of scenario
        """
        exp_y = (sum(model.proxy_Ys[tdx, sdx, sdx2]
                     for sdx2 in model.scenarios) / model.n_scenario)
        return model.Ys[tdx, sdx] == exp_y

    instance.y_decision_constraint = Constraint(
        instance.exp_periods, instance.scenarios, within=y_decision_rule
    )
    print ("min_ms_cvar_eventsp {} cvar constraints OK, "
           "{:.3f} secs".format(param, time() - t2))
    print ("constructing objective rule")
    t3 = time()

    # objective
    def cvar_objective_rule(model):
        cvar_expr_sum = 0
        for tdx in xrange(n_exp_period):
            scenario_expectation = sum(model.Ys[tdx, sdx] for sdx in
                                       model.scenarios)/model.n_scenario
            cvar_expr = (model.Z[tdx] - scenario_expectation /
                         (1. - model.alpha))
            cvar_expr_sum = cvar_expr_sum + cvar_expr
        return cvar_expr_sum

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)
    # solve
    print ("min_ms_cvar_eventsp {} objective OK {:.3f} secs, "
           "start solving:".format(param, time()-t3))

    t4 = time()
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    if verbose:
        display(instance)

    print ("solve min_ms_cvar_eventsp {} OK {:.2f} secs".format(
        param, time() - t4))
    print ("solver status: {}".format(results.solver.status))
    print ("solver termination cond: {}".format(
        results.solver.termination_condition))
    print (results.solver)

