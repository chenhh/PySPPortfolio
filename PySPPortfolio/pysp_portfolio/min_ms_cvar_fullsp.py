# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from __future__ import division
from time import time
from datetime import date
import os
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition

from PySPPortfolio.pysp_portfolio import *
from base_model import (SPTradingPortfolio, )


def min_ms_cvar_fullsp_portfolio(symbols, trans_dates, risk_rois,
                                 risk_free_rois, allocated_risk_wealth,
                                 allocated_risk_free_wealth, buy_trans_fee,
                                 sell_trans_fee, alpha, predict_risk_rois,
                                 predict_risk_free_roi, n_scenario,
                                 solver=DEFAULT_SOLVER, verbose=False):
    """
    after generating all scenarios, solving the SP at once

    symbols: list of string
    risk_rois: numpy.array, shape: (n_exp_period, n_stock)
    risk_free_rois: float
    allocated_risk_wealth: numpy.array,, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float
    predict_risk_ret: numpy.array, shape: (n_exp_period, n_stock, n_scenario)
    predict_risk_free_rois: float
    n_scenario: integer
    solver: str, supported by Pyomo

    """
    t0 = time()

    n_exp_period = risk_rois.shape[0]
    if n_exp_period > 3:
        raise ValueError("we can only solve 3-stage full sp.")
    n_stock = len(symbols)

    param = "{}_{}_m{}_p{}_s{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock, n_exp_period, n_scenario, alpha)
    print ("min_ms_cvar_sp {} objective ready to construct.".format(param))

    # concrete model
    instance = ConcreteModel(name="ms_min_cvar_fullsp_portfolio")

    # data
    instance.risk_rois = risk_rois
    instance.risk_free_rois = risk_free_rois
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alpha = alpha
    instance.predict_risk_rois = predict_risk_rois
    instance.predict_risk_free_roi = predict_risk_free_roi

    # Set
    instance.n_exp_period = n_exp_period
    instance.exp_periods = np.arange(n_exp_period)
    instance.symbols = np.arange(n_stock)
    instance.scenarios = np.arange(n_scenario)
    instance.n_scenario = n_scenario

    # stage 1 decision variables
    instance.buy_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.sell_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_wealth = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_free_wealth = Var(within=NonNegativeReals)
    instance.Z = Var(within=Reals)
    instance.predict_portfolio_wealth = Var(instance.scenarios,
                                       within=NonNegativeReals)
    instance.Ys = Var(instance.scenarios, within=NonNegativeReals)

    # stage 2 decision variables
    instance.buy_amounts2 = Var(instance.symbols, instance.scenarios,
                                within=NonNegativeReals)
    instance.sell_amounts2 = Var(instance.symbols,
                                 instance.scenarios, within=NonNegativeReals)
    instance.risk_wealth2 = Var(instance.symbols,
                                instance.scenarios, within=NonNegativeReals)
    instance.risk_free_wealth2 = Var(instance.scenarios,
                                     within=NonNegativeReals)
    instance.Z2 = Var(instance.scenarios, within=Reals)
    instance.predict_portfolio_wealth2 = Var(
        instance.scenarios, instance.scenarios, within=NonNegativeReals)
    instance.Ys2 = Var(instance.scenarios, instance.scenarios,
                       within=NonNegativeReals)

    # stage 1 constraint
    def risk_wealth_constraint_rule(model, mdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        prev_risk_wealth = model.allocated_risk_wealth[mdx]
        risk_roi = model.risk_rois[0, mdx]

        return (model.risk_wealth[mdx] ==
                (1. + risk_roi) * prev_risk_wealth +
                model.buy_amounts[mdx] - model.sell_amounts[mdx]
                )

    instance.risk_wealth_constraint = Constraint(
        instance.symbols, rule=risk_wealth_constraint_rule)

    # stage 1 constraint
    def risk_free_wealth_constraint_rule(model):
        """
        Parameters:
        ------------
        tdx: integer, time index of period
        """
        total_sell = sum((1. - model.sell_trans_fee) *
                         model.sell_amounts[mdx] for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) *
                        model.buy_amounts[mdx] for mdx in model.symbols)
        prev_risk_free_wealth = model.allocated_risk_free_wealth

        return (model.risk_free_wealth ==
                (1. + model.risk_free_rois) * prev_risk_free_wealth +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # stage 1 constraint
    def predict_portfolio_wealth_rule(model, sdx):
        portfolio_wealth = sum((1. + model.predict_risk_rois[0, mdx, sdx]) *
                          model.risk_wealth[mdx]
                          for mdx in model.symbols)
        return (model.predict_portfolio_wealth[sdx] == portfolio_wealth)

    instance.predict_portfolio_wealth_constraint = Constraint(
        instance.scenarios, rule=predict_portfolio_wealth_rule
    )

    # stage 1 constraint
    def cvar_constraint_rule(model, sdx):
        """
        auxiliary variable Y depends on scenario. CVaR <= VaR
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, scenario index
        """
        return model.Ys[sdx] >= (model.Z - model.predict_portfolio_wealth[sdx])

    instance.cvar_constraint = Constraint(
        instance.scenarios, rule=cvar_constraint_rule)

    # stage 2 constraint
    def risk_wealth2_constraint_rule(model, mdx, sdx):
        prev_risk_wealth = model.risk_wealth[mdx]
        risk_roi = model.predict_risk_rois[0, mdx, sdx]

        return (model.risk_wealth2[mdx, sdx] ==
                (1. + risk_roi) * prev_risk_wealth +
                model.buy_amounts2[mdx, sdx] - model.sell_amounts2[mdx, sdx]
                )

    instance.risk_wealth2_constraint = Constraint(
        instance.symbols, instance.scenarios, rule=risk_wealth2_constraint_rule)

    # stage 2 constraint
    def risk_free_wealth2_constraint_rule(model, sdx):
        """
        Parameters:
        ------------
        tdx: integer, time index of period
        """
        total_sell = sum((1. - model.sell_trans_fee) *
                         model.sell_amounts2[mdx, sdx] for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) *
                        model.buy_amounts2[mdx, sdx] for mdx in model.symbols)
        prev_risk_free_wealth = model.risk_free_wealth

        return (model.risk_free_wealth2[sdx] ==
                (1. + model.risk_free_rois) * prev_risk_free_wealth +
                total_sell - total_buy)

    instance.risk_free_wealth2_constraint = Constraint(
        instance.scenarios, rule=risk_free_wealth2_constraint_rule)

    # stage 2 constraint
    def predict_portfolio_wealth2_rule(model, sdx, sdx2):
        portfolio_wealth = sum((1. + model.predict_risk_rois[1, mdx, sdx2]) *
                               model.risk_wealth2[mdx, sdx]
                            for mdx in model.symbols)
        return (model.predict_portfolio_wealth2[sdx, sdx2] == portfolio_wealth)

    instance.predict_portfolio_wealth2_constraint = Constraint(
        instance.scenarios, instance.scenarios,
        rule=predict_portfolio_wealth2_rule
    )

    # stage 2 constraint
    def cvar2_constraint_rule(model, sdx, sdx2):
        """
        auxiliary variable Y depends on scenario. CVaR <= VaR
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, scenario index
        """
        return model.Ys2[sdx, sdx2] >= (model.Z2[sdx] -
        model.predict_portfolio_wealth2[sdx, sdx2])

    instance.cvar2_constraint = Constraint(
        instance.scenarios, instance.scenarios,
        rule=cvar2_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        s1_exp = (sum(model.Ys[sdx] for sdx in xrange(n_scenario)) /
                  float( n_scenario))
        cvar1 = (model.Z - s1_exp /(1. - model.alpha))

        exp_cvar2 = 0
        for sdx in xrange(n_scenario):
            s2_exp = (sum(model.Ys2[sdx, sdx2] for sdx2 in xrange(n_scenario)))
            cvar2 = (model.Z2[sdx] - s2_exp/(1. -
                                             model.alpha))/n_scenario/n_scenario
            exp_cvar2 += cvar2

        return cvar1 + exp_cvar2

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                            sense=maximize)
    # solve
    print "start solving:"
    t1 = time()
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)

    print ("solve min_ms_cvar_fullsp {} OK {:.2f} secs".format(
            param, time() - t1))
    print ("solver status: {}".format(results.solver.status))
    print ("solver termination cond: {}".format(
        results.solver.termination_condition))
    print (results.solver)

    print ("Objective: {}".format(instance.cvar_objective()))
    print ("VaR[0]: {}".format(instance.Z.value))
    for tdx in xrange(n_scenario):
        print ("VaR[1, {}]: {}".format(tdx, instance.Z2[tdx].value))

    for mdx in xrange(n_stock):
        print ("buy amounts:{}".format(instance.buy_amounts[mdx].value))
        print ("sell amounts:{}".format(instance.sell_amounts[mdx].value))
        print ("risk_wealth:{}".format(instance.risk_wealth[mdx].value))


def run_min_ms_cvar_fullsp_simulation(n_stock, win_length,
                                      start_date, end_date,
                                      n_scenario=200,
                                     bias=False, scenario_cnt=1, alpha=0.95,
                                     verbose=False):
    """
    multi-stage average scenario SP simulation
    the results are independent to the alphas

    Parameters:
    -------------------
    n_stock: integer, number of stocks of the EXP_SYMBOLS to the portfolios
    window_length: integer, number of periods for estimating scenarios
    n_scenario, int, number of scenarios
    bias: bool, biased moment estimators or not
    scenario_cnt: count of generated scenarios, default = 1
    alpha: float, for conditional risk
    Returns:
    --------------------
    reports
    """
    t0 = time()
    n_stock, win_length, = int(n_stock), int(win_length)
    n_scenario = int(n_scenario)

    # getting experiment symbols
    symbols = EXP_SYMBOLS[:n_stock]
    param = "{}_{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock, win_length, n_scenario, "biased" if bias else "unbiased",
        scenario_cnt, alpha)

    # read rois panel
    roi_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')
    if not os.path.exists(roi_path):
        raise ValueError("{} roi panel does not exist.".format(roi_path))

    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # shape: (n_period, n_stock)
    exp_risk_rois = roi_panel.loc[start_date:end_date, symbols,
                    'simple_roi'].T
    n_period = exp_risk_rois.shape[0]
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 100.

    scenario_name = "{}_{}_m{}_w{}_s{}_{}_{}.pkl".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        len(symbols), win_length, n_scenario,
        "biased" if bias else "unbiased", scenario_cnt)

    scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios',
                                 scenario_name)
    scenario_panel = pd.read_pickle(scenario_path)

    min_ms_cvar_fullsp_portfolio(symbols, exp_risk_rois.index,
                                 exp_risk_rois.as_matrix(),
                                 0,
                                 np.zeros(n_stock),
                                 initial_risk_free_wealth,
                                 BUY_TRANS_FEE,
                                 SELL_TRANS_FEE, alpha,
                                 scenario_panel.as_matrix(),
                                 0,
                                 n_scenario
                                 )
    print ("min_ms_cvar_fullsp_portfolio:{} {} secs".format(param, time()-t0))

if __name__ == '__main__':
    run_min_ms_cvar_fullsp_simulation(5, 70, date(2005,1,3), date(2005,1,5),
                                      alpha=0.9)
