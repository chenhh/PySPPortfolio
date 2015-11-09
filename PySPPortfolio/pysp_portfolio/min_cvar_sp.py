# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
from time import time
from datetime import (date,)

import numpy as np
import pandas as pd
import scipy.stats as spstats
from pyomo.environ import *

from scenario.moment_matching import heuristic_moment_matching
from base_model import SPTradingPortfolio


def min_cvar_sp_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth,
                          allocated_risk_free_wealth, buy_trans_fee,
                          sell_trans_fee, alpha, predict_risk_rois,
                          predict_risk_free_roi, n_scenario,
                          scenario_probs=None, solver="cplex", verbose=False):
    """
    two-stage minimize conditional value at risk stochastic programming
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
    model = ConcreteModel()

    # Set
    model.symbols = symbols
    model.scenarios = np.arange(n_scenario)

    # decision variables
    # first stage
    model.buy_amounts = Var(model.symbols, within=NonNegativeReals)
    model.sell_amounts = Var(model.symbols, within=NonNegativeReals)

    # second stage
    model.risk_wealth = Var(model.symbols, within=NonNegativeReals)
    model.risk_free_wealth = Var(within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    model.Z = Var()

    # aux variable, portfolio wealth less than than VaR (Z)
    model.Ys = Var(model.scenarios, within=NonNegativeReals)

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
                (1. + risk_rois[mdx]) * allocated_risk_wealth[mdx] +
                model.buy_amounts[mdx] - model.sell_amounts[mdx])

    model.risk_wealth_constraint = Constraint(
        model.symbols, rule=risk_wealth_constraint_rule)

    # constraint
    def risk_free_wealth_constraint_rule(model):
        total_sell = sum((1. - sell_trans_fee) * model.sell_amounts[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + buy_trans_fee) * model.buy_amounts[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth ==
                (1. + risk_free_roi) * allocated_risk_free_wealth +
                total_sell - total_buy)

    model.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # constraint
    def cvar_constraint_rule(model, sdx):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        wealth = sum((1. + predict_risk_rois.at[mdx, sdx]) *
                     model.risk_wealth[mdx]
                     for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - wealth)

    model.cvar_constraint = Constraint(model.scenarios,
                                       rule=cvar_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        scenario_expectation = sum(model.Ys[sdx] * scenario_probs[sdx]
                                    for sdx in xrange(n_scenario))
        return model.Z - 1. / (1. - alpha) * scenario_expectation

    model.cvar_objective = Objective(rule=cvar_objective_rule, sense=maximize)

    # solve
    opt = SolverFactory(solver)
    instance = model.create()
    results = opt.solve(instance)
    instance.load(results)
    if verbose:
        display(instance)

    # buy and sell amounts
    buy_amounts = pd.Series([instance.buy_amounts[symbol].value
                             for symbol in symbols], index=symbols)
    sell_amounts = pd.Series([instance.sell_amounts[symbol].value
                              for symbol in symbols], index=symbols)

    # value at risk (estimated)
    estimated_var = instance.Z.value

    if verbose:
        print "min_cvar_sp_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "estimated_var": estimated_var,
        "estimated_cvar": model.cvar_objective()
    }


class MinCVaRSPPortfolio(SPTradingPortfolio):
    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth, initial_risk_free_wealth,
                 buy_trans_fee=0.001425, sell_trans_fee=0.004425,
                 start_date=date(2005, 1, 1), end_date=date(2015, 4, 30),
                 window_length=200, alpha=0.05, n_scenario=200,
                 verbose=False):
        """
        Parameters:
         -----------------------
        alpha: float, 0<=value<0.5, 1-alpha is the confidence level of risk
        n_scenario: integer, number of scenarios in a period

        Data:
        -------------
        var_arr: pandas.Series, Value at risk of each period in the simulation
        cvar_arr: pandas.Series, conditional value at risk of each period
        """

        super(MinCVaRSPPortfolio, self).__init__(
           symbols, risk_rois, risk_free_rois, initial_risk_wealth,
           initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, verbose)

        self.alpha = alpha
        self.n_scenario = int(n_scenario)
        self.var_arr = pd.Series(np.zeros(self.n_exp_period),
                                index=self.exp_risk_rois.index)
        self.cvar_arr = pd.Series(np.zeros(self.n_exp_period),
                                  index = self.exp_risk_rois.index)

    def get_trading_func_name(self, *args, **kwargs):
        return "MinCVaRSP_M{}_W{}_a{}_s{}".format(
            self.n_stock, self.window_length, self.alpha, self.n_scenario)

    def add_results_to_reports(self, reports):
        """ add additional items to reports """
        reports['alpha'] = self.alpha
        reports['n_scenario'] = self.n_scenario
        reports['var_arr'] = self.var_arr
        reports['cvar_arr'] = self.cvar_arr
        return reports

    def get_estimated_risk_free_roi(self, *arg, **kwargs):
        """the risk free roi is set all zeros"""
        return 0.

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        heuristic moment matching

        Returns:
        -----------
        estimated_risk_rois, numpy.array, shape: (n_stock, n_scenario)
        """
        # current index in the exp_period
        tdx = kwargs['tdx']
        hist_end_idx = self.start_date_idx + tdx
        hist_start_idx = self.start_date_idx + tdx - self.window_length

        # shape: (window_length, n_stock), index slicing should plus 1
        hist_data = self.risk_rois.iloc[hist_start_idx:hist_end_idx+1]
        if self.verbose:
            print "HMM current: {} hist_data:[{}-{}]".format(
                                self.exp_risk_rois.index[tdx],
                                self.risk_rois.index[hist_start_idx],
                                self.risk_rois.index[hist_end_idx])

        # 1-4 th moments of historical data, shape: (n_stock, 4)
        tgt_moments = np.zeros((self.n_stock, 4))
        tgt_moments[:, 0] = hist_data.mean(axis=0)
        # the 2nd moment must be standard deviation, not the variance
        tgt_moments[:, 1] = hist_data.std(axis=0)
        tgt_moments[:, 2] = spstats.skew(hist_data, axis=0)
        tgt_moments[:, 3] = spstats.kurtosis(hist_data, axis=0)
        corr_mtx = np.corrcoef(hist_data.T)

        # scenarios shape: (n_stock, n_scenario)
        for idx, error_order in enumerate(xrange(-3, 0)):
            # if the HMM is not converge, relax the tolerance error
            try:
                max_moment_err = 10**error_order
                max_corr_err = 10**error_order
                scenarios = heuristic_moment_matching(
                                tgt_moments, corr_mtx, self.n_scenario,
                                max_moment_err, max_corr_err)
                break
            except ValueError as e:
                print e
                if idx >= 2:
                    raise ValueError('HMM not converge.')

        return pd.DataFrame(scenarios, index=self.symbols)


    def set_specific_period_action(self, *args, **kwargs):
        """
        user specified action after getting results
        """
        tdx = kwargs['tdx']
        results = kwargs['results']
        self.var_arr.iloc[tdx] = results["estimated_var"]
        self.cvar_arr.iloc[tdx] = results['estimated_cvar']


    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """ min_cvar function """

        # current exp_period index
        tdx = kwargs['tdx']
        results = min_cvar_sp_portfolio(
            self.symbols,
            self.exp_risk_rois.iloc[tdx, :],
            self.risk_free_rois.iloc[tdx],
            kwargs['allocated_risk_wealth'],
            kwargs['allocated_risk_free_wealth'],
            self.buy_trans_fee,
            self.sell_trans_fee,
            self.alpha,
            kwargs['estimated_risk_rois'],
            kwargs['estimated_risk_free_roi'],
            self.n_scenario,
        )
        return results


def all_scenarios_min_cvar_sp_portfolio(symbols, risk_rois, risk_free_rois,
                          allocated_risk_wealth,
                          allocated_risk_free_wealth, buy_trans_fee,
                          sell_trans_fee, alpha, predict_risk_rois,
                          predict_risk_free_roi, n_scenario,
                          scenario_probs=None, solver="cplex", verbose=False
    ):
    """
    after generating all scenarios, solving the SP at once
    symbols: list of string
    risk_rois: pandas.DataFrame, shape: (n_exp_period, n_stock)
    risk_free_rois: pandas.Series, shape: (n_exp_period,)
    allocated_risk_wealth: pandas.Series, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, 1-alpha is the significant level
    predict_risk_ret: pandas.Panel, shape: (n_exp_period, n_stock, n_scenario)
    predict_risk_free_roi: float
    n_scenario: integer
    scenario_probs: numpy.array, shape: (n_scenario,)
    solver: str, supported by Pyomo

    """
    t0 = time()
    if scenario_probs is None:
        scenario_probs = np.ones(n_scenario, dtype=np.float) / n_scenario

    n_exp_period = risk_rois.shape[0]

    # concrete model
    instance = ConcreteModel(name="all_scenarios_min_cvar_sp_portfolio")

    # Set
    instance.exp_periods = np.arange(n_exp_period)
    instance.symbols = symbols
    instance.scenarios = np.arange(n_scenario)

    # decision variables
    # in each period, we buy or sell stock
    instance.buy_amounts = Var(instance.exp_periods, instance.symbols,
                            within=NonNegativeReals)
    instance.sell_amounts = Var(instance.exp_periods, instance.symbols,
                             within=NonNegativeReals)

    instance.risk_wealth = Var(instance.exp_periods, instance.symbols,
                            within=NonNegativeReals)
    instance.risk_free_wealth = Var(instance.exp_periods,
                                 within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    instance.Z = Var(instance.exp_periods, within=Reals)

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Ys = Var(instance.exp_periods, instance.scenarios,
                   within=NonNegativeReals)

    # constraint
    def risk_wealth_constraint_rule(model, tdx, mdx):
        """
        Parameters:
        ------------
        tdx: integer, time index of period
        mdx: str, symbol
        """
        if tdx == 0:
            return (
                model.risk_wealth[tdx, mdx] ==
                (1. + risk_rois[tdx, mdx]) * allocated_risk_wealth[mdx] +
                model.buy_amounts[tdx, mdx] - model.sell_amounts[tdx, mdx]
            )
        else:
            return (
                model.risk_wealth[tdx, mdx] ==
                (1. + risk_rois[tdx, mdx]) * model.risk_wealth[tdx-1, mdx] +
                model.buy_amounts[tdx, mdx] - model.sell_amounts[tdx, mdx]
            )

    instance.risk_wealth_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=risk_wealth_constraint_rule)

    # constraint
    def risk_free_wealth_constraint_rule(model, tdx):
        """
        Parameters:
        ------------
        tdx: integer, time index of period
        """
        total_sell = sum((1. - sell_trans_fee) * model.sell_amounts[tdx, mdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + buy_trans_fee) * model.buy_amounts[tdx, mdx]
                        for mdx in model.symbols)
        if tdx == 0:
            return (
                model.risk_free_wealth[tdx] ==
                (1. + risk_free_rois[tdx]) * allocated_risk_free_wealth +
                total_sell - total_buy
            )
        else:
            return (
                model.risk_free_wealth[tdx] ==
                (1. + risk_free_rois[tdx]) * model.risk_free_wealth[tdx-1] +
                total_sell - total_buy
            )

    instance.risk_free_wealth_constraint = Constraint(
         instance.exp_periods, rule=risk_free_wealth_constraint_rule)

    # constraint
    def cvar_constraint_rule(model, tdx, sdx):
        """
        auxiliary variable Y depends on scenario. CVaR <= VaR
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, scenario index
        """
        wealth = sum((1. + predict_risk_rois[tdx, mdx, sdx]) *
                     model.risk_wealth[tdx, mdx]
                     for mdx in model.symbols)
        return model.Ys[tdx, sdx] >= (model.Z[tdx] - wealth)

    instance.cvar_constraint = Constraint(
        instance.exp_periods, instance.scenarios,
        rule=cvar_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        edx = n_exp_period -1
        scenario_expectation = sum(model.Ys[edx, sdx] * scenario_probs[sdx]
                                    for sdx in xrange(n_scenario))
        return model.Z[edx] - 1. / (1. - alpha) * scenario_expectation

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    # display(instance)

    # extract results
    edx = n_exp_period -1
    final_wealth = sum(instance.risk_wealth[edx, mdx].value
                       for mdx in symbols)
    final_wealth += + instance.risk_free_wealth[edx].value
    roi = final_wealth/allocated_risk_free_wealth -1
    print "final_wealth:{}, ROI:{:.4%}".format(final_wealth, roi)
