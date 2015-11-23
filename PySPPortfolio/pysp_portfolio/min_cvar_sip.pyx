# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: nonecheck=False

"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
import os
from time import time
from PySPPortfolio.pysp_portfolio import *

import numpy as np
import pandas as pd
from pyomo.environ import *
from min_cvar_sp import MinCVaRSPPortfolio

cimport numpy as cnp

ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INTP_t

def min_cvar_sip_portfolio(symbols,
                           cnp.ndarray[FLOAT_t, ndim=1] risk_rois,
                           double risk_free_roi,
                           cnp.ndarray[FLOAT_t, ndim=1] allocated_risk_wealth,
                           double allocated_risk_free_wealth,
                           double buy_trans_fee,
                           double sell_trans_fee,
                           double alpha,
                           cnp.ndarray[FLOAT_t, ndim=2] predict_risk_rois,
                           double predict_risk_free_roi,
                           int n_scenario,
                           int max_portfolio_size,
                           scenario_probs=None,
                           str solver=DEFAULT_SOLVER,
                           int verbose=False):
    """
    two stage minimize conditional value at risk stochastic programming
    portfolio

    symbols: list of string
    risk_rois: numpy.array, shape: (n_stock, )
    risk_free_roi: float,
    initial_risk_wealth: numpy.array, shape: (n_stock,)
    initial_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, significant level
    predict_risk_ret: numpy.array, shape: (n_scenario, n_stock)
    predict_risk_free_roi: float
    n_scenario: integer
    scenario_probs: numpy.array, shape: (n_scenario,)
    solver: str, supported by Pyomo
    :return:
    """
    t0 = time()
    if scenario_probs is None:
        scenario_probs = np.ones(n_scenario, dtype=np.float) / n_scenario

    # concrete model
    instance = ConcreteModel()

    # data
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
    instance.max_portfolio_size = max_portfolio_size


    # Set
    cdef Py_ssize_t n_stock = len(symbols)
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

    # aux variable, switching stock variable
    instance.chosen = Var(instance.symbols, within=Binary)

    # constraint
    def risk_wealth_constraint_rule(model, int mdx):
        """
        risk_wealth is a decision variable which depends on both buy_amount and
        sell_amount. i.e. the risk_wealth depends on scenario.

        buy_amount and sell_amount are first stage variable,
        risk_wealth is second stage variable.
        """
        return (model.risk_wealth[mdx] ==
                (1. + model.risk_rois[mdx]) *
                model.allocated_risk_wealth[mdx] +
                model.buy_amounts[mdx] - model.sell_amounts[mdx])

    instance.risk_wealth_constraint = Constraint(
        instance.symbols, rule=risk_wealth_constraint_rule)

    # constraint
    def risk_free_wealth_constraint_rule(model):
        total_sell = sum((1 - model.sell_trans_fee) * model.sell_amounts[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1 + model.buy_trans_fee) * model.buy_amounts[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth ==
                (1. + model.risk_free_roi) * model.allocated_risk_free_wealth +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # constraint
    def cvar_constraint_rule(model, int sdx):
        """auxiliary variable Y depends on scenario. CVaR <= VaR"""
        wealth = sum((1. + model.predict_risk_rois[mdx, sdx]) *
                     model.risk_wealth[mdx]
                     for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - wealth)

    instance.cvar_constraint = Constraint(instance.scenarios,
                                          rule=cvar_constraint_rule)

    # constraint
    def chosen_constraint_rule(model, int mdx):
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
        scenario_expectation = sum(model.Ys[sdx] * model.scenario_probs[sdx]
                                   for sdx in xrange(n_scenario))
        return model.Z - 1. / (1. - alpha) * scenario_expectation

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    if verbose:
        display(instance)

    # buy and sell amounts
    buy_amounts = pd.Series([instance.buy_amounts[mdx].value
                             for mdx in xrange(n_stock)], index=symbols)
    sell_amounts = pd.Series([instance.sell_amounts[mdx].value
                              for mdx in xrange(n_stock)], index=symbols)

    # chosen stock
    chosen_symbols = pd.Series([instance.chosen[mdx].value
                                for mdx in xrange(n_stock)], index=symbols)

    # value at risk (estimated)
    estimated_var = instance.Z.value

    if verbose:
        print "min_cvar_sp_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "estimated_var": estimated_var,
        "estimated_cvar": instance.cvar_objective(),
        "chosen_symbols": chosen_symbols,
    }


class MinCVaRSIPPortfolio(MinCVaRSPPortfolio):
    def __init__(self, candidate_symbols,
                 int max_portfolio_size, risk_rois,
                 risk_free_rois, initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=BUY_TRANS_FEE,
                 double sell_trans_fee=SELL_TRANS_FEE, start_date=START_DATE,
                 end_date=END_DATE,
                 int window_length=WINDOW_LENGTH,
                 int n_scenario=N_SCENARIO, bias=BIAS_ESTIMATOR,
                 float alpha=0.05,
                 int scenario_cnt=1, verbose=False):
        """
        the n_stock in SIP model represents the size of candidate stocks,
        not the portfolio size.

        Parameters:
         -----------------------
        max_portfolio_size: integer, maximum number of stocks in the portfolio
        alpha: float, 0<=value<0.5, 1-alpha is the confidence level of risk
        n_scenario: integer, number of scenarios in a period

        Data:
        -------------
        var_arr: pandas.Series, Value at risk of each period in the simulation
        cvar_arr: pandas.Series, conditional value at risk of each period
        """

        self.max_portfolio_size = int(max_portfolio_size)

        super(MinCVaRSIPPortfolio, self).__init__(
            candidate_symbols, risk_rois, risk_free_rois, initial_risk_wealth,
            initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, n_scenario, bias,
            alpha, scenario_cnt, verbose)

        assert self.n_stock == 50

        # overwrite scenario panel, load 50 stocks
        scenario_name = "{}_{}_m{}_w{}_s{}_{}_{}.pkl".format(
            start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"),
            self.n_stock, window_length,
            n_scenario, "biased" if bias else "unbiased", scenario_cnt)

        scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios',
                                     scenario_name)

        if not os.path.exists(scenario_path):
            raise ValueError("{} not exists.".format(scenario_name))
            self.scenario_panel = None
            self.scenario_cnt = 0
        else:
            self.scenario_panel = pd.read_pickle(scenario_path)
            self.scenario_cnt = scenario_cnt

        self.chosen_symbols_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index, columns=candidate_symbols)

    def valid_specific_parameters(self, *args, **kwargs):
        if self.max_portfolio_size > self.n_stock:
            raise ValueError('the max portfolio size {} > the number of '
                             'stock {} in the candidate set.'.format(
                self.max_portfolio_size, self.n_stock))

    def get_trading_func_name(self, *args, **kwargs):
        return "MinCVaRSIP_all{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            self.n_stock, self.max_portfolio_size,
            self.window_length,
            self.n_scenario, "biased" if self.bias_estimator else "unbiased",
            self.scenario_cnt, self.alpha)

    def add_results_to_reports(self, reports, *args, **kwargs):
        reports['alpha'] = self.alpha
        reports['scenario_cnt'] = self.scenario_cnt
        reports['max_portfolio_size'] = self.max_portfolio_size
        reports['var_arr'] = self.var_arr
        reports['cvar_arr'] = self.cvar_arr
        reports['chosen_symbols_df'] = self.chosen_symbols_df
        return reports

    def set_specific_period_action(self, *args, **kwargs):
        """
        user specified action after getting results
        """
        tdx = kwargs['tdx']
        results = kwargs['results']
        self.var_arr.iloc[tdx] = results["estimated_var"]
        self.cvar_arr.iloc[tdx] = results['estimated_cvar']
        self.chosen_symbols_df.iloc[tdx] = results['chosen_symbols']

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """ min_cvar function """
        tdx = kwargs['tdx']
        results = min_cvar_sip_portfolio(
            self.symbols,
            self.exp_risk_rois.iloc[tdx, :].as_matrix(),
            self.risk_free_rois.iloc[tdx],
            kwargs['allocated_risk_wealth'].as_matrix(),
            kwargs['allocated_risk_free_wealth'],
            self.buy_trans_fee,
            self.sell_trans_fee,
            self.alpha,
            kwargs['estimated_risk_rois'].as_matrix(),
            kwargs['estimated_risk_free_roi'],
            self.n_scenario,
            self.max_portfolio_size,
        )
        return results
