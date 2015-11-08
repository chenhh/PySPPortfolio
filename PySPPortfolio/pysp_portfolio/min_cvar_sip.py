# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""


from __future__ import division
import os
from time import time
from datetime import (datetime, date)
import numpy as np
import pandas as pd
import scipy.stats as spstats
from pyomo.environ import *
import matplotlib.pyplot as plt
from pyx.moment_matching import heuristic_moment_matching
from min_cvar_sp import MinCVaRSPPortfolio
from utils import generate_rois_df



def min_cvar_sip_portfolio(symbols, risk_rois, risk_free_roi,
                           allocated_risk_wealth, allocated_risk_free_wealth,
                           buy_trans_fee, sell_trans_fee, alpha,
                           predict_risk_rois, predict_risk_free_roi,
                           n_scenario, max_portfolio_size,
                           scenario_probs=None, solver="cplex",
                           verbose=False):
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

    # aux variable, switching stock variable
    model.chosen = Var(model.symbols, within=Binary)

    # constraint
    def risk_wealth_constraint_rule(model, mdx):
        """
        risk_wealth is a decision variable which depends on both buy_amount and
        sell_amount. i.e. the risk_wealth depends on scenario.

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
        total_sell = sum((1 - sell_trans_fee) * model.sell_amounts[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1 + buy_trans_fee) * model.buy_amounts[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth ==
                (1. + risk_free_roi) * allocated_risk_free_wealth +
                total_sell - total_buy)

    model.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # constraint
    def cvar_constraint_rule(model, sdx):
        """auxiliary variable Y depends on scenario. CVaR <= VaR"""
        wealth = sum((1. + predict_risk_rois.at[mdx, sdx]) *
                     model.risk_wealth[mdx]
                     for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - wealth)

    model.cvar_constraint = Constraint(model.scenarios,
                                       rule=cvar_constraint_rule)

    # constraint
    def chosen_constraint_rule(model, mdx):
        total_wealth = sum(allocated_risk_wealth) + allocated_risk_free_wealth
        return model.risk_wealth[mdx] <= model.chosen[mdx] * total_wealth

    model.chosen_constraint = Constraint(model.symbols,
                                         rule=chosen_constraint_rule)

    # constraint
    def portfolio_size_constraint_rule(model):
        return sum(
            model.chosen[mdx] for mdx in model.symbols) <= max_portfolio_size

    model.portfolio_size_constraint = Constraint(
        rule=portfolio_size_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        scenario_expectation = sum(model.Ys[sdx] * scenario_probs[sdx]
                                    for sdx in xrange(n_scenario))
        return model.Z - 1 / (1 - alpha) * scenario_expectation

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


def test_min_cvar_sip_portfolio():
    from ipro.dev import (STOCK_PKL_DIR, )

    stock_panel = pd.read_pickle(os.path.join(STOCK_PKL_DIR,
                                              "panel_largest50stocks.pkl"))
    symbols = ['2330', '2412', '2882', '2002']
    max_portfolio_size = 3
    n_stock = len(symbols)
    start_date = date(2010, 12, 1)
    n_scenario = 200

    risk_rois = stock_panel.loc[start_date].loc[symbols]['adj_roi'] / 100.
    risk_free_roi = 0
    initial_risk_wealth = pd.Series(len(symbols), index=symbols)
    initial_risk_free_wealth = 1e6
    buy_trans_fee = 0.001425
    sell_trans_fee = 0.004425

    # pre-start_date
    pre_start_date = date(2010, 7, 27)
    pre_risk_rois = stock_panel.loc[pre_start_date:start_date, symbols,
                    'adj_roi'].T / 100.

    print "test hist data:\n"
    print pre_risk_rois

    tgt_moments = np.zeros((n_stock, 4))
    tgt_moments[:, 0] = pre_risk_rois.mean(axis=0)
    tgt_moments[:, 1] = pre_risk_rois.std(axis=0)
    tgt_moments[:, 2] = spstats.skew(pre_risk_rois, axis=0)
    tgt_moments[:, 3] = spstats.kurtosis(pre_risk_rois, axis=0)
    corr_mtx = np.corrcoef(pre_risk_rois.T)

    # shape: (n_stock, n_scenario)
    scenarios = heuristic_moment_matching(tgt_moments, corr_mtx, n_scenario)

    predict_risk_rois = pd.DataFrame(scenarios, index=symbols)

    predict_risk_free_roi = 0
    alpha = 0.95

    results = min_cvar_sip_portfolio(symbols, risk_rois, risk_free_roi,
                                     initial_risk_wealth,
                                     initial_risk_free_wealth, buy_trans_fee,
                                     sell_trans_fee, alpha, predict_risk_rois,
                                     predict_risk_free_roi, n_scenario,
                                     max_portfolio_size,
                                     scenario_probs=None, solver="cplex")
    print results


class MinCVaRSIPPortfolio(MinCVaRSPPortfolio):
    def __init__(self, symbols, max_portfolio_size, risk_rois,
                 risk_free_rois, initial_risk_wealth,
                 initial_risk_free_wealth, buy_trans_fee=0.001425,
                 sell_trans_fee=0.004425, start_date=date(2005, 1, 1),
                 end_date=date(2015, 4, 30), window_length=200,
                 alpha=0.05, n_scenario=200, verbose=False):
        """
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

        self.alpha = alpha
        self.n_scenario = n_scenario
        self.max_portfolio_size = max_portfolio_size

        super(MinCVaRSIPPortfolio, self).__init__(
            symbols, risk_rois, risk_free_rois, initial_risk_wealth,
            initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, verbose)

        self.var_arr = pd.Series(np.zeros(self.n_exp_period),
                                index=self.exp_risk_rois.index)
        self_cvar_arr = pd.Series(np.zeros(self.n_exp_period),
                                index=self.exp_risk_rois.index)

    def valid_specific_parameters(self, *args, **kwargs):
        if self.max_portfolio_size > self.n_stock:
            raise ValueError('the max portfolio size {} > the number of '
                             'stock {} in the candidate set.'.format(
                self.max_portfolio_size, self.n_stock))

        if not (0 <= self.alpha <= 1):
            raise ValueError('error alpha value: {}'.format(self.alpha))

    def get_trading_func_name(self, *args, **kwargs):
        return "MinCVaRSIP_M{}_MC{}_W{}_a{}_s{}".format(
            self.max_portfolio_size, self.n_stock, self.window_length,
            self.alpha, self.n_scenario)

    def add_results_to_reports(self, reports):
        reports['alpha'] = self.alpha
        reports['n_scenario'] = self.n_scenario
        reports['max_portfolio_size'] = self.max_portfolio_size
        reports['var_arr'] = self.var_arr
        reports['cvar_arr'] = self.cvar_arr
        return reports

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """ min_cvar function """
        tdx = kwargs['tdx']
        results = min_cvar_sip_portfolio(
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
            self.max_portfolio_size,
        )

        return results


def run_min_cvar_sip_simulation(max_portfolio_size, window_length, alpha,
                             n_scenario=200):
    """
    :return: reports
    """
    from ipro.dev import (EXP_SYMBOLS, DROPBOX_UP_EXPERIMENT_DIR)

    max_portfolio_size = int(max_portfolio_size)
    window_length = int(window_length)
    alpha = float(alpha)

    symbols = EXP_SYMBOLS
    n_stock = len(symbols)
    risk_rois = generate_rois_df(symbols)
    start_date = date(2005, 1, 1)
    end_date = date(2015, 4, 30)

    exp_risk_rois = risk_rois.loc[start_date:end_date]
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6

    obj = MinCVaRSIPPortfolio(symbols, max_portfolio_size,
                            risk_rois, risk_free_rois,
                            initial_risk_wealth,
                            initial_risk_free_wealth, start_date=start_date,
                            end_date=end_date, window_length=window_length,
                            alpha=alpha, n_scenario=n_scenario, verbose=False)

    reports = obj.run()
    print reports

    file_name = '{}_SIP_{}-{}_m{}_mc{}_w{}_a{:.2f}_s{}.pkl'.format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        exp_risk_rois.index[0].strftime("%Y%m%d"),
        exp_risk_rois.index[-1].strftime("%Y%m%d"),
        max_portfolio_size,
        len(symbols),
        window_length,
        alpha,
        n_scenario)

    file_dir = os.path.join(DROPBOX_UP_EXPERIMENT_DIR, 'cvar_sip')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))

    return reports


if __name__ == '__main__':
    import sys
    import argparse

    sys.path.append(os.path.join(os.path.abspath('..'), '..'))
    parser = argparse.ArgumentParser()

    # test_min_cvar_sp_portfolio()
    parser.add_argument("-m", "--max_portfolio_size", required=True, type=int,
                        choices=range(5, 55, 5))
    parser.add_argument("-w", "--win_length", required=True, type=int)
    parser.add_argument("-a", "--alpha", required=True)
    args = parser.parse_args()
    run_min_cvar_sip_simulation(args.max_portfolio_size, args.win_length,
                             args.alpha)
