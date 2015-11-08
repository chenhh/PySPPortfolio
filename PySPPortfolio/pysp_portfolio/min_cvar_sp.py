# -*- coding: utf-8 -*-
"""
.. codeauthor:: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
stochastic programming
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
# from moment_matching import heuristic_moment_matching
from pyx.moment_matching import heuristic_moment_matching
from base import SPTradingPortfolio
from utils import generate_rois_df

__author__ = 'Hung-Hsin Chen'


def min_cvar_sp_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth,
                          allocated_risk_free_wealth, buy_trans_fee,
                          sell_trans_fee, alpha, predict_risk_rois,
                          predict_risk_free_roi, n_scenario,
                          scenario_probs=None, solver="cplex", verbose=False):
    """
    two-stage minimize conditional value at risk stochastic programming
    portfolio

    :param symbols: list of string
    :param risk_rois: pandas.Series, shape: (n_stock, )
    :param risk_free_roi: float,
    :param allocated_risk_wealth: pandas.Series, shape: (n_stock,)
    :param allocated_risk_free_wealth: float
    :param buy_trans_fee: float
    :param sell_trans_fee: float
    :param alpha: float, significant level
    :param predict_risk_ret: pandas.DataFrame, shape: (n_stock, n_scenario)
    :param predict_risk_free_roi: float
    :param n_scenario: integer
    :param scenario_probs: numpy.array, shape: (n_scenario,)
    :param solver:
    :return:
    """
    t0 = time()
    if scenario_probs is None:
        scenario_probs = np.ones(n_scenario, dtype=np.float) / n_scenario

    # Model
    model = ConcreteModel()

    # Set
    # model.symbols = np.arange(len(symbols))
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

    # objective
    def cvar_objective_rule(model):
        return model.Z - 1 / (1 - alpha) * sum(
            model.Ys[sdx] * scenario_probs[sdx]
            for sdx in xrange(n_scenario))

    model.cvar_objective = Objective(rule=cvar_objective_rule, sense=maximize)

    opt = SolverFactory(solver)

    instance = model.create()
    results = opt.solve(instance)
    instance.load(results)
    if verbose:
        display(instance)

    # buy and sell amounts
    buy_amounts = pd.Series([instance.buy_amounts[symbol].value for symbol in
                             symbols], index=symbols)
    sell_amounts = pd.Series([instance.sell_amounts[symbol].value for symbol in
                              symbols], index=symbols)

    # value at risk (estimated)
    estimated_var = instance.Z.value

    if verbose:
        print "min_cvar_sp_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "estimated_var": estimated_var,
        "cvar_objective": model.cvar_objective()
    }


def test_min_cvar_sp_portfolio():
    from ipro.dev import (STOCK_PKL_DIR, )

    stock_panel = pd.read_pickle(os.path.join(STOCK_PKL_DIR,
                                              "panel_largest50stocks.pkl"))
    symbols = ['2330', '2412', '2882' ]
    n_stock = len(symbols)
    start_date = date(2010, 12, 1)
    n_scenario = 200

    risk_rois = stock_panel.loc[start_date].loc[symbols]['adj_roi']/100.
    risk_free_roi = 0
    initial_risk_wealth = pd.Series([0, 0., 0], index=symbols)
    initial_risk_free_wealth = 1e6
    buy_trans_fee = 0.001425
    sell_trans_fee = 0.004425

    # pre-start_date
    pre_start_date = date(2010, 7, 27)
    pre_risk_rois = stock_panel.loc[pre_start_date:start_date, symbols,
                    'adj_roi'].T/100.

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

    results = min_cvar_sp_portfolio(symbols, risk_rois, risk_free_roi,
                                    initial_risk_wealth,
                                    initial_risk_free_wealth, buy_trans_fee,
                                    sell_trans_fee, alpha, predict_risk_rois,
                                    predict_risk_free_roi, n_scenario,
                                    scenario_probs=None, solver="ipopt")
    print results

    # verify results
    risk_wealth = initial_risk_wealth
    risk_wealth += results['buy_amounts']
    risk_wealth -= results['sell_amounts']
    print risk_wealth

    predict_portfolio_wealth = (risk_wealth * (predict_risk_rois + 1).T).sum(
        axis=1)
    predict_portfolio_wealth.sort()
    print predict_portfolio_wealth
    plt.plot(predict_portfolio_wealth.values)
    plt.show()


class MinCVaRSPPortfolio(SPTradingPortfolio):
    def __init__(self, symbols, risk_rois, risk_free_rois, initial_risk_wealth,
                 initial_risk_free_wealth, buy_trans_fee=0.001425,
                 sell_trans_fee=0.004425, start_date=date(2005, 1, 1),
                 end_date=date(2015, 4, 30), window_length=200,
                 alpha=0.05, n_scenario=200, verbose=False):

        super(MinCVaRSPPortfolio, self).__init__(
           symbols, risk_rois, risk_free_rois, initial_risk_wealth,
           initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, verbose)
        self.alpha = alpha
        self.n_scenario = n_scenario
        self.var_df = pd.Series(np.zeros(self.n_exp_period),
                                index=self.exp_risk_rois.index)

    def get_trading_func_name(self, *args, **kwargs):
        return "MinCVaRSP_M{}_W{}_a{}".format(
            self.n_stock, self.window_length, self.alpha)

    def add_results_to_reports(self, reports):
        """ add additional items to reports """
        reports['alpha'] = self.alpha
        reports['n_scenario'] = self.n_scenario
        reports['var_df'] = self.var_df
        return reports

    def get_estimated_risk_free_roi(self, *arg, **kwargs):
        """the risk free roi is set all zeros"""
        return 0.

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        heuristic moment matching
        """
        tdx = kwargs['tdx']
        hist_end_idx = self.start_date_idx + tdx
        hist_start_idx = self.start_date_idx + tdx - self.window_length

        # shape: (window_length, n_stock), index slciing should plus 1
        hist_data = self.risk_rois.iloc[hist_start_idx:hist_end_idx+1]
        if self.verbose:
            print "HMM current: {} hist_data:[{}-{}]".format(
                                self.exp_risk_rois.index[tdx],
                                self.risk_rois.index[hist_start_idx],
                                self.risk_rois.index[hist_end_idx])
        tgt_moments = np.zeros((self.n_stock, 4))
        tgt_moments[:, 0] = hist_data.mean(axis=0)
        # the 2nd moment must be standard deviation, not the variance
        tgt_moments[:, 1] = hist_data.std(axis=0)
        tgt_moments[:, 2] = spstats.skew(hist_data, axis=0)
        tgt_moments[:, 3] = spstats.kurtosis(hist_data, axis=0)
        corr_mtx = np.corrcoef(hist_data.T)

        # shape: (n_stock, n_scenario)
        for idx, error_order in enumerate(xrange(-3, 0)):
            try:
                max_moment_err = 10**(error_order)
                max_corr_err = 10**(error_order)
                scenarios = heuristic_moment_matching(tgt_moments, corr_mtx,
                                              self.n_scenario, max_moment_err,
                                                  max_corr_err)
                break
            except ValueError as e:
                print e
                if idx >= 2:
                    raise ValueError('HMM not converge.')

        return pd.DataFrame(scenarios, index=self.symbols)

    def set_specific_period_action(self, *args, **kwargs):
        tdx = kwargs['tdx']
        results = kwargs['results']
        self.var_df.iloc[tdx] = results["estimated_var"]


    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """
        min_cvar function
        """
        tdx = kwargs['tdx']
        results = min_cvar_sp_portfolio(self.symbols,
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


def run_min_cvar_sp_simulation(n_stock, window_length, alpha, n_scenario=200):
    """
    :return: reports
    """
    from ipro.dev import (EXP_SYMBOLS, DROPBOX_UP_EXPERIMENT_DIR)

    n_stock = int(n_stock)
    window_length = int(window_length)
    alpha = float(alpha)

    symbols = EXP_SYMBOLS[:n_stock]
    risk_rois = generate_rois_df(symbols)
    start_date = date(2005, 1, 1)
    end_date = date(2015, 4, 30)

    exp_risk_rois = risk_rois.loc[start_date:end_date]
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6

    obj = MinCVaRSPPortfolio(symbols, risk_rois, risk_free_rois,
                           initial_risk_wealth,
                           initial_risk_free_wealth, start_date=start_date,
                           end_date=end_date, window_length=window_length,
                           alpha=alpha, n_scenario=n_scenario, verbose=False)

    reports = obj.run()
    print reports

    file_name = '{}_SP_{}-{}_m{}_w{}_a{:.2f}_s{}.pkl'.format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        exp_risk_rois.index[0].strftime("%Y%m%d"),
        exp_risk_rois.index[-1].strftime("%Y%m%d"),
        len(symbols),
        window_length,
        alpha,
        n_scenario)

    file_dir = os.path.join(DROPBOX_UP_EXPERIMENT_DIR, 'cvar_sp')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))


    return reports

if __name__ == '__main__':
    import sys
    import argparse

    sys.path.append(os.path.join(os.path.abspath('..'), '..'))
    parser = argparse.ArgumentParser()

    test_min_cvar_sp_portfolio()

    # parser.add_argument("-m", "--n_stock", required=True, type=int,
    #                     choices=range(5, 55, 5))
    # parser.add_argument("-w", "--win_length", required=True, type=int)
    # parser.add_argument("-a", "--alpha", required=True)
    # args = parser.parse_args()
    #
    # run_min_cvar_sp_simulation(args.n_stock, args.win_length, args.alpha)