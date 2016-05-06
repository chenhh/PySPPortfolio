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
from time import time
import os
import numpy as np
import pandas as pd
import scipy.stats as spstats
from pyomo.environ import *

from PySPPortfolio.pysp_portfolio import *
from scenario.c_moment_matching import heuristic_moment_matching
from base_model import (SPTradingPortfolio, )

cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INTP_t

def min_cvar_sp_portfolio(symbols,
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
                          scenario_probs=None,
                          str solver=DEFAULT_SOLVER,
                          int verbose=False):
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

    cdef Py_ssize_t n_stock = len(symbols)
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
    def risk_wealth_constraint_rule(model, int mdx):
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
    def cvar_constraint_rule(model, int sdx):
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
    print ("Y:{}, VaR:{}, CVaR{}".format(
        [instance.Ys[sdx].value for sdx in xrange(n_scenario)],
        instance.Z.value,
        instance.cvar_objective,
    ))

    if verbose:
        display(instance)

    # buy and sell amounts
    buy_amounts = pd.Series([instance.buy_amounts[mdx].value
                             for mdx in xrange(n_stock)], index=symbols)
    sell_amounts = pd.Series([instance.sell_amounts[mdx].value
                              for mdx in xrange(n_stock)], index=symbols)

    # value at risk (estimated)
    cdef double estimated_var = instance.Z.value

    if verbose:
        print "min_cvar_sp_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "estimated_var": estimated_var,
        "estimated_cvar": instance.cvar_objective()
    }



class MinCVaRSPPortfolio(SPTradingPortfolio):
    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=BUY_TRANS_FEE,
                 double sell_trans_fee=SELL_TRANS_FEE,
                 start_date=START_DATE, end_date=END_DATE,
                 int window_length=WINDOW_LENGTH,
                 int n_scenario=N_SCENARIO,
                 bias=BIAS_ESTIMATOR,
                 double alpha=0.05,
                 int scenario_cnt=1,
                 verbose=False):
        """
        2nd-stage SP

        Parameters:
         -----------------------
        alpha: float, 0<=value<0.5, 1-alpha is the confidence level of risk

        Data:
        -------------
        var_arr: pandas.Series, Value at risk of each period in the simulation
        cvar_arr: pandas.Series, conditional value at risk of each period
        """

        super(MinCVaRSPPortfolio, self).__init__(
           symbols, risk_rois, risk_free_rois, initial_risk_wealth,
           initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, n_scenario, bias, verbose)

        self.alpha = float(alpha)

        # try to load generated scenario panel
        scenario_name = "{}_{}_m{}_w{}_s{}_{}_{}.pkl".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
            len(symbols), window_length, n_scenario,
            "biased" if bias else "unbiased", scenario_cnt)

        scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios',
                                 scenario_name)

        if not os.path.exists(scenario_path):
            raise ValueError("{} not exists.".format(scenario_name))
            self.scenario_panel = None
            self.scenario_cnt = 0
        else:
            self.scenario_panel = pd.read_pickle(scenario_path)
            self.scenario_cnt = scenario_cnt

        # additional results
        self.var_arr = pd.Series(np.zeros(self.n_exp_period),
                                index=self.exp_risk_rois.index)
        self.cvar_arr = pd.Series(np.zeros(self.n_exp_period),
                                  index = self.exp_risk_rois.index)


    def get_trading_func_name(self, *args, **kwargs):
        return "MinCVaRSP_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            self.n_stock, self.window_length, self.n_scenario,
             "biased" if self.bias_estimator else "unbiased",
             self.scenario_cnt, self.alpha)

    def add_results_to_reports(self, reports, *args, **kwargs):
        """ add additional items to reports """
        reports['alpha'] = self.alpha
        reports['scenario_cnt'] = self.scenario_cnt
        reports['var_arr'] = self.var_arr
        reports['cvar_arr'] = self.cvar_arr
        return reports

    def get_estimated_risk_free_rois(self, *arg, **kwargs):
        """ the risk free roi is set all zeros """
        return 0.

    def get_estimated_risk_rois(self, *args, **kwargs):
        """

        Returns:
        -----------
        estimated_risk_rois, numpy.array, shape: (n_stock, n_scenario)
        """
        # current index in the exp_period
        tdx, trans_date = kwargs['tdx'], kwargs['trans_date']
        if self.scenario_panel is not None:
            df = self.scenario_panel.loc[trans_date]
            assert self.symbols == df.index.tolist()
            return df
        else:
            # because we trade stock on the after-hour market, we known today
            # market information, therefore the historical interval contain
            # current day
            hist_end_idx = self.start_date_idx + tdx + 1
            hist_start_idx = self.start_date_idx + tdx - self.window_length + 1

            # shape: (window_length, n_stock)
            hist_data = self.risk_rois.iloc[hist_start_idx:hist_end_idx]
            if self.verbose:
                print ("HMM current: {} hist_data:[{}-{}]".format(
                                    self.exp_risk_rois.index[tdx],
                                    self.risk_rois.index[hist_start_idx],
                                    self.risk_rois.index[hist_end_idx]))

            # 1-4 th moments of historical data, shape: (n_stock, 4)
            tgt_moments = np.zeros((self.n_stock, 4))
            tgt_moments[:, 0] = hist_data.mean(axis=0)
            if self.bias_estimator:
                # the 2nd moment must be standard deviation, not the variance
                tgt_moments[:, 1] = hist_data.std(axis=0)
                tgt_moments[:, 2] = spstats.skew(hist_data, axis=0)
                tgt_moments[:, 3] = spstats.kurtosis(hist_data, axis=0)
            else:
                tgt_moments[:, 1] = hist_data.std(axis=0, ddof=1)
                tgt_moments[:, 2] = spstats.skew(hist_data, axis=0,
                                                 bias=False)
                tgt_moments[:, 3] = spstats.kurtosis(hist_data, axis=0,
                                                     bias=False)
            corr_mtx = np.corrcoef(hist_data.T)

            # scenarios shape: (n_stock, n_scenario)
            for idx, error_order in enumerate(xrange(-3, 0)):
                # if the HMM is not converge, relax the tolerance error
                try:
                    max_moment_err = 10**error_order
                    max_corr_err = 10**error_order
                    scenarios = heuristic_moment_matching(
                                    tgt_moments, corr_mtx,
                                    self.n_scenario,
                                    self.bias_estimator,
                                    max_moment_err, max_corr_err)
                    break
                except ValueError as e:
                    print e
                    if idx >= 2:
                        raise ValueError('{}: {} HMM not converge.'.format(
                            self.get_trading_func_name(),
                            self.exp_risk_rois.index[tdx]
                        ))
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
        )
        return results


def min_cvar_sp_portfolio2(symbols,
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
                          scenario_probs=None,
                          str solver=DEFAULT_SOLVER,
                          int verbose=False):
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
    instance.mean_predict_risk_rois = predict_risk_rois.mean(axis=1)
    instance.predict_risk_free_roi = predict_risk_free_roi

    cdef Py_ssize_t n_stock = len(symbols)
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
    def risk_wealth_constraint_rule(model, int mdx):
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
    def cvar_constraint_rule(model, int sdx):
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

    if verbose:
        display(instance)

    # buy and sell amounts
    buy_amounts = pd.Series([instance.buy_amounts[mdx].value
                             for mdx in xrange(n_stock)], index=symbols)
    sell_amounts = pd.Series([instance.sell_amounts[mdx].value
                              for mdx in xrange(n_stock)], index=symbols)

    # value at risk (estimated)
    cdef double estimated_var = instance.Z.value
    cdef double estimated_cvar = instance.cvar_objective()

    # EEV part
    # delete old CVaR constraint
    instance.del_component("cvar_constraint")
    instance.del_component("cvar_objective")
    instance.del_component("Ys")
    instance.del_component("Ys_index")

    # new variable
    instance.Y = Var(within=NonNegativeReals)

    # update CVaR constraint
    def cvar_constraint_rule(model):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        wealth = sum((1. + model.mean_predict_risk_rois[mdx]) *
                     model.risk_wealth[mdx]
                     for mdx in model.symbols)

        return model.Y >= (model.Z - wealth)

    instance.cvar_constraint = Constraint(rule=cvar_constraint_rule)

    # update objective
    def cvar_objective_rule(model):
        return model.Z - 1. / (1. - model.alpha) * model.Y

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve eev 1st stage
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    # value at risk (estimated)
    cdef double estimated_ev_var = instance.Z.value
    cdef double estimated_ev_cvar = instance.cvar_objective()

    # fixed the first-stage variables
    for mdx in instance.symbols:
        instance.buy_amounts[mdx].fixed = True
        instance.sell_amounts[mdx].fixed = True
        instance.risk_wealth[mdx].fixed = True
    instance.risk_free_wealth.fixed = True

    # Z is viewed as the first-stage variable
    instance.Z.fixed = True

    estimated_eev_cvar_arr = np.zeros(n_scenario)
    # EEV
    for sdx in xrange(n_scenario):
        scen_roi = predict_risk_rois[:, sdx]
        portfolio_value = (
            sum((1+scen_roi[mdx])* instance.risk_wealth[mdx].value
                for mdx in np.arange(n_stock)) +
            instance.risk_free_wealth.value)

        if estimated_var <= portfolio_value:
            estimated_eev_cvar_arr[sdx] = estimated_var
        else:
            diff = (estimated_var - portfolio_value)
            estimated_eev_cvar_arr[sdx] =( estimated_var -
                                           1/(1-alpha) * diff)

    estimated_eev_cvar = estimated_eev_cvar_arr.mean()
    vss = estimated_cvar - estimated_eev_cvar
    if verbose:
        print "min_cvar_sp_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "estimated_var": estimated_var,
        "estimated_cvar": estimated_cvar,
        "estimated_ev_var": estimated_ev_var,
        "estimated_ev_cvar": estimated_ev_cvar,
        "estimated_eev_cvar": estimated_eev_cvar,
        "vss": vss,
    }



class MinCVaRSPPortfolio2(SPTradingPortfolio):
    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=BUY_TRANS_FEE,
                 double sell_trans_fee=SELL_TRANS_FEE,
                 start_date=START_DATE, end_date=END_DATE,
                 int window_length=WINDOW_LENGTH,
                 int n_scenario=N_SCENARIO,
                 bias=BIAS_ESTIMATOR,
                 double alpha=0.05,
                 int scenario_cnt=1,
                 verbose=False):
        """
        2nd-stage SP

        Parameters:
         -----------------------
        alpha: float, 0<=value<0.5, 1-alpha is the confidence level of risk

        Data:
        -------------
        var_arr: pandas.Series, Value at risk of each period in the simulation
        cvar_arr: pandas.Series, conditional value at risk of each period
        """

        super(MinCVaRSPPortfolio2, self).__init__(
           symbols, risk_rois, risk_free_rois, initial_risk_wealth,
           initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, n_scenario, bias, verbose)

        self.alpha = float(alpha)

        # try to load generated scenario panel
        scenario_name = "{}_{}_m{}_w{}_s{}_{}_{}.pkl".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
            len(symbols), window_length, n_scenario,
            "biased" if bias else "unbiased", scenario_cnt)

        scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios',
                                 scenario_name)

        if not os.path.exists(scenario_path):
            raise ValueError("{} not exists.".format(scenario_name))
            self.scenario_panel = None
            self.scenario_cnt = 0
        else:
            self.scenario_panel = pd.read_pickle(scenario_path)
            if start_date != START_DATE or end_date != END_DATE:
                self.scenario_panel = self.scenario_panel.loc[
                                      start_date:end_date]
                print ("scenario panel dates:{}-{}".format(
                    self.scenario_panel.items[0],
                    self.scenario_panel.items[-1]))
            self.scenario_cnt = scenario_cnt

        # additional results
        self.var_arr = pd.Series(np.zeros(self.n_exp_period),
                                index=self.exp_risk_rois.index)
        self.cvar_arr = pd.Series(np.zeros(self.n_exp_period),
                                  index = self.exp_risk_rois.index)

        self.ev_var_arr = pd.Series(np.zeros(self.n_exp_period),
                                  index = self.exp_risk_rois.index)

        self.ev_cvar_arr = pd.Series(np.zeros(self.n_exp_period),
                                  index = self.exp_risk_rois.index)

        self.eev_cvar_arr = pd.Series(np.zeros(self.n_exp_period),
                                  index = self.exp_risk_rois.index)
        self.vss_arr = pd.Series(np.zeros(self.n_exp_period),
                                  index = self.exp_risk_rois.index)


    def get_trading_func_name(self, *args, **kwargs):
        return "MinCVaRSP2_{}_{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d"),
            self.n_stock, self.window_length, self.n_scenario,
             "biased" if self.bias_estimator else "unbiased",
             self.scenario_cnt, self.alpha)

    def add_results_to_reports(self, reports, *args, **kwargs):
        """ add additional items to reports """
        reports['alpha'] = self.alpha
        reports['scenario_cnt'] = self.scenario_cnt
        reports['var_arr'] = self.var_arr
        reports['cvar_arr'] = self.cvar_arr
        reports['ev_var_arr'] = self.ev_var_arr
        reports['ev_cvar_arr'] = self.ev_cvar_arr
        reports['eev_var_arr'] = self.eev_cvar_arr
        reports['vss_arr'] = self.vss_arr
        return reports

    def get_estimated_risk_free_rois(self, *arg, **kwargs):
        """ the risk free roi is set all zeros """
        return 0.

    def get_estimated_risk_rois(self, *args, **kwargs):
        """

        Returns:
        -----------
        estimated_risk_rois, numpy.array, shape: (n_stock, n_scenario)
        """
        # current index in the exp_period
        tdx, trans_date = kwargs['tdx'], kwargs['trans_date']
        if self.scenario_panel is not None:
            df = self.scenario_panel.loc[trans_date]
            assert self.symbols == df.index.tolist()
            return df
        else:
            # because we trade stock on the after-hour market, we known today
            # market information, therefore the historical interval contain
            # current day
            hist_end_idx = self.start_date_idx + tdx + 1
            hist_start_idx = self.start_date_idx + tdx - self.window_length + 1

            # shape: (window_length, n_stock)
            hist_data = self.risk_rois.iloc[hist_start_idx:hist_end_idx]
            if self.verbose:
                print ("HMM current: {} hist_data:[{}-{}]".format(
                                    self.exp_risk_rois.index[tdx],
                                    self.risk_rois.index[hist_start_idx],
                                    self.risk_rois.index[hist_end_idx]))

            # 1-4 th moments of historical data, shape: (n_stock, 4)
            tgt_moments = np.zeros((self.n_stock, 4))
            tgt_moments[:, 0] = hist_data.mean(axis=0)
            if self.bias_estimator:
                # the 2nd moment must be standard deviation, not the variance
                tgt_moments[:, 1] = hist_data.std(axis=0)
                tgt_moments[:, 2] = spstats.skew(hist_data, axis=0)
                tgt_moments[:, 3] = spstats.kurtosis(hist_data, axis=0)
            else:
                tgt_moments[:, 1] = hist_data.std(axis=0, ddof=1)
                tgt_moments[:, 2] = spstats.skew(hist_data, axis=0,
                                                 bias=False)
                tgt_moments[:, 3] = spstats.kurtosis(hist_data, axis=0,
                                                     bias=False)
            corr_mtx = np.corrcoef(hist_data.T)

            # scenarios shape: (n_stock, n_scenario)
            for idx, error_order in enumerate(xrange(-3, 0)):
                # if the HMM is not converge, relax the tolerance error
                try:
                    max_moment_err = 10**error_order
                    max_corr_err = 10**error_order
                    scenarios = heuristic_moment_matching(
                                    tgt_moments, corr_mtx,
                                    self.n_scenario,
                                    self.bias_estimator,
                                    max_moment_err, max_corr_err)
                    break
                except ValueError as e:
                    print e
                    if idx >= 2:
                        raise ValueError('{}: {} HMM not converge.'.format(
                            self.get_trading_func_name(),
                            self.exp_risk_rois.index[tdx]
                        ))
            return pd.DataFrame(scenarios, index=self.symbols)


    def set_specific_period_action(self, *args, **kwargs):
        """
        user specified action after getting results
        """
        tdx = kwargs['tdx']
        results = kwargs['results']
        self.var_arr.iloc[tdx] = results["estimated_var"]
        self.cvar_arr.iloc[tdx] = results['estimated_cvar']
        self.ev_var_arr.iloc[tdx] = results['estimated_ev_var']
        self.ev_cvar_arr.iloc[tdx] = results['estimated_ev_cvar']
        self.eev_cvar_arr.iloc[tdx] = results['estimated_eev_cvar']
        self.vss_arr.iloc[tdx] = results['vss']

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """ min_cvar function """

        # current exp_period index
        tdx = kwargs['tdx']
        results = min_cvar_sp_portfolio2(
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
        )
        return results
