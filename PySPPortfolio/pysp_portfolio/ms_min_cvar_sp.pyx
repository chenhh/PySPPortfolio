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
from pyomo.environ import *

from PySPPortfolio.pysp_portfolio import *
from base_model import (MS_SPTradingPortfolio, )

cimport numpy as cnp

ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INTP_t

def ms_min_cvar_sp_portfolio(symbols, trans_dates,
                             cnp.ndarray[FLOAT_t, ndim=2] risk_rois,
                             cnp.ndarray[FLOAT_t, ndim=1] risk_free_rois,
                             cnp.ndarray[FLOAT_t, ndim=1] allocated_risk_wealth,
                             double allocated_risk_free_wealth,
                             double buy_trans_fee,
                             double sell_trans_fee,
                             double alpha,
                             cnp.ndarray[FLOAT_t, ndim=3] predict_risk_rois,
                             cnp.ndarray[FLOAT_t, ndim=1] predict_risk_free_roi,
                             int n_scenario,
                             scenario_probs=False,
                             solver=DEFAULT_SOLVER,
                             verbose=True):
    """
    after generating all scenarios, solving the SP at once

    symbols: list of string
    risk_rois: numpy.array, shape: (n_exp_period, n_stock)
    risk_free_rois: numpy.array,, shape: (n_exp_period,)
    allocated_risk_wealth: numpy.array,, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, 1-alpha is the significant level
    predict_risk_ret: numpy.array, shape: (n_exp_period, n_stock, n_scenario)
    predict_risk_free_roi: float
    n_scenario: integer
    scenario_probs: numpy.array, shape: (n_scenario,)
    solver: str, supported by Pyomo

    """
    t0 = time()
    if scenario_probs == False:
        scenario_probs = np.ones(n_scenario, dtype=np.float) / n_scenario

    cdef INTP_t n_exp_period = risk_rois.shape[0]
    cdef int n_stock = len(symbols)

    param = "{}_{}_m{}_p{}_s{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock, n_exp_period, n_scenario, alpha)

    # concrete model
    instance = ConcreteModel(name="ms_min_cvar_sp_portfolio")

    # data
    instance.scenario_probs = scenario_probs
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
    instance.exp_periods = np.arange(n_exp_period)
    instance.symbols = np.arange(n_stock)
    instance.scenarios = np.arange(n_scenario)

    # decision variables
    # in each period, we buy or sell stock, shape: (n_exp_period, n_stock)
    instance.buy_amounts = Var(instance.exp_periods, instance.symbols,
                               within=NonNegativeReals)
    instance.sell_amounts = Var(instance.exp_periods, instance.symbols,
                                within=NonNegativeReals)

    # shape: (n_exp_period, n_stock)
    instance.risk_wealth = Var(instance.exp_periods, instance.symbols,
                               within=NonNegativeReals)
    # shape: (n_exp_period, )
    instance.risk_free_wealth = Var(instance.exp_periods,
                                    within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    # shape: (n_exp_period, )
    instance.Z = Var(instance.exp_periods, within=Reals)

    # aux variable, portfolio wealth less than than VaR (Z)
    # shape: (n_exp_period, n_scenario)
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
            return ( model.risk_wealth[tdx, mdx] ==
                (1. + model.risk_rois[tdx, mdx]) *
                     model.allocated_risk_wealth[mdx] +
                model.buy_amounts[tdx, mdx] - model.sell_amounts[tdx, mdx]
            )
        else:
            return (
                model.risk_wealth[tdx, mdx] ==
                (1. + model.risk_rois[tdx, mdx]) *
                    model.risk_wealth[tdx - 1, mdx] +
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
        total_sell = sum((1. - model.sell_trans_fee) *
                         model.sell_amounts[tdx,mdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) *
                        model.buy_amounts[tdx, mdx]
                        for mdx in model.symbols)
        if tdx == 0:
            return (
                model.risk_free_wealth[tdx] ==
                (1. + model.risk_free_rois[tdx]) *
                model.allocated_risk_free_wealth +
                total_sell - total_buy
            )
        else:
            return (
                model.risk_free_wealth[tdx] ==
                (1. + model.risk_free_rois[tdx]) *
                    model.risk_free_wealth[tdx - 1] +
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
        wealth = sum((1. + model.predict_risk_rois[tdx, mdx, sdx]) *
                     model.risk_wealth[tdx, mdx]
                     for mdx in model.symbols)
        return model.Ys[tdx, sdx] >= (model.Z[tdx] - wealth)

    instance.cvar_constraint = Constraint(
        instance.exp_periods, instance.scenarios,
        rule=cvar_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        edx = n_exp_period - 1
        scenario_expectation = sum(
            model.Ys[edx, sdx] * model.scenario_probs[sdx]
            for sdx in xrange(n_scenario))
        return model.Z[edx] - 1. / (1. - model.alpha) * scenario_expectation

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)
    print ("{} constraints and objective rules OK, {:.3f} secs".format(
        param, time() - t0))

    # solve
    t5 = time()
    opt = SolverFactory(solver)
    if solver == "cplex":
        opt.set_options('Threads=4')
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    if verbose:
        # display(instance)
        print ("{} OK {:3f} secs".format(param, time() - t5))

    # display(instance)

    # extract results
    edx = n_exp_period - 1

    # shape: (n_exp_period, n_stock)
    cdef:
        cnp.ndarray[FLOAT_t, ndim=2] buy_df = np.zeros((n_exp_period, n_stock))
        cnp.ndarray[FLOAT_t, ndim=2] sell_df = np.zeros((n_exp_period, n_stock))
        cnp.ndarray[FLOAT_t, ndim=2] risk_df = np.zeros((n_exp_period, n_stock))
        cnp.ndarray[FLOAT_t, ndim=1] risk_free_arr = np.zeros(n_exp_period)
        cnp.ndarray[FLOAT_t, ndim=1]  var_arr = np.zeros(n_exp_period)

    for tdx in xrange(n_exp_period):
        risk_free_arr[tdx] = instance.risk_free_wealth[tdx].value
        var_arr[tdx] = instance.Z[tdx].value

        for mdx in xrange(n_stock):
            buy_df[tdx, mdx] = instance.buy_amounts[tdx, mdx].value
            sell_df[tdx, mdx] = instance.sell_amounts[tdx, mdx].value
            risk_df[tdx, mdx] = instance.risk_wealth[tdx, mdx].value

    # shape: (n_exp_period, n_stock)
    buy_amounts_df = pd.DataFrame(buy_df, index=trans_dates, columns=symbols)
    sell_amounts_df = pd.DataFrame(sell_df, index=trans_dates, columns=symbols)
    risk_wealth_df = pd.DataFrame(risk_df, index=trans_dates, columns=symbols)

    # shape: (n_exp_period, )
    risk_free_wealth_arr = pd.Series(risk_free_arr, index=trans_dates)
    estimated_var_arr = pd.Series(var_arr, index=trans_dates)

    return {
        "buy_amounts_df": buy_amounts_df,
        "sell_amounts_df": sell_amounts_df,
        "risk_wealth_df": risk_wealth_df,
        "risk_free_wealth_arr": risk_free_wealth_arr,
        "estimated_var_arr": estimated_var_arr,
        "estimated_cvar": instance.cvar_objective(),
    }


class MS_MinCVaRSPPortfolio(MS_SPTradingPortfolio):
    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth, initial_risk_free_wealth,
                 buy_trans_fee=BUY_TRANS_FEE, sell_trans_fee=SELL_TRANS_FEE,
                 start_date=START_DATE, end_date=END_DATE,
                 window_length=WINDOW_LENGTH, n_scenario=N_SCENARIO,
                 bias=BIAS_ESTIMATOR, alpha=0.9, scenario_cnt=1,
                 verbose=False):

        """
        because the mutti-stage model will cost many time in constructing
        scenarios constraints, therefore we solve all alphas of the same
        parameters at once.
        """

        super(MS_MinCVaRSPPortfolio, self).__init__(
            symbols, risk_rois, risk_free_rois,
            initial_risk_wealth, initial_risk_free_wealth,
            buy_trans_fee, sell_trans_fee,
            start_date, end_date,
            window_length, n_scenario, bias,
            verbose
        )

        self.alpha = float(alpha)

        # try to load generated scenario panel
        scenario_name = "{}_{}_m{}_w{}_s{}_{}_{}.pkl".format(
            start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"),
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

        self.cvar = 0

    def get_trading_func_name(self, *args, **kwargs):
        return "MS_MinCVaRSP_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            self.n_stock, self.window_length, self.n_scenario,
            "biased" if self.bias_estimator else "unbiased",
            self.scenario_cnt, self.alpha)

    def add_results_to_reports(self, reports):
        """ add additional items to reports """
        reports['alpha'] = self.alpha
        reports['scenario_cnt'] = self.scenario_cnt
        reports['var_arr'] = self.var_arr
        reports['cvar'] = self.cvar
        return reports

    def get_estimated_risk_free_rois(self, *arg, **kwargs):
        """ the risk free roi is set all zeros """
        return np.zeros(self.n_exp_period)

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        Returns:
        -----------
        estimated_risk_rois, numpy.array, shape: (n_stock, n_scenario)
        """
        # current index in the exp_period

        if self.scenario_panel is None:
            raise ValueError('no pre-generated scenario panel.')

        return self.scenario_panel

    def set_specific_period_action(self, *args, **kwargs):
        """
        user specified action after getting results
        """
        results = kwargs['results']
        self.var_arr = results["estimated_var_arr"]
        self.cvar = results['estimated_cvar']

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """ min_cvar function """

        # current exp_period index
        results = ms_min_cvar_sp_portfolio(
            self.symbols,
            self.exp_risk_rois.index,
            self.exp_risk_rois.as_matrix(),
            self.risk_free_rois.as_matrix(),
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
