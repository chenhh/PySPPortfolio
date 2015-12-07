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
from min_cvar_sp import (MinCVaRSPPortfolio, )

cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INTP_t

def min_cvar_eev_portfolio(symbols,
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
                          str solver=DEFAULT_SOLVER,
                          int verbose=False):
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
    # shape:(n_stock, n_scneario)
    instance.all_predict_risk_rois = predict_risk_rois
    # shape: (n_stock,)
    instance.mean_predict_risk_rois = predict_risk_rois.mean(axis=1)
    # float
    instance.predict_risk_free_roi = predict_risk_free_roi

    cdef Py_ssize_t n_stock = len(symbols)
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
    # value at risk (estimated)
    cdef double estimated_var = instance.Z.value
    cdef double estimated_cvar = instance.cvar_objective()

    # fixed the first-stage variables
    instance.buy_amounts.fixed = True
    instance.sell_amounts.fixed = True
    instance.risk_wealth.fixed = True
    instance.risk_free_wealth.fixed = True

    estimated_eev_var_arr = np.zeros(n_scenario)
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
        estimated_eev_var_arr[sdx] = instance.Z.value
        estimated_eev_cvar_arr[sdx] = instance.cvar_objective()

        print "scenario:{}, VaR:{}, CVaR:{}".format(sdx+1,
            estimated_eev_var_arr[sdx], estimated_eev_cvar_arr[sdx])

    if verbose:
        print "min_cvar_eev_portfolio OK, {:.3f} secs".format(time() - t0)

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "estimated_var": estimated_var,
        "estimated_cvar": estimated_cvar,
        "estimated_eev_var": estimated_eev_var_arr.mean(),
        "estimated_eev_cvar": estimated_eev_cvar_arr.mean()
    }

class MinCVaREEVPortfolio(MinCVaRSPPortfolio):
    """
    expected of expected value (EEV) model
    """
    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=BUY_TRANS_FEE,
                 double sell_trans_fee=SELL_TRANS_FEE,
                 start_date=START_DATE, end_date=END_DATE,
                 int window_length=WINDOW_LENGTH,
                 int n_scenario=N_SCENARIO,
                 bias=BIAS_ESTIMATOR,
                 double alpha=0.95,
                 int scenario_cnt=1,
                 verbose=False):

        super(MinCVaREEVPortfolio, self).__init__(
            symbols, risk_rois, risk_free_rois, initial_risk_wealth,
            initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, n_scenario, bias,
            alpha, scenario_cnt, verbose)

        self.eev_var_arr = pd.Series(np.zeros(self.n_exp_period),
                                  index = self.exp_risk_rois.index)

        self.eev_cvar_arr = pd.Series(np.zeros(self.n_exp_period),
                                  index = self.exp_risk_rois.index)

    def get_trading_func_name(self, *args, **kwargs):
        return "MinCVaREEV_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            self.n_stock, self.window_length, self.n_scenario,
             "biased" if self.bias_estimator else "unbiased",
             self.scenario_cnt, self.alpha)

    def add_results_to_reports(self, reports, *args, **kwargs):
        """ add additional items to reports """
        reports['alpha'] = self.alpha
        reports['scenario_cnt'] = self.scenario_cnt
        reports['var_arr'] = self.var_arr
        reports['cvar_arr'] = self.cvar_arr
        reports['eev_var_arr'] = self.eev_var_arr
        reports['eev_cvar_arr'] = self.eev_cvar_arr
        return reports

    def set_specific_period_action(self, *args, **kwargs):
        """
        user specified action after getting results
        """
        tdx = kwargs['tdx']
        results = kwargs['results']
        self.var_arr.iloc[tdx] = results["estimated_var"]
        self.cvar_arr.iloc[tdx] = results['estimated_cvar']
        self.eev_var_arr.iloc[tdx] = results['estimated_eev_var']
        self.eev_cvar_arr.iloc[tdx] = results['estimated_eev_cvar']

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """
        min_cvar function, after getting 1st decision variables,
        they will be apply to each scenario to get corresponding CVaR
        and the average CVaR is the EEV solution
        """

        # current exp_period index
        tdx = kwargs['tdx']
        results = min_cvar_eev_portfolio(
            self.symbols,
            self.exp_risk_rois.iloc[tdx].as_matrix(),
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