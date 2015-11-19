# -*- coding: utf-8 -*-
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

def min_cvar_sip_portfolio(symbols, risk_rois, risk_free_roi,
                           allocated_risk_wealth, allocated_risk_free_wealth,
                           buy_trans_fee, sell_trans_fee, alpha,
                           predict_risk_rois, predict_risk_free_roi,
                           n_scenario, max_portfolio_size,
                           scenario_probs=None, solver=DEFAULT_SOLVER,
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
    instance = ConcreteModel()

    # Set
    instance.symbols = symbols
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

    instance.risk_wealth_constraint = Constraint(
        instance.symbols, rule=risk_wealth_constraint_rule)

    # constraint
    def risk_free_wealth_constraint_rule(model):
        total_sell = sum((1 - sell_trans_fee) * model.sell_amounts[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1 + buy_trans_fee) * model.buy_amounts[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth ==
                (1. + risk_free_roi) * allocated_risk_free_wealth +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # constraint
    def cvar_constraint_rule(model, sdx):
        """auxiliary variable Y depends on scenario. CVaR <= VaR"""
        wealth = sum((1. + predict_risk_rois.at[mdx, sdx]) *
                     model.risk_wealth[mdx]
                     for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - wealth)

    instance.cvar_constraint = Constraint(instance.scenarios,
                                       rule=cvar_constraint_rule)

    # constraint
    def chosen_constraint_rule(model, mdx):
        total_wealth = sum(allocated_risk_wealth) + allocated_risk_free_wealth
        return model.risk_wealth[mdx] <= model.chosen[mdx] * total_wealth

    instance.chosen_constraint = Constraint(instance.symbols,
                                         rule=chosen_constraint_rule)

    # constraint
    def portfolio_size_constraint_rule(model):
        return sum(
            model.chosen[mdx] for mdx in model.symbols) <= max_portfolio_size

    instance.portfolio_size_constraint = Constraint(
        rule=portfolio_size_constraint_rule)

    # objective
    def cvar_objective_rule(model):
        scenario_expectation = sum(model.Ys[sdx] * scenario_probs[sdx]
                                    for sdx in xrange(n_scenario))
        return model.Z - 1 / (1 - alpha) * scenario_expectation

    instance.cvar_objective = Objective(rule=cvar_objective_rule, sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
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
        "estimated_cvar": instance.cvar_objective()
    }


class MinCVaRSIPPortfolio(MinCVaRSPPortfolio):
    def __init__(self, candidate_symbols, max_portfolio_size, risk_rois,
                 risk_free_rois, initial_risk_wealth,
                 initial_risk_free_wealth, buy_trans_fee=BUY_TRANS_FEE,
                 sell_trans_fee=SELL_TRANS_FEE, start_date=START_DATE,
                 end_date=END_DATE, window_length=WINDOW_LENGTH,
                 n_scenario=N_SCENARIO, bias=BIAS_ESTIMATOR, alpha=0.05,
                 scenario_cnt=1, verbose=False):
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
            print ("{} not exists.".format(scenario_name))
            self.scenario_panel = None
            self.scenario_cnt = 0
        else:
            self.scenario_panel = pd.read_pickle(scenario_path)
            self.scenario_cnt = scenario_cnt


    def valid_specific_parameters(self, *args, **kwargs):
        if self.max_portfolio_size > self.n_stock:
            raise ValueError('the max portfolio size {} > the number of '
                             'stock {} in the candidate set.'.format(
                self.max_portfolio_size, self.n_stock))


    def get_trading_func_name(self, *args, **kwargs):
        return "MinCVaRSIP_all{}_m{}_w{}_s{}_{}_{}_a{}".format(
            self.n_stock, self.max_portfolio_size,
            self.window_length,
            self.n_scenario, "biased" if self.bias_estimator else "unbiased",
            self.scenario_cnt, self.alpha)

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


def multi_stage_scenarios_min_cvar_sip_portfolio(symbols, risk_rois,
                                               risk_free_roi,
                          allocated_risk_wealth,
                          allocated_risk_free_wealth, buy_trans_fee,
                          sell_trans_fee, alpha, predict_risk_rois,
                          predict_risk_free_roi, n_scenario,
                          max_portfolio_size,
                          scenario_probs=None, solver=DEFAULT_SOLVER, verbose=False
    ):
    """
    after generating all scenarios, solving the SIP at once
    symbols: list of string
    risk_rois: pandas.DataFrame, shape: (n_exp_period, n_stock)
    risk_free_roi: pandas.Series, shape: (n_exp_period,)
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
    pass