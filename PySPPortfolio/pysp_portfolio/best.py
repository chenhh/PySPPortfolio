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

from PySPPortfolio.pysp_portfolio import *
from base_model import (PortfolioReportMixin, ValidPortfolioParameterMixin)


def best_ms_portfolio(symbols, trans_dates, risk_rois,
                     risk_free_rois, allocated_risk_wealth,
                     allocated_risk_free_wealth, buy_trans_fee,
                     sell_trans_fee, solver=DEFAULT_SOLVER, verbose=False):
    """
    after generating all scenarios, solving the SP at once

    symbols: list of string
    risk_rois: numpy.array, shape: (n_exp_period, n_stock)
    risk_free_rois: numpy.array,, shape: (n_exp_period,)
    allocated_risk_wealth: numpy.array,, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    solver: str, supported by Pyomo

    """
    t0 = time()

    n_exp_period = risk_rois.shape[0]
    n_stock = len(symbols)

    # concrete model
    instance = ConcreteModel(name="best_ms_portfolio")

    # data
    instance.risk_rois = risk_rois
    instance.risk_free_rois = risk_free_rois
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee

    # Set
    instance.exp_periods = np.arange(n_exp_period)
    instance.symbols = np.arange(n_stock)

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

    # constraint
    def risk_wealth_constraint_rule(model, tdx, mdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        if tdx == 0:
            prev_risk_wealth = model.allocated_risk_wealth[mdx]
        else:
            prev_risk_wealth = model.risk_wealth[tdx - 1, mdx]

        return ( model.risk_wealth[tdx, mdx] ==
                (1. + model.risk_rois[tdx, mdx]) * prev_risk_wealth +
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
                         model.sell_amounts[tdx, mdx] for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) *
                        model.buy_amounts[tdx, mdx] for mdx in model.symbols)
        if tdx == 0:
            prev_risk_free_wealth = model.allocated_risk_free_wealth
        else:
            prev_risk_free_wealth = model.risk_free_wealth[tdx - 1]

        return (model.risk_free_wealth[tdx] ==
                (1. + model.risk_free_rois[tdx]) * prev_risk_free_wealth +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        instance.exp_periods, rule=risk_free_wealth_constraint_rule)


    print ("best_ms_constraints and objective rules OK, "
               "{:.3f} secs".format(time() - t0))

     # objective
    def wealth_objective_rule(model):
        Tdx = n_exp_period - 1
        return (sum(model.risk_wealth[Tdx, mdx] for mdx in model.symbols) +
                model.risk_free_wealth[Tdx])

    instance.wealth_objective = Objective(rule=wealth_objective_rule,
                                        sense=maximize)
    t1 = time()

    param = "best_ms_{}_{}_m{}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock)
    # solve
    opt = SolverFactory(solver)
    if solver == "cplex":
        opt.options["workmem"] = 4096
        # turnoff presolve
        # opt.options['preprocessing_presolve'] = 'n'
        # Barrier algorithm and its upper bound
        opt.options['lpmethod'] = 4
        opt.options['barrier_limits_objrange'] =1e75
    results = opt.solve(instance, tee=True)
    instance.solutions.load_from(results)
    if verbose:
        display(instance)

    print ("solve best_ms_portfolio {} OK {:.2f} secs".format(
        param, time() - t1))
    print ("solver status: {}".format(results.solver.status))
    print ("solver termination cond: {}".format(
        results.solver.termination_condition))
    print (results.solver)

    # extract results
    Tdx = n_exp_period - 1
    buy_df = np.zeros((n_exp_period, n_stock))
    sell_df = np.zeros((n_exp_period, n_stock))
    risk_df = np.zeros((n_exp_period, n_stock))
    risk_free_arr = np.zeros(n_exp_period)

    for tdx in xrange(n_exp_period):
        # shape: (n_exp_period,)
        risk_free_arr[tdx] = instance.risk_free_wealth[tdx].value

        for mdx in xrange(n_stock):
            # shape: (n_exp_period, n_stock)
            buy_df[tdx, mdx] = instance.buy_amounts[tdx, mdx].value
            sell_df[tdx, mdx] = instance.sell_amounts[tdx, mdx].value
            risk_df[tdx, mdx] = instance.risk_wealth[tdx, mdx].value

    # shape: (n_exp_period, n_stock)
    buy_amounts_df = pd.DataFrame(buy_df, index=trans_dates,
                                  columns=symbols)
    sell_amounts_df = pd.DataFrame(sell_df, index=trans_dates,
                                   columns=symbols)
    risk_wealth_df = pd.DataFrame(risk_df, index=trans_dates,
                                  columns=symbols)

    # shape: (n_exp_period, )
    risk_free_wealth_arr = pd.Series(risk_free_arr, index=trans_dates)

    print ("{} final_total_wealth: {:.2f}".format(
        param, risk_df[Tdx].sum() + risk_free_arr[Tdx]))

    return  {
        # shape: (n_exp_period, n_stock)
        "buy_amounts_df": buy_amounts_df,
        "sell_amounts_df": sell_amounts_df,
        "risk_wealth_df": risk_wealth_df,
        # shape: (n_exp_period, )
        "risk_free_wealth_arr": risk_free_wealth_arr,
        "final_wealth": instance.wealth_objective(),
    }


class BestMSPortfolio(PortfolioReportMixin, ValidPortfolioParameterMixin):
    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth,
                 initial_risk_free_wealth,
                 buy_trans_fee=BUY_TRANS_FEE,
                 sell_trans_fee=SELL_TRANS_FEE,
                 start_date=START_DATE, end_date=END_DATE,
                 verbose=False):
        """
        """

        self.symbols = symbols
        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        # valid number of symbols
        self.valid_dimension("n_symbol", len(symbols),
                             len(initial_risk_wealth))
        self.initial_risk_wealth = initial_risk_wealth
        self.initial_risk_free_wealth = initial_risk_free_wealth

        # valid transaction fee
        self.buy_trans_fee = buy_trans_fee
        self.valid_trans_fee(buy_trans_fee)
        self.valid_trans_fee(sell_trans_fee)
        self.sell_trans_fee = sell_trans_fee

        self.verbose = verbose

        # .loc() will contain the end_date element
        self.valid_trans_date(start_date, end_date)
        self.exp_risk_rois = risk_rois.loc[start_date:end_date]
        self.exp_risk_free_rois = risk_free_rois.loc[start_date:end_date]
        self.n_exp_period = self.exp_risk_rois.shape[0]
        self.n_stock = self.exp_risk_rois.shape[1]

        # date index in total data
        self.start_date_idx = self.risk_rois.index.get_loc(
            self.exp_risk_rois.index[0])

        # results data
        # risk wealth DataFrame, shape: (n_exp_period, n_stock)
        self.risk_wealth_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # risk_free Series, shape: (n_exp_period, )
        self.risk_free_wealth = pd.Series(np.zeros(self.n_exp_period),
                                          index=self.exp_risk_free_rois.index)

        # buying amount DataFrame, shape: (n_exp_period, n_stock)
        self.buy_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # selling amount DataFrame, shape: (n_exp_period, n_stock)
        self.sell_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # cumulative loss in transaction fee in the simulation
        self.trans_fee_loss = 0


    def get_trading_func_name(self, *args, **kwargs):
        return "Best_MS_m{}".format(self.n_stock)

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """
        min_cvar function

        Return
        -------------
        results_dict:
          - key: alpha, str
          - value: results, dict
        """
        results_dict = best_ms_portfolio(
            self.symbols,
            self.exp_risk_rois.index,
            self.exp_risk_rois.as_matrix(),
            self.risk_free_rois.as_matrix(),
            kwargs['allocated_risk_wealth'].as_matrix(),
            kwargs['allocated_risk_free_wealth'],
            self.buy_trans_fee,
            self.sell_trans_fee,
        )
        return results_dict

    def run(self, *args, **kwargs):
        """  # solve all scenarios at once """
        t0 = time()

        func_name = self.get_trading_func_name()
        print (func_name)

        # determining the buy and sell amounts
        results = self.get_current_buy_sell_amounts(
            allocated_risk_wealth=self.initial_risk_wealth,
            allocated_risk_free_wealth=self.initial_risk_free_wealth,
            *args, **kwargs)

        t1 = time()
        # shape: (n_exp_period, n_stock)
        risk_wealth_df = results['risk_wealth_df']
        buy_amounts_df = results['buy_amounts_df']
        sell_amounts_df = results['sell_amounts_df']

        # shape: (n_exp_period, )
        risk_free_wealth = results['risk_free_wealth_arr']

        # end of iterations, computing statistics
        Tdx = self.n_exp_period - 1
        final_wealth = results['final_wealth']

        # compute transaction fee
        trans_fee_loss = (buy_amounts_df.sum() * self.buy_trans_fee +
                          sell_amounts_df.sum() * self.sell_trans_fee)

        # get reports
        reports = self.get_performance_report(
            func_name,
            self.symbols,
            self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[Tdx],
            self.buy_trans_fee,
            self.sell_trans_fee,
            (self.initial_risk_wealth.sum() +
             self.initial_risk_free_wealth),
            final_wealth,
            self.n_exp_period,
            trans_fee_loss,
            risk_wealth_df,
            risk_free_wealth)

        # model additional elements to reports
        reports['buy_amounts_df'] = buy_amounts_df
        reports['sell_amounts_df'] =sell_amounts_df

        # add simulation time
        reports['simulation_time'] = time() - t1

        print ("{} OK [{}-{}], {} {:.4f}.secs".format(
                func_name, self.exp_risk_rois.index[0],
                self.exp_risk_rois.index[Tdx],
              final_wealth, time() - t0))
        return reports



