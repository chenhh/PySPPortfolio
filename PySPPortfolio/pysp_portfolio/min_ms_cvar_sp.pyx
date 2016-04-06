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
from pyomo.opt import SolverStatus, TerminationCondition


from PySPPortfolio.pysp_portfolio import *
from base_model import (SPTradingPortfolio, )

cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INTP_t

def min_ms_cvar_sp_portfolio(symbols, trans_dates,
                         cnp.ndarray[FLOAT_t, ndim=2] risk_rois,
                         cnp.ndarray[FLOAT_t, ndim=1] risk_free_rois,
                         cnp.ndarray[FLOAT_t, ndim=1] allocated_risk_wealth,
                         double allocated_risk_free_wealth,
                         double buy_trans_fee,
                         double sell_trans_fee,
                         alphas,
                         cnp.ndarray[FLOAT_t, ndim=3] predict_risk_rois,
                         cnp.ndarray[FLOAT_t, ndim=1] predict_risk_free_roi,
                         int n_scenario,
                         str solver=DEFAULT_SOLVER,
                         int verbose=False):
    """
    after generating all scenarios, solving the SP at once

    symbols: list of string
    risk_rois: numpy.array, shape: (n_exp_period, n_stock)
    risk_free_rois: numpy.array,, shape: (n_exp_period,)
    allocated_risk_wealth: numpy.array,, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alphas: list of float
    predict_risk_ret: numpy.array, shape: (n_exp_period, n_stock, n_scenario)
    predict_risk_free_roi: float
    n_scenario: integer
    solver: str, supported by Pyomo

    """
    t0 = time()

    cdef INTP_t n_exp_period = risk_rois.shape[0]
    cdef int n_stock = len(symbols)

    # concrete model
    instance = ConcreteModel(name="ms_min_cvar_sp_portfolio")

    # data
    instance.risk_rois = risk_rois
    instance.risk_free_rois = risk_free_rois
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alphas = alphas
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
    def risk_wealth_constraint_rule(model, int tdx, int mdx):
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
    def risk_free_wealth_constraint_rule(model, int tdx):
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

    # constraint
    def cvar_constraint_rule(model, int tdx, int sdx):
        """
        auxiliary variable Y depends on scenario. CVaR <= VaR
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, scenario index
        """
        risk_wealth = sum((1. + model.predict_risk_rois[tdx, mdx, sdx]) *
                     model.risk_wealth[tdx, mdx]
                     for mdx in model.symbols)
        return model.Ys[tdx, sdx] >= (model.Z[tdx] - risk_wealth)

    instance.cvar_constraint = Constraint(
        instance.exp_periods, instance.scenarios,
        rule=cvar_constraint_rule)

    print ("min_ms_cvar_sp constraints and objective rules OK, "
               "{:.3f} secs".format(time() - t0))

    cdef:
        int Tdx = n_exp_period - 1
        cnp.ndarray[FLOAT_t, ndim=2] buy_df = np.zeros((n_exp_period,
                                                        n_stock))
        cnp.ndarray[FLOAT_t, ndim=2] sell_df = np.zeros((n_exp_period,
                                                         n_stock))
        cnp.ndarray[FLOAT_t, ndim=2] risk_df = np.zeros((n_exp_period,
                                                         n_stock))
        cnp.ndarray[FLOAT_t, ndim=1] risk_free_arr = np.zeros(n_exp_period)
        cnp.ndarray[FLOAT_t, ndim=1] var_arr = np.zeros(n_exp_period)

    results_dict = {}
    for adx, alpha in enumerate(alphas):
        t1 = time()
        param = "{}_{}_m{}_p{}_s{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock, n_exp_period, n_scenario, alpha)

        # objective
        def cvar_objective_rule(model):
            cvar_expr_sum = 0
            for tdx in xrange(n_exp_period):
                print "tdx:{} start".format(tdx)
                scenario_expectation = sum(model.Ys[tdx, sdx]
                    for sdx in xrange(n_scenario)) / float(n_scenario)

                cvar_expr = (model.Z[tdx] -  scenario_expectation /
                        (1. - model.alphas[adx]))
                print "tdx:{} cvar_expr OK: {}".format(tdx, cvar_expr)
                cvar_expr_sum += cvar_expr
                print "CVaR expr {}: {}".format(tdx, cvar_expr_sum)

            return cvar_expr_sum
            # scenario_expectation = sum(model.Ys[Tdx, sdx]
            #     for sdx in xrange(n_scenario)) / float(n_scenario)
            # return (model.Z[Tdx] - 1. / (1. - model.alphas[adx]) *
            #         scenario_expectation)


        instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                            sense=maximize)
        # solve
        opt = SolverFactory(solver)
        # if solver == "cplex":
        #     opt.options["workmem"] = 4096
            # turnoff presolve
            # opt.options['preprocessing_presolve'] = 'n'
            # Barrier algorithm and its upper bound
            # opt.options['lpmethod'] = 4
            # opt.options['barrier_limits_objrange'] =1e75
        results = opt.solve(instance, tee=True)
        instance.solutions.load_from(results)
        if verbose:
            display(instance)

        print ("solve min_ms_cvar_sp {} OK {:.2f} secs".format(
            param, time() - t1))
        print ("solver status: {}".format(results.solver.status))
        print ("solver termination cond: {}".format(
            results.solver.termination_condition))
        print (results.solver)

        # extract results
        for tdx in xrange(n_exp_period):
            # shape: (n_exp_period,)
            risk_free_arr[tdx] = instance.risk_free_wealth[tdx].value
            var_arr[tdx] = instance.Z[tdx].value

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
        estimated_var_arr = pd.Series(var_arr, index=trans_dates)

        print ("{} final_total_wealth: {:.2f}".format(
            param, risk_df[Tdx].sum() + risk_free_arr[Tdx]))

        alpha_str = "{:.2f}".format(alpha)
        results_dict[alpha_str] = {
            # shape: (n_exp_period, n_stock)
            "buy_amounts_df": buy_amounts_df,
            "sell_amounts_df": sell_amounts_df,
            "risk_wealth_df": risk_wealth_df,
            # shape: (n_exp_period, )
            "risk_free_wealth_arr": risk_free_wealth_arr,
            "estimated_var_arr": estimated_var_arr,
            # float
            "estimated_cvar": instance.cvar_objective(),
        }

        # reset variables
        buy_df = np.zeros((n_exp_period, n_stock))
        sell_df = np.zeros((n_exp_period, n_stock))
        risk_df = np.zeros((n_exp_period, n_stock))
        risk_free_arr = np.zeros(n_exp_period)
        var_arr = np.zeros(n_exp_period)

        # delete objective
        instance.del_component("cvar_objective")

    return results_dict


class MinMSCVaRSPPortfolio(SPTradingPortfolio):
    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=BUY_TRANS_FEE,
                 double sell_trans_fee=SELL_TRANS_FEE,
                 start_date=START_DATE, end_date=END_DATE,
                 int window_length=WINDOW_LENGTH,
                 int n_scenario=N_SCENARIO,
                 bias=BIAS_ESTIMATOR,
                 alphas=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                         0.85, 0.9, 0.95],
                 int scenario_cnt=1,
                 verbose=False):
        """
        because the multi-stage model will cost many time in constructing
        scenarios constraints, we solve all alphas of the same
        parameters at once.
        """

        super(MinMSCVaRSPPortfolio, self).__init__(
            symbols, risk_rois, risk_free_rois, initial_risk_wealth,
            initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, n_scenario, bias,
            verbose)

        self.alphas = alphas

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


    def get_trading_func_name(self, *args, **kwargs):
        return "MS_MinCVaRSP_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            self.n_stock, self.window_length, self.n_scenario,
            "biased" if self.bias_estimator else "unbiased",
            self.scenario_cnt, kwargs['alpha'])


    def get_estimated_risk_free_rois(self, *arg, **kwargs):
        """ the risk free roi is set all zeros """
        return np.zeros(self.n_exp_period)

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        Returns:
        -----------
        estimated_risk_rois, pandas.Panel,
            shape: (n_exp_period, n_stock, n_scenario)
        """
        if self.scenario_panel is None:
            raise ValueError('no pre-generated scenario panel.')

        return self.scenario_panel

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """
        min_cvar function

        Return
        -------------
        results_dict:
          - key: alpha, str
          - value: results, dict
        """
        results_dict = min_ms_cvar_sp_portfolio(
            self.symbols,
            self.exp_risk_rois.index,
            self.exp_risk_rois.as_matrix(),
            self.risk_free_rois.as_matrix(),
            kwargs['allocated_risk_wealth'].as_matrix(),
            kwargs['allocated_risk_free_wealth'],
            self.buy_trans_fee,
            self.sell_trans_fee,
            self.alphas,    # all alphas
            kwargs['estimated_risk_rois'].as_matrix(),
            kwargs['estimated_risk_free_roi'],
            self.n_scenario,
        )
        return results_dict

    def run(self, *args, **kwargs):
        """  # solve all scenarios at once """
        t0 = time()

        # estimated_risk_rois: shape: (n_exp_period, n_stock, n_scenario)
        try:
            estimated_risk_rois = self.get_estimated_risk_rois()
        except ValueError as e:
            raise ValueError("generating scenario error: {}".format(e))

        # estimated_risk_free_rois: shape: (n_exp_period,)
        estimated_risk_free_rois = self.get_estimated_risk_free_rois()

        # determining the buy and sell amounts
        results_dict = self.get_current_buy_sell_amounts(
            estimated_risk_rois=estimated_risk_rois,
            estimated_risk_free_roi=estimated_risk_free_rois,
            allocated_risk_wealth=self.initial_risk_wealth,
            allocated_risk_free_wealth=self.initial_risk_free_wealth,
            *args, **kwargs)

        reports_dict = {}
        for alpha_value in self.alphas:
            func_name = self.get_trading_func_name(alpha=alpha_value)
            print (func_name)
            t1 = time()
            alpha_str = "{:.2f}".format(alpha_value)

            # shape: (n_exp_period, n_stock)
            risk_wealth_df = results_dict[alpha_str]['risk_wealth_df']
            buy_amounts_df = results_dict[alpha_str]['buy_amounts_df']
            sell_amounts_df = results_dict[alpha_str]['sell_amounts_df']

            # shape: (n_exp_period, )
            risk_free_wealth = results_dict[alpha_str]['risk_free_wealth_arr']
            estimated_var_arr = results_dict[alpha_str]["estimated_var_arr"]
            # float
            estimated_cvar = results_dict[alpha_str]["estimated_cvar"]

            # end of iterations, computing statistics
            Tdx = self.n_exp_period - 1
            final_wealth = (risk_wealth_df.iloc[Tdx].sum() +
                            risk_free_wealth[Tdx])

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
            reports['window_length'] = self.window_length
            reports['n_scenario'] = self.n_scenario
            reports['alpha'] = alpha_value
            reports['scenario_cnt'] = self.scenario_cnt
            reports['buy_amounts_df'] = buy_amounts_df
            reports['sell_amounts_df'] =sell_amounts_df

            # add simulation time
            reports['simulation_time'] = time() - t1

            reports_dict[alpha_str] = reports
            print ("{} {} OK [{}-{}], {:.4f}.secs".format(
                func_name, alpha_str, self.exp_risk_rois.index[0],
                self.exp_risk_rois.index[Tdx], time() - t1))

        print ("{} {} OK [{}-{}], {:.4f}.secs".format(
                func_name, self.alphas, self.exp_risk_rois.index[0],
                self.exp_risk_rois.index[Tdx], time() - t0))
        return reports_dict



