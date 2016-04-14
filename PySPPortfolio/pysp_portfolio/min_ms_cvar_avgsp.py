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
from pyomo.opt import SolverStatus, TerminationCondition
from PySPPortfolio.pysp_portfolio import *
from base_model import (SPTradingPortfolio, )


def min_ms_cvar_avgsp_portfolio(symbols, trans_dates, risk_rois,
                                risk_free_rois, allocated_risk_wealth,
                                allocated_risk_free_wealth, buy_trans_fee,
                                sell_trans_fee, alpha, predict_risk_rois,
                                predict_risk_free_rois, n_scenario=200,
                                solver = DEFAULT_SOLVER, verbose=False):
    """
    in each period, using average scenarios
    the results of avgsp are independent to the alpha,

    symbols: list of string
    risk_rois: numpy.array, shape: (n_exp_period, n_stock)
    risk_free_rois: numpy.array,, shape: (n_exp_period,)
    allocated_risk_wealth: numpy.array,, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alphas: float
    predict_risk_rois: numpy.array, shape: (n_exp_period, n_stock, n_scenario)
    predict_risk_free_rois: numpy.array, shape: (n_exp_period, n_scenario)
    n_scenario: integer
    solver: str, supported by Pyomo

    """
    t0 = time()
    n_exp_period = risk_rois.shape[0]
    n_stock = len(symbols)

    # concrete model
    instance = ConcreteModel(name="ms_min_cvar_avgsp_portfolio")
    param = "{}_{}_m{}_p{}_s{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock, n_exp_period, n_scenario, alpha)

    # data
    instance.risk_rois = risk_rois
    instance.risk_free_rois = risk_free_rois
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alpha = alpha
    # shape: (n_exp_period, n_stock)
    instance.avg_predict_risk_rois = predict_risk_rois.mean(axis=2)
    # shape: (n_exp_period,)
    instance.avg_predict_risk_free_rois = predict_risk_free_rois.mean(axis=1)

    # Set
    instance.n_exp_period = n_exp_period
    instance.exp_periods = np.arange(n_exp_period)
    instance.symbols = np.arange(n_stock)
    instance.scenarios = np.arange(n_scenario)
    instance.n_scenario = n_scenario

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
    # in each stage, there is only one scneario,
    # shape: (n_exp_period, ),
    instance.Ys = Var(instance.exp_periods, within=NonNegativeReals)

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
            risk_roi = model.risk_rois[tdx, mdx]
        else:
            prev_risk_wealth = model.risk_wealth[tdx - 1, mdx]
            risk_roi = model.avg_predict_risk_rois[tdx - 1, mdx]

        return (model.risk_wealth[tdx, mdx] ==
                (1. + risk_roi) * prev_risk_wealth +
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
                (1. + model.avg_predict_risk_free_rois[tdx]) *
                prev_risk_free_wealth + total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        instance.exp_periods, rule=risk_free_wealth_constraint_rule)

    # constraint
    def cvar_constraint_rule(model, tdx):
        """
        auxiliary variable Y depends on scenario. CVaR <= VaR
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, scenario index
        """
        risk_wealth = sum((1. + model.avg_predict_risk_rois[tdx, mdx]) *
                          model.risk_wealth[tdx, mdx] for mdx in model.symbols)
        return model.Ys[tdx] >= (model.Z[tdx] - risk_wealth)

    instance.cvar_constraint = Constraint(
        instance.exp_periods, rule=cvar_constraint_rule)

    print ("min_ms_cvar_avgsp {} constraints OK, "
           "{:.3f} secs".format(param, time() - t0))

    buy_df = np.zeros((n_exp_period, n_stock))
    sell_df = np.zeros((n_exp_period, n_stock))
    risk_df = np.zeros((n_exp_period, n_stock))
    risk_free_arr = np.zeros(n_exp_period)
    var_arr = np.zeros(n_exp_period)

    # objective
    def cvar_objective_rule(model):
        cvar_expr_sum = 0
        for tdx in xrange(n_exp_period):
            scenario_expectation = model.Ys[tdx]
            cvar_expr = (model.Z[tdx] - scenario_expectation /
                         (1. - model.alpha))
            cvar_expr_sum = cvar_expr_sum + cvar_expr
        return cvar_expr_sum

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)
    # solve
    print "min_ms_cvar_avgsp {} start solving:".format(param)
    t1 = time()
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    if verbose:
        display(instance)

    print ("solve min_ms_cvar_avgsp {} OK {:.2f} secs".format(
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

    Tdx = instance.n_exp_period - 1
    print ("{} estimated_final_total_wealth: {:.2f}".format(
        param, risk_df[Tdx].sum() + risk_free_arr[Tdx]))

    results= {
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
    return results

class MinMSCVaRAvgSPPortfolio(SPTradingPortfolio):
    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth, initial_risk_free_wealth,
                 buy_trans_fee=BUY_TRANS_FEE,
                 sell_trans_fee=SELL_TRANS_FEE,
                 start_date=START_DATE, end_date=END_DATE,
                 window_length=WINDOW_LENGTH,
                 n_scenario=N_SCENARIO,
                 bias=BIAS_ESTIMATOR,
                 alpha=0.9,
                 scenario_cnt=1,
                 verbose=False):
        """
        Multistage min cvar average scenario
        """
        super(MinMSCVaRAvgSPPortfolio, self).__init__(
            symbols, risk_rois, risk_free_rois, initial_risk_wealth,
            initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, n_scenario, bias,
            verbose)

        self.alpha = float(alpha)

        # try to load generated scenario panel
        scenario_name = "{}_{}_m{}_w{}_s{}_{}_{}.pkl".format(
            start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"),
            len(symbols), window_length, n_scenario,
            "biased" if bias else "unbiased", scenario_cnt)

        scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios',
                                     scenario_name)
        # check if scenario cache file exists
        if not os.path.exists(scenario_path):
            raise ValueError("{} not exists.".format(scenario_name))
            self.scenario_panel = None
            self.scenario_cnt = 0
        else:
            self.scenario_panel = pd.read_pickle(scenario_path)
            self.scenario_cnt = scenario_cnt


    def get_trading_func_name(self, *args, **kwargs):
        return "MS_MinCVaRAvgSP_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            self.n_stock, self.window_length, self.n_scenario,
            "biased" if self.bias_estimator else "unbiased",
            self.scenario_cnt, self.alpha)


    def get_estimated_risk_free_rois(self, *arg, **kwargs):
        """ the risk free roi is set all zeros """
        return np.zeros((self.n_exp_period, self.n_scenario))

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
        results = min_ms_cvar_avgsp_portfolio(
            self.symbols,
            self.exp_risk_rois.index,
            self.exp_risk_rois.as_matrix(),
            self.risk_free_rois.as_matrix(),
            kwargs['allocated_risk_wealth'].as_matrix(),
            kwargs['allocated_risk_free_wealth'],
            self.buy_trans_fee,
            self.sell_trans_fee,
            self.alpha,    # all alphas
            kwargs['estimated_risk_rois'].as_matrix(),
            kwargs['estimated_risk_free_roi'],
            self.n_scenario,
        )
        return results

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
        results = self.get_current_buy_sell_amounts(
            estimated_risk_rois=estimated_risk_rois,
            estimated_risk_free_roi=estimated_risk_free_rois,
            allocated_risk_wealth=self.initial_risk_wealth,
            allocated_risk_free_wealth=self.initial_risk_free_wealth,
            *args, **kwargs)

        func_name = self.get_trading_func_name()
        # shape: (n_exp_period, n_stock)
        risk_wealth_df = results['risk_wealth_df']
        buy_amounts_df = results['buy_amounts_df']
        sell_amounts_df = results['sell_amounts_df']

        # shape: (n_exp_period, )
        risk_free_wealth = results['risk_free_wealth_arr']
        estimated_var_arr = results["estimated_var_arr"]
        estimated_cvar = results["estimated_cvar"]

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
        reports['alpha'] = self.alpha
        reports['scenario_cnt'] = self.scenario_cnt
        reports['buy_amounts_df'] = buy_amounts_df
        reports['sell_amounts_df'] =sell_amounts_df

        # add simulation time
        reports['simulation_time'] = time() - 0

        print ("{} {} OK [{}-{}], {:.4f}.secs".format(
            func_name, self.alpha, self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[Tdx], time() - t0))

        return reports