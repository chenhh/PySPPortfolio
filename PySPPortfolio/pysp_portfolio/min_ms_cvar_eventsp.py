# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
from time import time
from datetime import datetime
import os
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from PySPPortfolio.pysp_portfolio import *
from base_model import (SPTradingPortfolio, )

def min_ms_cvar_eventsp_portfolio(symbols, trans_dates, risk_rois,
                                risk_free_rois, allocated_risk_wealth,
                                allocated_risk_free_wealth, buy_trans_fee,
                                sell_trans_fee, alpha, predict_risk_rois,
                                predict_risk_free_roi, n_scenario=200,
                                solver = DEFAULT_SOLVER, verbose=False,
                                solver_io="python", keepfiles=False):
    """
    in each period, when the decision variables have branch, using the
    expected decisions

    symbols: list of string
    risk_rois: numpy.array, shape: (n_exp_period, n_stock)
    risk_free_rois: numpy.array,, shape: (n_exp_period,)
    allocated_risk_wealth: numpy.array, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alphas: float
    predict_risk_rois: numpy.array,
    shape: (n_exp_period, n_stock, n_scenario)
    predict_risk_free_rois: numpy.array, shape: (n_exp_period,)
    n_scenario: integer
    solver: str, supported by Pyomo
    solve_io: {"lp", "nl", "os", "python"}
    """

    t0 = time()
    n_exp_period = risk_rois.shape[0]
    n_stock = len(symbols)
    print ("start time: {}".format(datetime.now()))
    # concrete model
    instance = ConcreteModel(name="ms_min_cvar_eventsp_portfolio")
    param = "{}_{}_m{}_p{}_s{}_a{:.2f}".format(
        trans_dates[0].strftime("%Y%m%d"), trans_dates[-1].strftime("%Y%m%d"),
        n_stock, n_exp_period, n_scenario, alpha)

    # data
    instance.risk_rois = risk_rois
    instance.risk_free_rois = risk_free_rois
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alpha = alpha
    # shape: (n_exp_period, n_stock, n_scenario)
    instance.predict_risk_rois = predict_risk_rois
    # shape: (n_exp_period, )
    instance.predict_risk_free_roi = predict_risk_free_roi

    # Set
    instance.n_exp_period = n_exp_period
    instance.exp_periods = np.arange(n_exp_period)
    instance.symbols = np.arange(n_stock)
    instance.scenarios = np.arange(n_scenario)
    instance.n_scenario = n_scenario

    # decision variables
    # the expected buy or sell actions, shape: (n_exp_period, n_stock)
    # decision from period 1 to T.
    instance.buy_amounts = Var(instance.exp_periods, instance.symbols,
                               within=NonNegativeReals)
    instance.sell_amounts = Var(instance.exp_periods, instance.symbols,
                                within=NonNegativeReals)
    instance.proxy_buy_amounts = Var(instance.exp_periods, instance.symbols,
                                 instance.scenarios, within=NonNegativeReals)
    instance.proxy_sell_amounts = Var(instance.exp_periods, instance.symbols,
                                  instance.scenarios, within=NonNegativeReals)

    # shape: (n_exp_period, n_stock)
    # decision from period 1 to T
    instance.risk_wealth = Var(instance.exp_periods, instance.symbols,
                               within=NonNegativeReals)
    instance.proxy_risk_wealth = Var(instance.exp_periods, instance.symbols,
                                instance.scenarios, within=NonNegativeReals)
    # shape: (n_exp_period, )
    # decision from period 1 to T
    instance.risk_free_wealth = Var(instance.exp_periods,
                                    within=NonNegativeReals)
    instance.proxy_risk_free_wealth = Var(instance.exp_periods,
                                  instance.scenarios, within=NonNegativeReals)

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    # shape: (n_exp_period, )
    # decision from period 2 to T+1
    instance.Z = Var(instance.exp_periods, within=Reals)

    # aux variable, portfolio wealth less than than VaR (Z)
    # in each stage, there is only one scenario,
    # shape: (n_exp_period, ),
    # decision from period 2 to T+1
    instance.Ys = Var(instance.exp_periods, instance.scenarios,
                      within=NonNegativeReals)

    print ("combinations (exp_period, stock, scenarios)=({}, {}, {})".format(
        n_exp_period, n_stock, n_scenario))
    print ("constructing risk wealth constraints")

    # constraint
    def risk_wealth_constraint_rule(model, tdx, mdx, sdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        sdx: integer, index of scenario
        """
        if tdx == 0:
            prev_risk_wealth = model.allocated_risk_wealth[mdx]
            risk_roi = model.risk_rois[tdx, mdx]
        else:
            prev_risk_wealth = model.risk_wealth[tdx-1, mdx]
            # t-th day realization
            risk_roi = model.predict_risk_rois[tdx-1 , mdx, sdx]

        return (model.proxy_risk_wealth[tdx, mdx, sdx] ==
                (1. + risk_roi) * prev_risk_wealth +
                model.proxy_buy_amounts[tdx, mdx, sdx] -
                model.proxy_sell_amounts[tdx, mdx, sdx]
                )

    instance.risk_wealth_constraint = Constraint(
        instance.exp_periods, instance.symbols, instance.scenarios,
        rule=risk_wealth_constraint_rule)

    # explicit constraint
    def risk_wealth_root_rule(model, mdx, sdx):
        """
        because the risk_roi has the same value in all scenarios in the root
        node, it should have the same decision value on the period.
        The risk wealth is the final value of buy and sell amount,
        then we can give the constraint on risk_wealth which will imply to
        buy and sell amounts.
        """
        return (model.proxy_risk_wealth[0, mdx, sdx-1] ==
                model.proxy_risk_wealth[0, mdx, sdx])

    instance.risk_wealth_root_constraint = Constraint(
        instance.symbols, range(1, n_scenario),
        rule=risk_wealth_root_rule)


    # risk wealth constraint
    def risk_wealth_expected_decision_rule(model, tdx, mdx):
        """
        expectation of decision variables

        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        exp_wealth = sum(model.proxy_risk_wealth[tdx, mdx, sdx]
                         for sdx in model.scenarios) / model.n_scenario
        return model.risk_wealth[tdx, mdx] == exp_wealth

    instance.risk_wealth_decision_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=risk_wealth_expected_decision_rule
    )

    # buy amount constraint
    def buy_expected_decision_rule(model, tdx, mdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        exp_buy = sum(model.proxy_buy_amounts[tdx, mdx, sdx]
                     for sdx in model.scenarios) / model.n_scenario
        return model.buy_amounts[tdx, mdx] == exp_buy

    instance.buy_decision_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=buy_expected_decision_rule
    )

    # buy amount constraint
    def sell_expected_decision_rule(model, tdx, mdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        exp_sell = sum(model.proxy_sell_amounts[tdx, mdx, sdx]
                      for sdx in model.scenarios) / model.n_scenario
        return model.sell_amounts[tdx, mdx] == exp_sell

    instance.sell_decision_constraint = Constraint(
        instance.exp_periods, instance.symbols,
        rule=sell_expected_decision_rule
    )
    print ("min_ms_cvar_eventsp {} risk wealth decisions constraints OK, "
           "{:.3f} secs".format(param, time() - t0))
    print ("constructing risk free wealth constraints")
    t1 = time()

    # constraint
    def risk_free_wealth_constraint_rule(model, tdx, sdx):
        """
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, index of scenario
        """
        total_sell = sum((1. - model.sell_trans_fee) *
                         model.proxy_sell_amounts[tdx, mdx, sdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) *
                        model.proxy_buy_amounts[tdx, mdx, sdx]
                        for mdx in model.symbols)
        if tdx == 0:
            prev_risk_free_wealth = model.allocated_risk_free_wealth
        else:
            prev_risk_free_wealth = model.risk_free_wealth[tdx - 1]

        return (model.proxy_risk_free_wealth[tdx, sdx] ==
                (1. + model.risk_free_rois[tdx]) *
                prev_risk_free_wealth + total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        instance.exp_periods, instance.scenarios,
        rule=risk_free_wealth_constraint_rule)

    # risk wealth constraint
    def risk_free_wealth_decision_rule(model, tdx):
        """
        Parameters
        ------------
        tdx: integer, time index of period
        """
        exp_wealth = sum(model.proxy_risk_free_wealth[tdx, sdx]
                         for sdx in model.scenarios) / model.n_scenario
        return model.risk_free_wealth[tdx] == exp_wealth

    instance.risk_free_wealth_decision_constraint = Constraint(
        instance.exp_periods, rule=risk_free_wealth_decision_rule
    )
    print ("min_ms_cvar_eventsp {} risk free wealth decisions constraints OK, "
           "{:.3f} secs".format(param, time() - t1))
    print ("constructing cvar constraints")
    t2 = time()

    # constraint
    def cvar_constraint_rule(model, tdx, sdx):
        """
        auxiliary variable Y depends on scenario. CVaR <= VaR
        Parameters:
        ------------
        tdx: integer, time index of period
        sdx: integer, scenario index
        sdx2: integer, the descent scenario index of sdx
        """
        risk_wealth = sum((1. + model.predict_risk_rois[tdx, mdx, sdx]) *
                          model.risk_wealth[tdx, mdx]
                          for mdx in model.symbols)
        return model.Ys[tdx, sdx] >= (model.Z[tdx] - risk_wealth)

    instance.cvar_constraint = Constraint(
        instance.exp_periods, instance.scenarios,
        rule=cvar_constraint_rule)

    print ("min_ms_cvar_eventsp {} cvar constraints OK, "
           "{:.3f} secs".format(param, time() - t2))
    print ("constructing objective rule.")
    t3 = time()

    # objective
    def cvar_objective_rule(model):
        cvar_expr_sum = 0
        for tdx in xrange(n_exp_period):
            scenario_expectation = sum(model.Ys[tdx, sdx] for sdx in
                                       model.scenarios)/model.n_scenario
            cvar_expr = (model.Z[tdx] - scenario_expectation /
                         (1. - model.alpha))
            cvar_expr_sum = cvar_expr_sum + cvar_expr
        return cvar_expr_sum/model.n_exp_period

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)
    # solve
    print ("min_ms_cvar_eventsp {} objective OK {:.3f} secs, "
           "start solving:".format(param, time()-t3))

    t4 = time()
    opt = SolverFactory(solver, solver_io=solver_io)
    results = opt.solve(instance, keepfiles=keepfiles)
    instance.solutions.load_from(results)
    if verbose:
        display(instance)

    print ("solve min_ms_cvar_eventsp {} OK {:.2f} secs".format(
        param, time() - t4))
    print ("solver status: {}".format(results.solver.status))
    print ("solver termination cond: {}".format(
        results.solver.termination_condition))
    print (results.solver)

     # extract results
    proxy_buy_pnl = np.zeros((n_exp_period, n_stock, n_scenario))
    proxy_sell_pnl = np.zeros((n_exp_period, n_stock, n_scenario))
    proxy_risk_pnl= np.zeros((n_exp_period, n_stock, n_scenario))
    proxy_risk_free_df = np.zeros((n_exp_period, n_scenario))
    # proxy_var_df = np.zeros((n_exp_period, n_scenario))
    buy_df = np.zeros((n_exp_period, n_stock))
    sell_df = np.zeros((n_exp_period, n_stock))
    risk_df = np.zeros((n_exp_period, n_stock))
    risk_free_arr = np.zeros(n_exp_period)
    var_arr = np.zeros(n_exp_period)

    for tdx in xrange(n_exp_period):
        # shape: (n_exp_period,)
        risk_free_arr[tdx] = instance.risk_free_wealth[tdx].value
        var_arr[tdx] = instance.Z[tdx].value

        for mdx in xrange(n_stock):
            # shape: (n_exp_period, n_stock)
            buy_df[tdx, mdx] = instance.buy_amounts[tdx, mdx].value
            sell_df[tdx, mdx] = instance.sell_amounts[tdx, mdx].value
            risk_df[tdx, mdx] = instance.risk_wealth[tdx, mdx].value

            for sdx in xrange(n_scenario):
                proxy_buy_pnl[tdx, mdx, sdx] = instance.proxy_buy_amounts[
                    tdx, mdx, sdx].value
                proxy_sell_pnl[tdx, mdx, sdx] = instance.proxy_sell_amounts[
                    tdx, mdx, sdx].value
                proxy_risk_pnl[tdx, mdx, sdx] = instance.proxy_risk_wealth[
                    tdx, mdx, sdx].value

        for sdx in xrange(n_scenario):
            proxy_risk_free_df[tdx, sdx] = \
                instance.proxy_risk_free_wealth[tdx, sdx].value
            # proxy_var_df[tdx, sdx] = instance.proxy_Z[tdx, sdx].value

    # shape: (n_exp_period, n_stock, n_scenario)
    proxy_buy_amounts_pnl = pd.Panel(proxy_buy_pnl, items=trans_dates,
                                     major_axis=symbols)
    proxy_sell_amounts_pnl = pd.Panel(proxy_sell_pnl, items=trans_dates,
                                     major_axis=symbols)
    proxy_risk_wealth_pnl = pd.Panel(proxy_risk_pnl, items=trans_dates,
                                     major_axis=symbols)

    # shape: (n_exp_period, n_scenario)
    proxy_risk_free_wealth_df = pd.DataFrame(proxy_risk_free_df,
                                           index=trans_dates)
    # proxy_estimated_var_df =   pd.DataFrame(proxy_var_df,
    #                                        index=trans_dates)

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
    exp_final_wealth = risk_df[Tdx].sum() + risk_free_arr[Tdx]
    print ("{} expected_final_total_wealth: {:.2f}".format(
        param, exp_final_wealth ))

    results = {
        # shape: (n_exp_period, n_stock, n_scenario)
        "proxy_buy_amounts_pnl": proxy_buy_amounts_pnl,
        "proxy_sell_amounts_pnl": proxy_sell_amounts_pnl,
        "proxy_risk_wealth_pnl": proxy_risk_wealth_pnl,

        # shape: (n_exp_period, n_scenario)
        "proxy_risk_free_wealth_df": proxy_risk_free_wealth_df,
        # "proxy_estimated_var_df": proxy_estimated_var_df,

        # shape: (n_exp_period, n_stock)
        "buy_amounts_df": buy_amounts_df,
        "sell_amounts_df": sell_amounts_df,
        "risk_wealth_df": risk_wealth_df,
        # shape: (n_exp_period, )
        "risk_free_wealth_arr": risk_free_wealth_arr,
        "estimated_var_arr": estimated_var_arr,
        # float
        "estimated_cvar": instance.cvar_objective(),
        "expected_final_wealth":exp_final_wealth,
    }
    return results

class MinMSCVaREventSPPortfolio(SPTradingPortfolio):
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
                 verbose=False, solver_io="python", keepfiles=False):
        """
        Multistage min cvar event scenario
        """
        super(MinMSCVaREventSPPortfolio, self).__init__(
            symbols, risk_rois, risk_free_rois, initial_risk_wealth,
            initial_risk_free_wealth, buy_trans_fee, sell_trans_fee,
            start_date, end_date, window_length, n_scenario, bias,
            verbose)

        self.alpha = float(alpha)
        self.solver_io = solver_io
        self.keepfiles = keepfiles

        # try to load generated scenario panel
        scenario_name = "{}_{}_m{}_w{}_s{}_{}_{}.pkl".format(
            START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
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
            if start_date != START_DATE or end_date != END_DATE:
                self.scenario_panel = self.scenario_panel.loc[
                                      start_date:end_date]
                print ("scenario panel dates:{}-{}".format(
                    self.scenario_panel.items[0],
                    self.scenario_panel.items[-1]))
            self.scenario_cnt = scenario_cnt

    def get_trading_func_name(self, *args, **kwargs):
        return "MS_MinCVaREventSP_{}_{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d"),
            self.n_stock, self.window_length, self.n_scenario,
            "biased" if self.bias_estimator else "unbiased",
            self.scenario_cnt, self.alpha)

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
        results = min_ms_cvar_eventsp_portfolio(
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
            solver_io=self.solver_io,
            keepfiles=self.keepfiles,
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


        # end of iterations, computing statistics
        Tdx = self.n_exp_period - 1
        final_wealth = (risk_wealth_df.iloc[Tdx].sum() + risk_free_wealth[Tdx])

        # compute transaction fee
        trans_fee_loss = (buy_amounts_df.sum() * self.buy_trans_fee +
                          sell_amounts_df.sum() * self.sell_trans_fee)

        # get reports
        simulation_reports = self.get_performance_report(
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
        simulation_reports['window_length'] = self.window_length
        simulation_reports['n_scenario'] = self.n_scenario
        simulation_reports['alpha'] = self.alpha
        simulation_reports['scenario_cnt'] = self.scenario_cnt
        simulation_reports['buy_amounts_df'] = buy_amounts_df
        simulation_reports['sell_amounts_df'] =sell_amounts_df
        simulation_reports['estimated_var_arr']= results['estimated_var_arr']
        simulation_reports['estimated_cvar'] = results['estimated_cvar']

        simulation_reports["proxy_buy_amounts_pnl"]=results["proxy_buy_amounts_pnl"]
        simulation_reports["proxy_sell_amounts_pnl"]=results[
            "proxy_sell_amounts_pnl"]
        simulation_reports[
            "proxy_risk_wealth_pnl"]=results["proxy_risk_wealth_pnl"]

        # shape: (n_exp_period, n_scenario)
        simulation_reports["proxy_risk_free_wealth_df"]= results[
            "proxy_risk_free_wealth_df"]
        # simulation_reports[
        #     "proxy_estimated_var_df"]=results["proxy_estimated_var_df"]

        # add simulation time
        simulation_reports['simulation_time'] = time() - t0

        print ("{} {} OK [{}-{}], {:.4f}.secs".format(
            func_name, self.alpha, self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[Tdx], time() - t0))

        return simulation_reports