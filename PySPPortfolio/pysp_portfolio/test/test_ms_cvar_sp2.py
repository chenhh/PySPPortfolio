# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from __future__ import division
from time import time
import numpy as np
import pandas as pd
from pyomo.environ import *


def ms_cvar_sp2_portfolio(symbols, trans_dates, risk_rois, risk_free_rois,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alphas,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario, solver="cplex", verbose=False):
    """
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
    n_exp_period = risk_rois.shape[0]
    n_stock = len(symbols)

    # concrete model
    instance = ConcreteModel(name="ms_cvar2_sp_portfolio")

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
    instance.n_exp_period = n_exp_period
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
        Parameters
        ------------
        tdx: integer, time index of period
        mdx: integer, index of symbol
        """
        if tdx == 0:
            prev_risk_wealth = model.allocated_risk_wealth[mdx]
        else:
            prev_risk_wealth = model.risk_wealth[tdx - 1, mdx]

        return (model.risk_wealth[tdx, mdx] ==
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


    # constraint
    def cvar_constraint_rule(model, tdx, sdx):
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
    int
    Tdx = n_exp_period - 1
    cnp.ndarray[FLOAT_t, ndim = 2] buy_df = np.zeros((n_exp_period,
                                                      n_stock))
    cnp.ndarray[FLOAT_t, ndim = 2] sell_df = np.zeros((n_exp_period,
                                                       n_stock))
    cnp.ndarray[FLOAT_t, ndim = 2] risk_df = np.zeros((n_exp_period,
                                                       n_stock))
    cnp.ndarray[FLOAT_t, ndim = 1] risk_free_arr = np.zeros(n_exp_period)
    cnp.ndarray[FLOAT_t, ndim = 1] var_arr = np.zeros(n_exp_period)

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
                scenario_expectation = sum(model.Ys[tdx, sdx]
                                           for sdx in xrange(n_scenario)) / float(
                    n_scenario)

                cvar_expr = (model.Z[tdx] - scenario_expectation /
                             (1. - model.alphas[adx])) / float(n_scenario)
                cvar_expr_sum = cvar_expr_sum + cvar_expr
                print "CVaR tdx:{} OK".format(tdx)
            return cvar_expr_sum
            # Tdx = n_exp_period - 1
            # scenario_expectation = sum(model.Ys[Tdx, sdx]
            #     for sdx in xrange(n_scenario)) / float(n_scenario)
            # return (model.Z[Tdx] - 1. / (1. - model.alphas[adx]) *
            #         scenario_expectation)


        instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                            sense=maximize)
        # solve
        print "start solving:"
        opt = SolverFactory(solver)
        # if solver == "cplex":
        #     opt.options["workmem"] = 4096
        # turnoff presolve
        # opt.options['preprocessing_presolve'] = 'n'
        # Barrier algorithm and its upper bound
        # opt.options['lpmethod'] = 4
        # opt.options['barrier_limits_objrange'] =1e75
        # results = opt.solve(instance, tee=True)
        results = opt.solve(instance)
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