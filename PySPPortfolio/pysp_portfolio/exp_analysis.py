# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
import os
import pandas as pd
import numpy as np
from PySPPortfolio.pysp_portfolio import *
from gen_results import (all_experiment_parameters,)

def load_results(prob_type, n_stock, win_length, n_scenario=200,
                     bias=False, scenario_cnt=1, alpha=0.95):
    """
    Parameters:
    -----------------------------------------------------
    prob_type: str, {min_cvar_sp, min_cvar_sip}
    n_stock: integer
    win_length: integer
    n_scenario: integer
    bias: boolean
    scenario_cnt: integer
    alpha: float

    """
    if prob_type == "min_cvar_sp":
        param = "{}_{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock, win_length, n_scenario, "biased" if bias else "unbiased",
        scenario_cnt, alpha)
    elif prob_type == "min_cvar_sip":
        param = "{}_{}_all{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        len(EXP_SYMBOLS), n_stock, win_length, n_scenario,
        "biased" if bias else "unbiased", scenario_cnt, alpha)
    else:
        raise ValueError('unknown prob_type: {}'.format(prob_type))

    # read results
    file_name = '{}_{}.pkl'.format(prob_type, param)
    file_path = os.path.join(EXP_SP_PORTFOLIO_DIR, prob_type, file_name)
    if not os.path.exists(file_path):
        print ("results {} not exists.".format(file_path))
        return None

    # cum_roi to df
    results = pd.read_pickle(file_path)
    return results

def all_results_to_dataframe(sheet="alpha"):
    """
    n_stock: {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}: length 10
    win_length: {50, 60, ..., 240}, length: 20
    n_scenario:  200
    biased: {unbiased,}
    cnt: {1,2,3}
    alpha: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                                  0.85, 0.9, 0.95): length: 10

    the sheet can be {n_stock, win_length, alpha}

    """
    if not sheet in ("n_stock", "win_length", "alpha"):
        raise ValueError('{} cannot be sheet'.format(sheet))


    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')
    name = ["m{}_w{}_s200_unbiased_{}".format(n_stock, win_length, cnt)
                    for cnt in (1,2,3)
                    for n_stock in xrange(5,50 + 5, 5)
                    for win_length in xrange(50, 240 + 10, 10)]
    columns = ['n_stock',
              'start_date', 'end_date', 'n_exp_period',
              'trans_fee_loss',
              'cum_roi', 'daily_roi', 'daily_mean_roi', 'daily_std_roi',
              'daily_kurt_roi', 'sharpe', 'sortino_full', 'sortino_partial',
              'max_abs_drawdown', 'SPA_l_pvalue', 'SPA_c_pvalue',
              'SPA_u_pvalue', 'simulation_time']



    if sheet == "alpha":
        # panel: shape:(alpha, n_stock * win_length, n_result_columns)
        results_panel = pd.Panel(
                            np.zeros((len(alphas), len(name), len(columns))),
                            items=alphas, major_axis=name, minor_axis=columns)


    params = all_experiment_parameters()

    # min_cvar_sp results
    prob_types = ('min_cvar_sp', 'min_cvar_sip')

    for m, w, n, b, c, a in params:
        bias = True if b == "biased" else False
        results = load_results('min_cvar_sp', m, w, n, bias, c, float(a))
        if results:
            major = "m{}_w{}_s200_unbiased_{}".format(m,w,c)
            for key in columns:
                results_panel.loc[a, major, key] = results[key]
            print ("{}_{} OK".format(major, a))

    results_panel.to_excel(os.path.join(TMP_DIR, 'min_cvar_sp.xlsx'))




if __name__ == '__main__':
    all_results_to_dataframe()

