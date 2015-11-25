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
    if prob_type in ("min_cvar_sp", "ms_min_cvar_sp", "min_cvar_eev"):
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

    n_stocks = range(5, 50 + 5, 5)
    win_lengths = range(50, 240 + 10, 10)
    cnts = range(1, 3+1)
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')
    columns = ['n_stock', 'win_length', 'alpha', "scenario_cnt",
              'start_date', 'end_date', 'n_exp_period',
              'trans_fee_loss',
              'cum_roi', 'daily_roi', 'daily_mean_roi', 'daily_std_roi',
              'daily_kurt_roi', 'sharpe', 'sortino_full', 'sortino_partial',
              'max_abs_drawdown', 'SPA_l_pvalue', 'SPA_c_pvalue',
              'SPA_u_pvalue', 'simulation_time']


    if sheet == "n_stock":
        name = ["w{}_s200_unbiased_{}_a{}".format(win_length, cnt, alpha)
                    for win_length in win_lengths
                    for cnt in cnts
                    for alpha in alphas]

        # panel: shape:(n_stock,  win_length * alpha * cnt, n_result_columns)
        results_panel = pd.Panel(
                            np.zeros((len(n_stocks), len(name), len(columns))),
                            items=n_stocks, major_axis=name,
                            minor_axis=columns)

    elif sheet == "win_length":
        name = ["m{}_s200_unbiased_{}_a{}".format(n_stock, cnt, alpha)
                    for n_stock in n_stocks
                    for cnt in cnts
                    for alpha in alphas]

        # panel: shape:(win_length,  n_stock * alpha * cnt, n_result_columns)
        results_panel = pd.Panel(
                            np.zeros((len(win_lengths), len(name),
                                      len(columns))),
                            items=win_lengths, major_axis=name,
                            minor_axis=columns)

    elif sheet == "alpha":
        name = ["m{}_w{}_s200_unbiased_{}".format(n_stock, win_length, cnt)
                    for cnt in cnts
                    for n_stock in n_stocks
                    for win_length in win_lengths]

        # panel: shape:(alpha, n_stock * win_length * cnt, n_result_columns)
        results_panel = pd.Panel(
                            np.zeros((len(alphas), len(name), len(columns))),
                            items=alphas, major_axis=name, minor_axis=columns)

    params = all_experiment_parameters()
    n_param = len(params)

    # min_cvar_sp results
    # prob_types = ('min_cvar_sp', 'min_cvar_sip')
    prob_types = ('min_cvar_sp',)

    for prob_type in prob_types:
        for rdx, (m, w, n, b, c, a) in enumerate(params):
            bias = True if b == "biased" else False
            results = load_results(prob_type, m, w, n, bias, c, float(a))
            if results:
                if sheet == "n_stock":
                    major = "w{}_s200_unbiased_{}_a{}".format(w, c, a)
                    item_key = m


                elif sheet == "win_length":
                    major = "m{}_s200_unbiased_{}_a{}".format(m, c, a)
                    item_key = w

                elif sheet == "alpha":
                    major = "m{}_w{}_s200_unbiased_{}".format(m,w,c)
                    item_key = a

                for col_key in columns:
                    if col_key not in ('win_length', 'scenario_cnt'):
                        results_panel.loc[item_key, major, col_key] = results[
                            col_key]
                    else:
                        results_panel.loc[item_key, major, 'win_length'] = w
                        results_panel.loc[item_key, major, 'scenario_cnt'] = c

                print ("[{}/{}] {}: {}_{} OK".format(
                    rdx +1, n_param, sheet , major, a))

        results_panel.to_excel(os.path.join(TMP_DIR,
                                    '{}_{}.xlsx'.format(prob_type, sheet)))


def all_results_to_4dpanel(prob_type="min_cvar_sp"):
    """
    axis_0: n_stock
    axis_1: win_length
    axis_2: alpha
    axis_3: cnt + columns
    """
    cnts = range(1, 3+1)
    n_stocks = ["m{}".format(str(v)) for v in range(5, 50 + 5, 5)]
    win_lengths = ["w{}".format(str(v)) for v in range(50, 240 + 10, 10)]
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')
    columns = ['n_stock', 'win_length', 'alpha', 'scenario_cnt',
              'start_date', 'end_date', 'n_exp_period',
              'trans_fee_loss',
              'cum_roi', 'daily_roi', 'daily_mean_roi', 'daily_std_roi',
              'daily_kurt_roi', 'sharpe', 'sortino_full', 'sortino_partial',
              'max_abs_drawdown', 'SPA_l_pvalue', 'SPA_c_pvalue',
              'SPA_u_pvalue', 'simulation_time']

    params = all_experiment_parameters()
    n_param = len(params)

    # 3 4d-panel
    panels = [pd.Panel4D(np.zeros((len(n_stocks),len(win_lengths),
                                    len(alphas), len(columns))),
                           labels=n_stocks, items=win_lengths,
                           major_axis=alphas, minor_axis=columns)
              for _ in cnts]

    for rdx, (m, w, n, b, c, a) in enumerate(params):
        bias = True if b == "biased" else False
        results = load_results(prob_type, m, w, n, bias, c, float(a))
        stock_key = "m{}".format(m)
        win_key = "w{}".format(w)
        for col_key in columns:
            if col_key in ('win_length', 'scenario_cnt'):
                # additioanl columns
                panels[c-1].loc[stock_key, win_key, a, 'win_length'] = w
                panels[c-1].loc[stock_key, win_key, a, 'scenario_cnt'] = c
            else:
                panels[c-1].loc[stock_key, win_key, a, col_key] = results[
                    col_key]

        print ("[{}/{}] {} OK".format(rdx+1, n_param, results['func_name']))

    for cnt in xrange(3):
        file_name = "{}_exp_results_{}.pkl".format(prob_type, cnt+1)
        file_path = os.path.join(TMP_DIR, file_name)
        results = {}
        results['labels'] = panels[cnt].labels
        results['items'] = panels[cnt].items
        results['major_axis'] = panels[cnt].major_axis
        results['minor_axis'] = panels[cnt].minor_axis
        results['data'] = panels[cnt].as_matrix()
        pd.to_pickle(results, file_path)


def plot_results(prob_type="min_cvar_sp", scenario_cnt=1):
    """

    :param prob_type:
    :return:
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D



    if not prob_type in ('min_cvar_sp', 'min_cvar_sip'):
        raise ValueError('unknown problem type:{}'.format(prob_type))

    file_path = os.path.join(EXP_SP_PORTFOLIO_DIR, "reports",
                             "{}_exp_results_{}.pkl".format(prob_type,
                                                            scenario_cnt))
    results = pd.read_pickle(file_path)
    panel =  pd.Panel4D(results['data'], labels=results['labels'],
                        items=results['items'],
                        major_axis=results['major_axis'],
                        minor_axis=results['minor_axis'])
    # n_stock, win_length, alpha, columns
    roi_df = panel.loc[:, "w50", :, 'cum_roi']
    print roi_df
    roi_df.plot(kind='bar')
    plt.show()



if __name__ == '__main__':
    # all_results_to_dataframe("n_stock")
    # all_results_to_dataframe("win_length")
    # all_results_to_dataframe("alpha")
    # all_results_to_4dpanel(prob_type="min_cvar_sp")
    # plot_results()
    reports = load_results("min_cvar_eev", 5, 100, alpha=0.5)
    print reports
    # wdf = reports['wealth_df']
    # wfree = reports['risk_free_wealth']
    # warr = wdf.sum(axis=1) + wfree
    # warr[0] = 0
    # print warr
    # print warr.pct_change()
