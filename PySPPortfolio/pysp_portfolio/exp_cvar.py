# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

import os
from time import time
import numpy as np
import pandas as pd
from PySPPortfolio.pysp_portfolio import *
from min_cvar_sp import (MinCVaRSPPortfolio,)
from min_cvar_sip import (MinCVaRSIPPortfolio,)


def run_min_cvar_sp_simulation(n_stock, win_length, n_scenario=200,
                               bias=False, scenario_cnt=1, alpha=0.95,
                               verbose=False):
    """
    2nd stage SP simulation

    Parameters:
    -------------------
    n_stock: integer, number of stocks of the EXP_SYMBOLS to the portfolios
    window_length: integer, number of periods for estimating scenarios
    n_scenario, int, number of scenarios
    bias: bool, biased moment estimators or not
    scenario_cnt: count of generated scenarios, default = 1
    alpha: float, for conditional risk

    Returns:
    --------------------
    reports
    """
    t0 = time()
    n_stock, win_length,  = int(n_stock), int(win_length)
    n_scenario, alpha = int(n_scenario), float(alpha)

    # getting experiment symbols
    symbols = EXP_SYMBOLS[:n_stock]
    param = "{}_{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock, win_length, n_scenario, "biased" if bias else "unbiased",
        scenario_cnt, alpha)

    # read rois panel
    roi_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')
    if not os.path.exists(roi_path):
        raise ValueError("{} roi panel does not exist.".format(roi_path))

    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # shape: (n_period, n_stock)
    risk_rois =roi_panel.loc[:, symbols, 'simple_roi'].T
    exp_risk_rois = roi_panel.loc[START_DATE:END_DATE, symbols,
                    'simple_roi'].T
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6

    instance = MinCVaRSPPortfolio(symbols, risk_rois, risk_free_rois,
                           initial_risk_wealth, initial_risk_free_wealth,
                           window_length=win_length, n_scenario=n_scenario,
                           bias=bias, alpha=alpha, scenario_cnt=scenario_cnt,
                           verbose=verbose)
    reports = instance.run()

    file_name = 'min_cvar_sp_{}.pkl'.format(param)

    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'cvar_sp')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("min cvar sp {} OK, {:.3f} secs".format(param, time()-t0))

    return reports

def run_min_cvar_sip_simulation(max_portfolio_size, window_length,
                                n_scenario=200, bias=False, scenario_cnt=1,
                                alpha=0.95, verbose=False):
    """
    2nd stage SIP simulation
    in the model, all stocks are used as candidate symbols.

    Parameters:
    -------------------
    max_portfolio_size: integer, number of stocks in the portfolio.
    window_length: integer, number of periods for estimating scenarios
    n_scenario, int, number of scenarios
    bias: bool, biased moment estimators or not
    scenario_cnt: count of generated scenarios, default = 1
    alpha: float, for conditional risk

    Returns:
    --------------------
    reports
    """
    t0 = time()
    max_portfolio_size = int(max_portfolio_size)
    window_length = int(window_length)
    n_scenario = int(n_scenario)
    alpha = float(alpha)

    symbols = EXP_SYMBOLS
    n_stock = len(symbols)
    param = "{}_{}_all{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        len(symbols), max_portfolio_size, window_length, n_scenario,
        "biased" if bias else "unbiased", scenario_cnt, alpha)

    # read rois panel
    roi_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')
    if not os.path.exists(roi_path):
        raise ValueError("{} roi panel does not exist.".format(roi_path))


    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # shape: (n_period, n_stock)
    risk_rois =roi_panel.loc[:, symbols, 'simple_roi'].T
    exp_risk_rois = roi_panel.loc[START_DATE:END_DATE, symbols, 'simple_roi'].T
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6

    instance = MinCVaRSIPPortfolio(symbols, max_portfolio_size,
                            risk_rois, risk_free_rois,
                            initial_risk_wealth,
                            initial_risk_free_wealth,
                            window_length=window_length,
                            n_scenario=n_scenario,
                            bias=bias,
                            alpha=alpha,
                            scenario_cnt=scenario_cnt,
                            verbose=verbose)

    reports = instance.run()

    file_name = 'min_cvar_sip_{}.pkl'.format(param)
    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'cvar_sip')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("min cvar sip {} OK, {:.3f} secs".format(param, time()-t0))


def analysis_results(prob_type, n_stock, win_length, n_scenario=200,
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
        return

    results = pd.read_pickle(file_path)
    print results



if __name__ == '__main__':
    import sys
    import argparse

    # run_min_cvar_sp_simulation(5, 50, scenario_cnt=1, alpha=0.95,
    #                            verbose=True)
    # run_min_cvar_sip_simulation(5, 100, scenario_cnt=1, alpha=0.95,
    #                            verbose=True)

    analysis_results("min_cvar_sp", 5, 50, n_scenario=200,
                     bias=False, scenario_cnt=1, alpha=0.95)

    # parser.add_argument("-m", "--n_stock", required=True, type=int,
    #                     choices=range(5, 55, 5))
    # parser.add_argument("-w", "--win_length", required=True, type=int)
    # parser.add_argument("-a", "--alpha", required=True)
    # args = parser.parse_args()
    #
    # run_min_cvar_sp_simulation(args.n_stock, args.win_length, args.alpha)