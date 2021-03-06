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
from min_cvar_sp import (MinCVaRSPPortfolio, MinCVaRSPPortfolio2 )
from min_cvar_sip import (MinCVaRSIPPortfolio,MinCVaRSIPPortfolio2)
from min_ms_cvar_sp import (MinMSCVaRSPPortfolio,)
from min_ms_cvar_eventsp import (MinMSCVaREventSPPortfolio,)
from min_ms_cvar_avgsp import (MinMSCVaRAvgSPPortfolio,)
from min_cvar_eev import (MinCVaREEVPortfolio,)
from min_cvar_eevip import (MinCVaREEVIPPortfolio,)
from buy_and_hold import (BAHPortfolio,)
from best import (BestMSPortfolio, BestPortfolio)
from datetime import date

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

    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'min_cvar_sp')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("min cvar sp {} OK, {:.3f} secs".format(param, time()-t0))

    return reports


def run_min_cvar_sp2_simulation(n_stock, win_length, n_scenario=200,
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

    instance = MinCVaRSPPortfolio2(symbols, risk_rois, risk_free_rois,
                           initial_risk_wealth, initial_risk_free_wealth,
                           window_length=win_length, n_scenario=n_scenario,
                           bias=bias, alpha=alpha, scenario_cnt=scenario_cnt,
                           verbose=verbose)
    reports = instance.run()

    file_name = 'min_cvar_sp2_{}.pkl'.format(param)

    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'min_cvar_sp2')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("min cvar sp2 {} OK, {:.3f} secs".format(param, time()-t0))

    return reports


def run_min_cvar_sp2_yearly_simulation(n_stock, win_length, n_scenario=200,
                               bias=False, scenario_cnt=1, alpha=0.95,
                               verbose=False, start_date=START_DATE,
                                    end_date=END_DATE):
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
        start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"),
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
    exp_risk_rois = roi_panel.loc[start_date:end_date, symbols,
                    'simple_roi'].T
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6

    instance = MinCVaRSPPortfolio2(symbols, risk_rois, risk_free_rois,
                           initial_risk_wealth, initial_risk_free_wealth,
                           window_length=win_length, n_scenario=n_scenario,
                           bias=bias, alpha=alpha, scenario_cnt=scenario_cnt,
                           verbose=verbose, start_date=start_date,
                                   end_date=end_date)
    reports = instance.run()

    file_name = 'min_cvar_sp2_yearly_{}.pkl'.format(param)

    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'min_cvar_sp2_yearly')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("min cvar sp2 yearly {} OK, {:.3f} secs".format(param, time()-t0))

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
    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'min_cvar_sip')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("min cvar sip {} OK, {:.3f} secs".format(param, time()-t0))

    return reports


def run_min_cvar_sip2_simulation(max_portfolio_size, window_length,
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

    instance = MinCVaRSIPPortfolio2(symbols, max_portfolio_size,
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

    file_name = 'min_cvar_sip2_{}.pkl'.format(param)
    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'min_cvar_sip2')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("min cvar sip2 {} OK, {:.3f} secs".format(param, time()-t0))

    return reports


def run_min_cvar_sip2_yearly_simulation(max_portfolio_size, window_length,
                                n_scenario=200, bias=False, scenario_cnt=1,
                                alpha=0.95, verbose=False,
                                start_date=START_DATE, end_date=END_DATE):
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
        start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"),
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
    exp_risk_rois = roi_panel.loc[start_date:end_date, symbols, 'simple_roi'].T
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6
    instance = MinCVaRSIPPortfolio2(symbols, max_portfolio_size,
                            risk_rois, risk_free_rois,
                            initial_risk_wealth,
                            initial_risk_free_wealth,
                            window_length=window_length,
                            n_scenario=n_scenario,
                            bias=bias,
                            alpha=alpha,
                            scenario_cnt=scenario_cnt,
                            verbose=verbose,
                            start_date=start_date,
                            end_date=end_date)

    reports = instance.run()

    file_name = 'min_cvar_sip2_yearly_{}.pkl'.format(param)
    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'min_cvar_sip2_yearly')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("min cvar sip2 yearly {} OK, {:.3f} secs".format(param, time()-t0))

    return reports



def run_min_ms_cvar_sp_simulation(n_stock, win_length, n_scenario=200,
                               bias=False, scenario_cnt=1,
                               alphas=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                                       0.8, 0.85, 0.9, 0.95],
                                  verbose=False):
    """
    multi-stage SP simulation

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
    n_scenario = int(n_scenario)

    # getting experiment symbols
    symbols = EXP_SYMBOLS[:n_stock]
    param = "{}_{}_m{}_w{}_s{}_{}_{}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock, win_length, n_scenario, "biased" if bias else "unbiased",
        scenario_cnt)

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

    print ("min ms cvar sp {} start.".format(param))
    t1 = time()
    instance = MinMSCVaRSPPortfolio(symbols, risk_rois, risk_free_rois,
                           initial_risk_wealth, initial_risk_free_wealth,
                           window_length=win_length, n_scenario=n_scenario,
                           bias=bias, alphas=alphas, scenario_cnt=scenario_cnt,
                           verbose=verbose)
    print ("min ms cvar sp {} ready to run: {:.3f} secs".format(
            param, time() - t1))
    reports_dict = instance.run()

    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'min_ms_cvar_sp')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    for alpha_str, reports in reports_dict.items():
        alpha = reports['alpha']
        file_name = 'min_ms_cvar_sp_{}_a{:.2f}.pkl'.format(param, alpha)
        pd.to_pickle(reports, os.path.join(file_dir, file_name))
        print ("ms min cvar sp {}_a{:.2f} OK, {:.3f} secs".format(
            param, alpha, time() - t0))

    return reports_dict


def run_min_ms_cvar_eventsp_simulation(n_stock, win_length, n_scenario=200,
                               bias=False, scenario_cnt=1, alpha = 0.95,
                               verbose=False, start_date=date(2005,1,3),
                               end_date=date(2014,12,31), solver_io="lp",
                                       keepfiles=False):
    """
    multi-stage event scenario SP simulation

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
    n_stock, win_length, = int(n_stock), int(win_length)
    n_scenario = int(n_scenario)

    # getting experiment symbols
    symbols = EXP_SYMBOLS[:n_stock]

    # read rois panel
    roi_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')
    if not os.path.exists(roi_path):
        raise ValueError("{} roi panel does not exist.".format(roi_path))

    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # shape: (n_period, n_stock)
    risk_rois = roi_panel.loc[:, symbols, 'simple_roi'].T
    exp_risk_rois = roi_panel.loc[start_date:end_date, symbols,
                    'simple_roi'].T
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1

    param = "{}_{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
        exp_risk_rois.index[0].strftime("%Y%m%d"),
        exp_risk_rois.index[-1].strftime("%Y%m%d"),
        n_stock, win_length, n_scenario, "biased" if bias else "unbiased",
        scenario_cnt, alpha)

    instance = MinMSCVaREventSPPortfolio(symbols, risk_rois, risk_free_rois,
                                         initial_risk_wealth,
                                         initial_risk_free_wealth,
                                         window_length=win_length,
                                         n_scenario=n_scenario,
                                         bias=bias, alpha=alpha,
                                         scenario_cnt=scenario_cnt,
                                         start_date=start_date,
                                         end_date=end_date,
                                         verbose=verbose,
                                         solver_io=solver_io,
                                         keepfiles=keepfiles
                                         )
    reports = instance.run()
    print reports
    prob_name = "min_ms_cvar_eventsp"
    file_name = '{}_{}.pkl'.format(prob_name, param)
    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, prob_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("{} {} OK, {:.3f} secs".format(prob_name, param, time() - t0))

    return reports


def run_min_ms_cvar_avgsp_simulation(n_stock, win_length, n_scenario=200,
                               bias=False, scenario_cnt=1, alpha = 0.95,
                               verbose=False):
    """
    multi-stage average scenario SP simulation
    the results are independent to the alphas

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
    n_stock, win_length, = int(n_stock), int(win_length)
    n_scenario = int(n_scenario)

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
    risk_rois = roi_panel.loc[:, symbols, 'simple_roi'].T
    exp_risk_rois = roi_panel.loc[START_DATE:END_DATE, symbols,
                    'simple_roi'].T
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6
    instance = MinMSCVaRAvgSPPortfolio(symbols, risk_rois, risk_free_rois,
                                   initial_risk_wealth,
                                   initial_risk_free_wealth,
                                   window_length=win_length,
                                   n_scenario=n_scenario,
                                   bias=bias, alpha=alpha,
                                   scenario_cnt=scenario_cnt,
                                   verbose=verbose)
    reports = instance.run()
    # the reports is the scenario simulation results, it still need to compute
    # truly wealth process by using the wealth process
    # print reports.keys()

    prob_name = "min_ms_cvar_avgsp"
    file_name = '{}_{}.pkl'.format(prob_name, param)
    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, prob_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("{} {} OK, {:.3f} secs".format(prob_name, param, time() - t0))

    return reports

def run_min_cvar_eev_simulation(n_stock, win_length, n_scenario=200,
                               bias=False, scenario_cnt=1, alpha=0.95,
                               verbose=False):
    """
    2nd stage expected of expected value simulation

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

    instance = MinCVaREEVPortfolio(symbols, risk_rois, risk_free_rois,
                           initial_risk_wealth, initial_risk_free_wealth,
                           window_length=win_length, n_scenario=n_scenario,
                           bias=bias, alpha=alpha, scenario_cnt=scenario_cnt,
                           verbose=verbose)
    reports = instance.run()

    prob_name = "min_cvar_eev"
    file_name = '{}_{}.pkl'.format(prob_name, param)
    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, prob_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("{} {} OK, {:.3f} secs".format(prob_name, param, time()-t0))

    return reports


def run_min_cvar_eevip_simulation(max_portfolio_size, window_length,
                                n_scenario=200,
                               bias=False, scenario_cnt=1, alpha=0.95,
                               verbose=False):
    """
    2nd stage expected of expected value simulation

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

    instance = MinCVaREEVIPPortfolio(symbols, max_portfolio_size,
                            risk_rois, risk_free_rois,
                            initial_risk_wealth,
                            initial_risk_free_wealth,
                            window_length=window_length,
                            n_scenario=n_scenario,
                            bias=bias,
                            alpha=alpha,
                            scenario_cnt=scenario_cnt,
                            verbose=verbose,
                            )

    reports = instance.run()

    prob_name = "min_cvar_eevip"
    file_name = '{}_{}.pkl'.format(prob_name, param)
    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, prob_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("{} {} OK, {:.3f} secs".format(prob_name, param, time()-t0))

    return reports


def run_bah_simulation(n_stock ,verbose=False):
    """
    The Buy-And-Hold (BAH) strategy,
    """
    t0 = time()
    # read rois panel
    roi_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')
    if not os.path.exists(roi_path):
        raise ValueError("{} roi panel does not exist.".format(roi_path))

    param = "{}_{}_m{}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock)

    symbols = EXP_SYMBOLS[:n_stock]
    n_stock = len(symbols)
    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # shape: (n_period, n_stock)
    exp_risk_rois = roi_panel.loc[START_DATE:END_DATE, symbols,
                    'simple_roi'].T
    n_exp_period = exp_risk_rois.shape[0]
    exp_risk_free_rois = pd.Series(np.zeros(n_exp_period),
                                   index=exp_risk_rois.index)

    allocated_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_wealth = 1e6


    instance = BAHPortfolio(symbols, exp_risk_rois, exp_risk_free_rois,
                            allocated_risk_wealth, initial_wealth,
                            start_date=START_DATE, end_date=END_DATE)
    reports = instance.run()

    file_name = 'bah_{}.pkl'.format(param)

    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'bah')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("BAH {} OK, {:.3f} secs".format(param, time()-t0))



def run_best_simulation(n_stock ,verbose=False):
    """
    The best stage-wise strategy,
    """
    t0 = time()
    # read rois panel
    roi_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')
    if not os.path.exists(roi_path):
        raise ValueError("{} roi panel does not exist.".format(roi_path))

    param = "{}_{}_m{}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock)

    symbols = EXP_SYMBOLS[:n_stock]
    n_stock = len(symbols)
    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # shape: (n_exp_period, n_stock)
    exp_risk_rois = roi_panel.loc[START_DATE:END_DATE, symbols,
                    'simple_roi'].T
    n_exp_period = exp_risk_rois.shape[0]
    # shape: (n_exp_period, )
    exp_risk_free_rois = pd.Series(np.zeros(n_exp_period),
                                   index=exp_risk_rois.index)

    allocated_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_wealth = 1e6

    instance = BestPortfolio(symbols, exp_risk_rois, exp_risk_free_rois,
                            allocated_risk_wealth, initial_wealth,
                            start_date=START_DATE, end_date=END_DATE)
    reports = instance.run()

    file_name = 'best_{}.pkl'.format(param)

    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'best')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("best {} OK, {:.3f} secs".format(param, time()-t0))



def run_best_ms_simulation(n_stock ,verbose=False):
    """
    The best multi-stage strategy,
    """
    t0 = time()
    # read rois panel
    roi_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')
    if not os.path.exists(roi_path):
        raise ValueError("{} roi panel does not exist.".format(roi_path))

    param = "{}_{}_m{}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
        n_stock)

    symbols = EXP_SYMBOLS[:n_stock]
    n_stock = len(symbols)
    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # shape: (n_period, n_stock)
    exp_risk_rois = roi_panel.loc[START_DATE:END_DATE, symbols,
                    'simple_roi'].T
    n_exp_period = exp_risk_rois.shape[0]
    exp_risk_free_rois = pd.Series(np.zeros(n_exp_period),
                                   index=exp_risk_rois.index)

    allocated_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_wealth = 1e6


    instance = BestMSPortfolio(symbols, exp_risk_rois, exp_risk_free_rois,
                            allocated_risk_wealth, initial_wealth,
                            start_date=START_DATE, end_date=END_DATE)
    reports = instance.run()

    file_name = 'best_ms_{}.pkl'.format(param)

    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'best_ms')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))
    print ("best_ms {} OK, {:.3f} secs".format(param, time()-t0))


if __name__ == '__main__':
    pass
    # for n_stock in xrange(5, 50+5, 5):
    #     run_bah_simulation(n_stock)
    # params = [
    #     (5, 100 ,0.5),
    #     #         (10, 50, 0.7),
    #           # (15, 80, 0.5), (20, 110, 0.5),
    #           # (25, 100, 0.55), (30, 110, 0.6),
    #           # (35, 110, 0.5), (40, 110, 0.5), (45, 120, 0.55),
    #           # (50 120, 0.5)
    #           ]
    #
    # for m, w, a in params:
    #     for cnt in xrange(1, 3+1):
    #         try:
    #             run_min_cvar_eev_simulation(m, w, scenario_cnt=cnt, alpha=a,
    #                            verbose=True)
    #         except ValueError as e:
    #             print e
    #             continue
    # run_min_cvar_eev_simulation(10, 220, scenario_cnt=1, alpha=0.95)
    # for m in xrange(5, 55, 5):
    #     run_bah_simulation(m)
    # run_min_cvar_sip2_simulation(10, 190, scenario_cnt=1, alpha=0.95,
    #                            verbose=True)
    # run_min_ms_cvar_avgsp_simulation(10, 200, scenario_cnt=1, alpha=0.9)
    run_min_ms_cvar_eventsp_simulation(15, 100, n_scenario=200,
                                       alpha=0.65,
                                       start_date=date(2005, 1, 3),
                                       end_date=date(2005, 1, 31),
                                       solver_io="lp",
                                       keepfiles=False)

    # analysis_results("min_cvar_sp", 5, 50, n_scenario=200,
    #                  bias=False, scenario_cnt=1, alpha=0.95)
    # run_min_ms_cvar_sp_simulation(10, 220, n_scenario=200,
    #                            bias=False, scenario_cnt=1,
    #                               alphas=[0.5, 0.55],
    #                            # alphas=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    #                            #          0.85, 0.9, 0.95],
    #                            verbose=False)
    # for n_stock in xrange(15, 50+5, 5):
    # run_best_simulation(10)
    # run_best_ms_simulation(5)
    # run_min_cvar_eevip_simulation(10, 190)
