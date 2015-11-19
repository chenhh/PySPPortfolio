# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

import os
import pandas as pd
from PySPPortfolio.pysp_portfolio import *
from min_cvar_sp import (MinCVaRSPPortfolio,)


def run_min_cvar_sp_simulation(n_stock, win_length, n_scenario=200,
                               bias=False, scenario_cnt=1, alpha=0.95):
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
    n_stock, win_length,  = int(n_stock), int(win_length)
    n_scenario, alpha = int(n_scenario), float(alpha)

    # getting experiment symbols
    symbols = EXP_SYMBOLS[:n_stock]
    param = "{}_{}_m{}_w{}_s{}_{}_{}".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
            n_stock, win_length, n_scenario, bias, scenario_cnt)

    # read rois panel
    roi_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')
    if not os.path.exists(roi_path):
        raise ValueError("{} roi panel does not exist.".format(roi_path))

    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # read scenarios panel
    scenario_path = os.path.join(EXP_SCENARIO_DIR, "{}.pkl".format(param))

    if not os.path.exists(scenario_path):
        raise ValueError("{} scenario not exists.".format(scenario_path))

    # shape: (n_exp_period, n_stock, n_scenario)
    scenario_panel = pd.read_pickle(scenario_path)

    # results data
    risk_rois = generate_rois_df(symbols)

    exp_risk_rois = risk_rois.loc[start_date:end_date]
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6

    instance = MinCVaRSPPortfolio(symbols, risk_rois, risk_free_rois,
                           initial_risk_wealth, initial_risk_free_wealth,
                           alpha=alpha, verbose=False)

    reports = instance.run()
    print reports

    file_name = '{}_SP_{}-{}_m{}_w{}_a{:.2f}_s{}.pkl'.format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        exp_risk_rois.index[0].strftime("%Y%m%d"),
        exp_risk_rois.index[-1].strftime("%Y%m%d"),
        len(symbols),
        win_length,
        alpha,
        n_scenario)

    file_dir = os.path.join(EXP_SP_PORTFOLIO_DIR, 'cvar_sp')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))


    return reports

def run_min_cvar_sip_simulation(max_portfolio_size, window_length, alpha,
                             n_scenario=200):
    """
    :return: reports
    """
    from ipro.dev import (EXP_SYMBOLS, DROPBOX_UP_EXPERIMENT_DIR)

    max_portfolio_size = int(max_portfolio_size)
    window_length = int(window_length)
    alpha = float(alpha)

    symbols = EXP_SYMBOLS
    n_stock = len(symbols)
    risk_rois = generate_rois_df(symbols)
    start_date = date(2005, 1, 1)
    end_date = date(2015, 4, 30)

    exp_risk_rois = risk_rois.loc[start_date:end_date]
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6

    obj = MinCVaRSIPPortfolio(symbols, max_portfolio_size,
                            risk_rois, risk_free_rois,
                            initial_risk_wealth,
                            initial_risk_free_wealth, start_date=start_date,
                            end_date=end_date, window_length=window_length,
                            alpha=alpha, n_scenario=n_scenario, verbose=False)

    reports = obj.run()
    print reports

    file_name = '{}_SIP_{}-{}_m{}_mc{}_w{}_a{:.2f}_s{}.pkl'.format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        exp_risk_rois.index[0].strftime("%Y%m%d"),
        exp_risk_rois.index[-1].strftime("%Y%m%d"),
        max_portfolio_size,
        len(symbols),
        window_length,
        alpha,
        n_scenario)

    file_dir = os.path.join(DROPBOX_UP_EXPERIMENT_DIR, 'cvar_sip')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))

    return reports



if __name__ == '__main__':
    import sys
    import argparse

    sys.path.append(os.path.join(os.path.abspath('..'), '..'))
    parser = argparse.ArgumentParser()

    test_min_cvar_sp_portfolio()

    # parser.add_argument("-m", "--n_stock", required=True, type=int,
    #                     choices=range(5, 55, 5))
    # parser.add_argument("-w", "--win_length", required=True, type=int)
    # parser.add_argument("-a", "--alpha", required=True)
    # args = parser.parse_args()
    #
    # run_min_cvar_sp_simulation(args.n_stock, args.win_length, args.alpha)