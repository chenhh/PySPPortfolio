# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from time import time
from datetime import date
import os
import numpy as np
import pandas as pd
from PySPPortfolio.pysp_portfolio import *
from PySPPortfolio.pysp_portfolio.min_ms_cvar_avgsp import (
    min_ms_cvar_avgsp_portfolio,
)

def test_min_ms_cvar_avgsp(n_stock, win_length, alpha, scenario_cnt=1):
    """
    :param n_stock: range(5, 55)
    :param win_length:  range(50, 250)
    :param alphas: list
    :return:
    """
    t_start_date, t_end_date = date(2005, 1, 3), date(2014, 12, 31)

    symbols = EXP_SYMBOLS[:n_stock]
    # read rois panel
    roi_path = os.path.join( os.path.abspath(os.path.curdir), '..', 'data',
                             'pkl', 'TAIEX_2005_largest50cap_panel.pkl')
    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # shape: (n_period, n_stock)
    risk_rois = roi_panel.loc[t_start_date:t_end_date,
                                symbols, 'simple_roi'].T
    n_period = len(risk_rois.index)
    n_scenario = 200

    risk_free_roi = np.zeros(n_period, dtype=np.float)
    allocated_risk_wealth = np.zeros(n_stock, dtype=np.float)
    allocated_risk_free_wealth = 1e6
    buy_trans_fee =  0.001425
    sell_trans_fee = 0.004425

    # read scenario
    scenario_name = "{}_{}_m{}_w{}_s{}_{}_{}.pkl".format(
            START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
            len(symbols), win_length, n_scenario, "unbiased", scenario_cnt)
    scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios',
                                     scenario_name)
    scenario_panel = pd.read_pickle(scenario_path)

    predict_risk_rois = scenario_panel.loc[t_start_date:t_end_date]
    predict_risk_free_rois = np.zeros((n_period, n_scenario))

    # model
    t0 = time()
    res = min_ms_cvar_avgsp_portfolio(symbols, risk_rois.index,
                                   risk_rois.as_matrix(), risk_free_roi,
                                allocated_risk_wealth,
                          allocated_risk_free_wealth, buy_trans_fee,
                          sell_trans_fee, alpha,
                          predict_risk_rois.as_matrix(),
                          predict_risk_free_rois, n_scenario,
                            solver="cplex", verbose=False)

    print res
    pd.to_pickle(res, os.path.join(TMP_DIR, 'min_ms_cvar_avgsp.pkl'))
    print predict_risk_rois.mean(axis=2)
    print "all_scenarios_min_cvar_avgsp_portfolio: "
    print "(n_period, n_stock, n_scenarios):({}, {}, {}): {:.4f} secs".format(
        n_period, n_stock, 200, time() - t0
    )

if __name__ == '__main__':
    test_min_ms_cvar_avgsp(10, 70, 0.5)