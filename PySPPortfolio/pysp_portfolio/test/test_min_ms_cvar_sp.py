# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from time import time
import numpy as np
import pandas as pd
from PySPPortfolio.pysp_portfolio import *
from PySPPortfolio.pysp_portfolio.min_ms_cvar_sp import (
    min_ms_cvar_sp_portfolio,)

def test_min_ms_cvar_sp():
    n_period, n_stock, n_scenario = 250, 50, 200
    initial_money = 1e6

    symbols = EXP_SYMBOLS[:n_stock]
    trans_dates = pd.date_range('2000/1/1', periods=n_period)

    risk_rois = np.random.randn(n_period, n_stock)
    risk_free_roi = np.zeros(n_period)
    allocated_risk_wealth = np.zeros(n_stock)
    allocated_risk_free_wealth = initial_money
    buy_trans_fee =  0.001425
    sell_trans_fee = 0.004425
    alphas = [0.1, 0.2, 0.3]
    predict_risk_rois = np.random.randn(n_period, n_stock, n_scenario)
    predict_risk_free_rois = np.zeros(n_period)

    # model
    t0 = time()
    res = min_ms_cvar_sp_portfolio(symbols, trans_dates, risk_rois,
                                  risk_free_roi,
                          allocated_risk_wealth,
                          allocated_risk_free_wealth, buy_trans_fee,
                          sell_trans_fee, alphas, predict_risk_rois,
                          predict_risk_free_rois, n_scenario, verbose=False)

    print res
    print "all_scenarios_min_cvar_sp_portfolio: "
    print "(n_period, n_stock, n_scenarios):({}, {}, {}): {:.4f} secs".format(
        n_period, n_stock, n_scenario, time() - t0
    )

if __name__ == '__main__':
    test_min_ms_cvar_sp()
