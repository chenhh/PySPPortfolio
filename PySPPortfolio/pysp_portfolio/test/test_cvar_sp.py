# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from time import time
import numpy as np
from PySPPortfolio.pysp_portfolio.min_cvar_sp import (
    all_scenarios_min_cvar_sp_portfolio,)

def test_all_scenarios_min_cvar_sp():
    n_period, n_stock, n_scenario = 10, 50, 20
    initial_money = 1e6

    symbols = np.arange(n_stock)
    risk_rois = np.random.randn(n_period, n_stock)
    risk_free_roi = np.zeros(n_period)
    allocated_risk_wealth = np.zeros(n_stock)
    allocated_risk_free_wealth = initial_money
    buy_trans_fee =  0.001425
    sell_trans_fee = 0.004425
    alpha = 0.05
    predict_risk_rois = np.random.randn(n_period, n_stock, n_scenario)
    predict_risk_free_roi = 0

    # model
    t0 = time()
    all_scenarios_min_cvar_sp_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth,
                          allocated_risk_free_wealth, buy_trans_fee,
                          sell_trans_fee, alpha, predict_risk_rois,
                          predict_risk_free_roi, n_scenario)
    print "all_scenarios_min_cvar_sp_portfolio: "
    print "(n_period, n_stock, n_scenarios):({}, {}, {}): {:.4f} secs".format(
        n_period, n_stock, n_scenario, time() - t0
    )

if __name__ == '__main__':
    test_all_scenarios_min_cvar_sp()
