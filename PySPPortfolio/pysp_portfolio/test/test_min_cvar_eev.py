# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
import numpy as np
from PySPPortfolio.pysp_portfolio.min_cvar_sp import (
    min_cvar_sp_portfolio, min_cvar_eev_sp_portfolio)

from test_cvar import (min_cvar_sp_portfolio as  min_cvar_sp_portfolio2)

def test_min_cvar_eev_sp():
    n_stock = 10
    n_scenario = 200
    symbols = np.arange(n_stock)
    risk_rois = np.random.randn(n_stock)
    risk_free_roi = 0
    allocated_risk_wealth = np.zeros(n_stock)
    allocated_risk_free_wealth = 1e6
    buy_trans_fee = 0.001425
    sell_trans_fee = 0.004425
    alpha = 0.50
    predict_risk_rois =  np.random.randn(n_stock, n_scenario)
    predict_risk_free_roi = 0
    results =min_cvar_sp_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario)
    print results
    print results['buy_amounts'].sum()
    print "*"*50
    results2 =min_cvar_sp_portfolio2(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario)
    print results2
    print results2['buy_amounts'].sum()
    print "*"*50
    # results_eev =min_cvar_eev_sp_portfolio(symbols, risk_rois, risk_free_roi,
    #                       allocated_risk_wealth, allocated_risk_free_wealth,
    #                       buy_trans_fee, sell_trans_fee, alpha,
    #                       predict_risk_rois, predict_risk_free_roi,
    #                       n_scenario)
    #
    # print results_eev

if __name__ == '__main__':
    test_min_cvar_eev_sp()