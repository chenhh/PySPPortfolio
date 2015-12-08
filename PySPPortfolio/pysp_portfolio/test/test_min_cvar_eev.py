# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
import numpy as np
from PySPPortfolio.pysp_portfolio.min_cvar_sp import (
    min_cvar_sp_portfolio, )
from PySPPortfolio.pysp_portfolio.min_cvar_eev import (
    min_cvar_eev_portfolio,)

from test_cvar import (min_cvar_sp_portfolio as  min_cvar_sp_portfolio2)

def test_min_cvar_eev_sp():
    n_stock = 5
    n_scenario = 200
    symbols = np.arange(n_stock)
    risk_rois = np.random.randn(n_stock)/100
    risk_free_roi = 0
    allocated_risk_wealth = np.zeros(n_stock)
    allocated_risk_free_wealth = 1e6
    buy_trans_fee = 0.001425
    sell_trans_fee = 0.004425
    alpha = 0.2
    predict_risk_rois =  np.random.randn(n_stock, n_scenario)/100
    predict_risk_free_roi = 0
    # print predict_risk_rois

    results =min_cvar_sp_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario)
    # print results
    print "SP:"
    print "VaR:",results['estimated_var']
    print "CVaR:",results['estimated_cvar']
    print "*"*50
    results2 =min_cvar_eev_portfolio(symbols, risk_rois, risk_free_roi,
                          allocated_risk_wealth, allocated_risk_free_wealth,
                          buy_trans_fee, sell_trans_fee, alpha,
                          predict_risk_rois, predict_risk_free_roi,
                          n_scenario)
    # print results2
    print "EEV:"
    print "1st VaR:", results2['estimated_var']
    print "1st CVaR:", results2['estimated_cvar']
    print "buy:", results2['buy_amounts'].sum()
    print "sell:", results['sell_amounts'].sum()
    print "EEV VaR:", results2['estimated_eev_var']
    print "EEV CVaR:", results2['estimated_eev_cvar']
    print "*"*50

if __name__ == '__main__':
    test_min_cvar_eev_sp()