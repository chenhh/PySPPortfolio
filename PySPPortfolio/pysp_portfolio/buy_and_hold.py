# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from __future__ import division
import os
from time import time
from datetime import date
import numpy as np
import pandas as pd
from base_model import (PortfolioReportMixin, ValidPortfolioParameterMixin)
from PySPPortfolio.pysp_portfolio import *


class BAHPortfolio(PortfolioReportMixin, ValidPortfolioParameterMixin):

    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth, initial_risk_free_wealth,
                 buy_trans_fee=0.001425, sell_trans_fee=0.004425,
                 start_date=date(2005, 1, 3), end_date=date(2014, 12, 31),
                 verbose=False):
        """
        buy and hold portfolio
        that is, one invests wealth among a pool of assets with an initial
        portfolio b1 and holds the portfolio until the end.

        Parameters:
        -------------
        symbols: list of symbols, size: n_stock
        risk_rois: pandas.DataFrame, shape: (n_period, n_stock)
        risk_free_rois: pandas.series, shape: (n_exp_period, )
        initial_risk_wealth: pandas.series, shape: (n_stock,)
        initial_risk_free_wealth: float
        buy_trans_fee: float, 0<=value < 1,
            the fee will not change in the simulation
        sell_trans_fee: float, 0<=value < 1, the same as above
        start_date: datetime.date, first date of simulation
        end_date: datetime.date, last date of simulation
        verbose: boolean
        """
        self.symbols = symbols
        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        # valid number of symbols
        self.valid_dimension("n_symbol", len(symbols),
                             len(initial_risk_wealth))
        self.initial_risk_wealth = initial_risk_wealth
        self.initial_risk_free_wealth = initial_risk_free_wealth

        # valid transaction fee
        self.buy_trans_fee = buy_trans_fee
        self.valid_trans_fee(buy_trans_fee)
        self.valid_trans_fee(sell_trans_fee)
        self.sell_trans_fee = sell_trans_fee

        self.verbose = verbose

        # .loc() will contain the end_date element
        self.valid_trans_date(start_date, end_date)
        self.exp_risk_rois = risk_rois.loc[start_date:end_date]
        self.exp_risk_free_rois = risk_free_rois.loc[start_date:end_date]
        self.n_exp_period = self.exp_risk_rois.shape[0]
        self.n_stock = self.exp_risk_rois.shape[1]

        # date index in total data
        self.start_date_idx = self.risk_rois.index.get_loc(
            self.exp_risk_rois.index[0])

        # results data
        # risk wealth DataFrame, shape: (n_exp_period, n_stock)
        self.risk_wealth_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # risk_free Series, shape: (n_exp_period, )
        self.risk_free_wealth = pd.Series(np.zeros(self.n_exp_period),
                                          index=self.exp_risk_free_rois.index)

        # buying amount DataFrame, shape: (n_exp_period, n_stock)
        self.buy_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # selling amount DataFrame, shape: (n_exp_period, n_stock)
        self.sell_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        # cumulative loss in transaction fee in the simulation
        self.trans_fee_loss = 0


    def get_trading_func_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        func_name: str, Function name of the class
        """
        return "BAH_m{}".format(self.n_stock)


    def run(self):
        """
        run simulation

        Returns:
        ----------------
        standard report
        """
        t0 = time()

        # get function name
        func_name = self.get_trading_func_name()

        # current wealth of each stock in the portfolio
        # at the first period, allocated fund uniformly to each stock
        allocated_risk_wealth = self.initial_risk_wealth
        allocated_risk_free_wealth = self.initial_risk_free_wealth

        for tdx in xrange(self.n_exp_period):
            t1 = time()
            if tdx == 0:
                # uniformly allocation fund
                 buy_amounts = pd.Series(
                    np.ones(self.n_stock) *
                    self.initial_risk_free_wealth/self.n_stock,
                    iindex=self.symbols)
                 self.buy_amounts_df.iloc[tdx] = buy_amounts
                 buy_amounts_sum = buy_amounts.sum()
                 sell_amounts_sum = 0

            elif tdx == self.n_exp_period - 1:
                # sell all stocks at the last period
                sell_amounts = self.risk_wealth_df.iloc[tdx-1]
                sell_amounts_sum = sell_amounts.sum()
                buy_amounts_sum = 0

            self.trans_fee_loss += (
                buy_amounts_sum * self.buy_trans_fee +
               sell_amounts_sum * self.sell_trans_fee
            )

            total_buy = (buy_amounts_sum * (1 + self.buy_trans_fee))
            total_sell = ( sell_amounts_sum * (1 - self.sell_trans_fee))

            # capital allocation
            self.risk_wealth_df.iloc[tdx] = (
                (1 + self.exp_risk_rois.iloc[tdx]) *
                allocated_risk_wealth +
                self.buy_amounts_df.iloc[tdx] - self.sell_amounts_df.iloc[tdx]
            )
            self.risk_free_wealth.iloc[tdx] = (
                (1 + self.exp_risk_free_rois.iloc[tdx]) *
                allocated_risk_free_wealth -
                total_buy + total_sell
            )

            # update wealth
            allocated_risk_wealth = self.risk_wealth_df.iloc[tdx]
            allocated_risk_free_wealth = self.risk_free_wealth.iloc[tdx]

            print ("[{}/{}] {} {} OK, "
                  "current_wealth:{:.2f}, {:.3f} secs".format(
                    tdx + 1, self.n_exp_period,
                    self.exp_risk_rois.index[tdx].strftime("%Y%m%d"),
                    func_name,
                    (self.risk_wealth_df.iloc[tdx].sum() +
                     self.risk_free_wealth.iloc[tdx]),
                    time() - t1))

        # end of iterations, computing statistics
        final_wealth = (self.risk_wealth_df.iloc[-1].sum() +
                        self.risk_free_wealth[-1])

        # get reports
        output, reports = self.get_performance_report(
            func_name,
            self.symbols,
            self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[-1],
            self.buy_trans_fee,
            self.sell_trans_fee,
            (self.initial_risk_wealth.sum() + self.initial_risk_free_wealth),
            final_wealth,
            self.n_exp_period,
            self.trans_fee_loss,
            self.risk_wealth_df,
            None)

        # model additional elements to reports
        reports['risk_free_wealth'] = self.risk_free_wealth

        # add simulation time
        reports['simulation_time'] = time() - t0

        if self.verbose:
            print (output)
        print ("{} OK n_stock:{}, [{}-{}], {:.4f}.secs".format(
            func_name, self.n_stock,
            self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[-1],
            time() - t0))

        return reports



def buy_and_hold(n_stock, buy_trans_fee=BUY_TRANS_FEE,
                 sell_trans_fee=SELL_TRANS_FEE):
    """
    The Buy-And-Hold (BAH) strategy,
    """
    # read rois panel
    roi_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')
    if not os.path.exists(roi_path):
        raise ValueError("{} roi panel does not exist.".format(roi_path))

    symbols = EXP_SYMBOLS[:n_stock]

    # shape: (n_period, n_stock, {'simple_roi', 'close_price'})
    roi_panel = pd.read_pickle(roi_path)

    # shape: (n_period, n_stock)
    exp_risk_rois = roi_panel.loc[START_DATE:END_DATE, symbols,
                    'simple_roi'].T
    n_exp_period = exp_risk_rois.shape[0]

    initial_wealth = 1e6
    # shape: (n_exp_period, n_stock)
    risk_wealth_df = pd.DataFrame(np.zeros((n_exp_period, n_stock)),
                                  index=exp_risk_rois.index,
                                  columns=symbols)

    # initial allocation
    risk_wealth_df.iloc[0] = (initial_wealth/n_stock) * (1. - BUY_TRANS_FEE)

    for tdx in xrange(1, n_exp_period):
        trans_date = exp_risk_rois.index[tdx]
        risk_wealth_df.iloc[tdx] = (risk_wealth_df.iloc[tdx-1] *
                                    exp_risk_rois.iloc[tdx])
        p_roi = (risk_wealth_df.iloc[tdx].sum()/
                 risk_wealth_df.iloc[tdx-1].sum() - 1)
        print ("[{}/{}] {}: portfolio roi: {:.2%}".format(
            tdx+1, n_exp_period, trans_date.strftime("%Y%m%d"), p_roi))



