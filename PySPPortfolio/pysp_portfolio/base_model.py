# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from datetime import (date, )
from time import time
import numpy as np
import pandas as pd

from utils import (check_common_variable, get_performance_report)


class SPTradingPortfolio(object):
    def __init__(self, symbols, risk_rois, risk_free_rois,
                 initial_risk_wealth, initial_risk_free_wealth,
                 buy_trans_fee=0.001425, sell_trans_fee=0.004425,
                 start_date=date(2005, 1, 1), end_date=date(2015, 4, 30),
                 window_length=200, verbose=False):
        """
        stepwise stochastic programming trading portfolio

        Parameters:
        -------------
        symbols: list of symbols, size: n_stock
        risk_rois: pandas.dataframe, shape: (n_period, n_stock)
        risk_free_rois: pandas.series, shape: (n_exp_period, )
        initial_risk_wealth: pandas.series, shape: (n_stock,)
        initial_risk_free_wealth: float
        buy_trans_fee: float, 0<=value < 1
        sell_trans_fee: float, 0<=value < 1
        start_date: datetime.date
        end_date: datetime.date
        window_length: integer, historical data for estimated parameters
        verbose: boolean

        Returns:
        --------------
        reports: dict
            - risk_wealth_df: pandas.DataFrame, shape: (n_exp_period, n_stock)
            - risk_free_wealth_df: pandas.Series, shape: (n_exp_period,)
            - buy_amount_df: pandas.DataFrame, shape: (n_exp_period, n_stock)
            - sell_amount_df: pandas.DataFrame, shape: (n_exp_period, n_stock)
            - estimated_risk_roi_error: pandas.Series, shape: (n_exp_period,)
                if the realized ROI of s-th periods less than the
                estimated VaR, then the s-th element is set True.
            - trans_fee_loss: float, the cumulative loss of transaction fee.

        """
        self.symbols = symbols
        self.risk_rois = risk_rois

        # valid number of symbols
        if len(symbols) != risk_rois.shape[1]:
            raise ValueError(
                "Mismatch number of symbols between symbols and risk_rois:{}, "
                "{}".format(len(symbols, risk_rois.shape[1])))

        self.risk_free_rois = risk_free_rois

        # valid number of periods
        if len(risk_free_rois) < risk_rois.shape[0]:
            raise ValueError("The experiment periods {} should be larger than "
                             "the total data periods {}.".format(
                len(risk_free_rois), risk_rois.shape[0]))

        self.initial_risk_wealth = initial_risk_wealth
        self.initial_risk_free_wealth = initial_risk_free_wealth
        self.buy_trans_fee = buy_trans_fee
        self.sell_trans_fee = sell_trans_fee
        self.verbose = verbose
        self.risk_rois = risk_rois
        self.exp_risk_rois = risk_rois.loc[start_date:end_date]
        self.exp_risk_free_rois = risk_free_rois.loc[start_date:end_date]
        self.n_exp_period = self.exp_risk_rois.shape[0]
        self.n_stock = self.exp_risk_rois.shape[1]

        # date index in total data
        self.window_length = window_length
        self.start_date_idx = self.risk_rois.index.get_loc(
            self.exp_risk_rois.index[0])

        if self.start_date_idx < window_length:
            raise ValueError('There are no enough data for estimating '
                             'parameters.')

        # check common parameters
        check_common_variable(buy_trans_fee, sell_trans_fee, start_date,
                              end_date)

        self.check_specific_parameters()

        # initial wealth and weight dataframe, shape: (n_period, n_stock)
        self.risk_wealth_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns)

        self.risk_free_wealth = pd.Series(np.zeros(self.n_exp_period),
                                          index=self.exp_risk_free_rois.index)

        self.buy_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        self.sell_amounts_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=self.exp_risk_rois.columns
        )

        self.estimated_risk_roi_error = pd.Series(np.zeros(
            self.n_exp_period).astype(np.bool),
            index=self.exp_risk_rois.index)

        self.trans_fee_loss = 0

    def check_specific_parameters(self, *args, **kwargs):
        """implemented by user"""
        pass

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        estimating next period risky assets rois,
        implemented by user

        Returns:
        ----------------------------
        A pandas.Dataframe, shape: (n_stock, n_scenario)
        """
        raise NotImplementedError('get_estimated_rois')

    def get_estimated_risk_free_roi(self, *arg, **kwargs):
        """
        estimating next period risk free asset rois,
        implemented by user, and it should return a float number.

        Returns:
        ------------------------------
        A float number
        """
        raise NotImplementedError('get_estimated_risk_free_roi')

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """
        stochastic programming for determining current period
        buy amounts and sell amounts,
        implemented by user, and it must return a dict contains
        the following elements:
        {
            "buy_amounts": buy_amounts, pandas.Series, shape: (n_stock, )
            "sell_amounts": sell_amounts, , pandas.Series, shape: (n_stock, )
        }
        """
        raise NotImplementedError('get_current_buy_sell_amounts')

    def get_trading_func_name(self, *args, **kwargs):
        """implemented by user

        Returns:
        ------------
        str, Function name of the class
        """
        raise NotImplementedError('get_trading_func_name have to implemented.')

    def set_specific_period_action(self, *args, **kwargs):
        pass

    def add_results_to_reports(self, reports):
        """ add Additional results to reports after simulation """
        return reports

    def run(self):
        """
        run simulation

        Returns:
        ----------------
        standard report
        """
        t0 = time()
        func_name = self.get_trading_func_name()

        allocated_risk_wealth = self.initial_risk_wealth
        allocated_risk_free_wealth = self.initial_risk_free_wealth

        estimated_risk_roi_error_count = 0
        for tdx in xrange(self.n_exp_period):
            t1 = time()
            # estimating next period rois, shape: (n_stock, n_scenario)
            try:
                estimated_risk_rois = self.get_estimated_risk_rois(tdx=tdx)
            except ValueError:
                self.estimated_risk_roi_error[tdx] = True

            estimated_risk_free_roi = self.get_estimated_risk_free_roi(tdx=tdx)

            if self.estimated_risk_roi_error[tdx] == False:
                # determining the buy and sell amounts
                results = self.get_current_buy_sell_amounts(
                    tdx=tdx,
                    estimated_risk_rois=estimated_risk_rois,
                    estimated_risk_free_roi=estimated_risk_free_roi,
                    allocated_risk_wealth=allocated_risk_wealth,
                    allocated_risk_free_wealth=allocated_risk_free_wealth
                )
                # record results
                self.set_specific_period_action(tdx=tdx, results=results)
                buy_amounts = results["buy_amounts"]
                sell_amounts = results["sell_amounts"]

            else:
                buy_amounts = pd.Series(np.zeros(self.n_stock
                                                 ), index=self.symbols)
                sell_amounts = pd.Series(np.zeros(self.n_stock
                                                  ), index=self.symbols)
                estimated_risk_roi_error_count += 1

            self.buy_amounts_df.iloc[tdx] = buy_amounts
            self.sell_amounts_df.iloc[tdx] = sell_amounts

            # record the transaction loss
            self.trans_fee_loss += (self.buy_amounts_df.iloc[tdx].sum() *
                                    self.buy_trans_fee +
                                    self.sell_amounts_df.iloc[tdx].sum() *
                                    self.sell_trans_fee
                                    )
            # buy and sell amounts consider the transaction cost
            total_buy = (self.buy_amounts_df.iloc[tdx].sum() *
                         (1 + self.buy_trans_fee))
            total_sell = (self.sell_amounts_df.iloc[tdx].sum() *
                          (1 - self.sell_trans_fee))

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

            print "[{}/{}] {} {} OK, estime error count:{} " \
                  "current_wealth:{}, {:.3f} secs".format(
                tdx + 1, self.n_exp_period,
                func_name,
                self.exp_risk_rois.index[tdx],
                estimated_risk_roi_error_count,
                (self.risk_wealth_df.iloc[tdx].sum() +
                 self.risk_free_wealth.iloc[tdx]),
                time() - t1)

        final_wealth = (self.risk_wealth_df.iloc[-1].sum() +
                        self.risk_free_wealth[-1])

        output, reports = get_performance_report(
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

        reports['risk_free_wealth'] = self.risk_free_wealth
        reports['buy_amounts_df'] = self.buy_amounts_df
        reports['sell_amounts_df'] = self.sell_amounts_df
        reports['estimated_risk_roi_error'] = self.estimated_risk_roi_error
        reports['estimated_risk_roi_error_count'] = \
            self.estimated_risk_roi_error.sum()

        if self.verbose:
            print output
        print "{} OK n_stock:{}, " \
              "[{}-{}], {:.4f}.secs".format(func_name, self.n_stock,
                                            self.exp_risk_rois.index[0],
                                            self.exp_risk_rois.index[-1],
                                            time() - t0)
        # add results to reports
        reports = self.add_results_to_reports(reports)

        return reports
