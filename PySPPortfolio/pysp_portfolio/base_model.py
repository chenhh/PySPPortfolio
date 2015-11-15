# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from datetime import (date, )
from time import time
import platform
import numpy as np
import pandas as pd

from arch.bootstrap.multiple_comparrison import (SPA,)
from utils import (sharpe, sortino_full, sortino_partial, maximum_drawdown)

class PortfolioReportMixin(object):

    @staticmethod
    def get_performance_report(func_name, symbols, start_date, end_date,
                           buy_trans_fee, sell_trans_fee,
                           initial_wealth, final_wealth, n_exp_period,
                           trans_fee_loss, wealth_df, weights_df):
        """
        standard reports

        Parameters:
        ------------------
        func_name: str
        symbols: list of string
        start_date, end_date: datetime.date
        buy_trans_fee, sell_trans_fee: float
        initial_wealth, final_wealth: float
        n_exp_period: integer
        trans_fee_loss: float
        wealth_df: pandas.DataFrame, shape:(n_stock, n_exp_period)
        weights_df: pandas.DataFrame, shape:(n_stock, n_exp_period)

        """
        reports = {}
        # platform information
        reports['os_uname'] = "|".join(platform.uname())

        # return analysis
        reports['func_name'] = func_name
        reports['symbols'] = symbols
        reports['start_date'] = start_date
        reports['end_date'] = end_date
        reports['buy_trans_fee'] = buy_trans_fee
        reports['sell_trans_fee'] = sell_trans_fee
        reports['initial_wealth'] = initial_wealth
        reports['final_wealth'] = final_wealth
        reports['n_exp_period'] = n_exp_period
        reports['trans_fee_loss'] = trans_fee_loss
        reports['wealth_df'] = wealth_df
        reports['weights_df'] = weights_df

        # analysis
        reports['n_stock'] = len(symbols)
        reports['cum_roi'] = final_wealth / initial_wealth - 1.
        reports['daily_roi'] = np.power(final_wealth / initial_wealth,
                                        1. / n_exp_period) - 1

        # wealth_arr, shape: (n_stock,)
        wealth_arr = wealth_df.sum(axis=1)
        wealth_daily_rois = wealth_arr.pct_change()
        wealth_daily_rois[0] = 0

        reports['daily_mean_roi'] = wealth_daily_rois.mean()
        reports['daily_std_roi'] = wealth_daily_rois.std()
        reports['daily_skew_roi'] = wealth_daily_rois.skew()

        # excess Kurtosis
        reports['daily_kurt_roi'] = wealth_daily_rois.kurt()
        reports['sharpe'] = sharpe(wealth_daily_rois)
        reports['sortino_full'], reports['sortino_full_semi_std'] = \
            sortino_full(wealth_daily_rois)

        reports['sortino_partial'], reports['sortino_partial_semi_std'] = \
            sortino_partial(wealth_daily_rois)

        reports['max_drawdown'], reports['max_abs_drawdown'] = \
            maximum_drawdown(wealth_arr)

        # statistics test
        # SPA test, benchmark is no action
        spa = SPA(wealth_daily_rois, np.zeros(wealth_arr.size), reps=1000)
        spa.seed(np.random.randint(0, 2 ** 31 - 1))
        spa.compute()
        reports['SPA_l_pvalue'] = spa.pvalues[0]
        reports['SPA_c_pvalue'] = spa.pvalues[1]
        reports['SPA_u_pvalue'] = spa.pvalues[2]

        # outputs
        outputs = 'func: {}, [{}-{}] n_period: {}\n'.format(
            func_name, start_date, end_date, n_exp_period)

        outputs += "final wealth: {}, trans_fee_loss: {}\n".format(
            final_wealth, trans_fee_loss)

        outputs += "cum_roi:{:.6%}, daily_roi:{:.6%}\n".format(
            reports['cum_roi'], reports['daily_roi'])

        outputs += "roi (mean, std, skew, ex_kurt): "
        outputs += "({:.6%}, {:.6%}, {:.6f}, {:.6f})\n".format(
            reports['daily_mean_roi'], reports['daily_std_roi'],
            reports['daily_skew_roi'], reports['daily_kurt_roi']
        )

        outputs += "Sharpe: {:.6%}\n".format(reports['sharpe'])
        outputs += "Sortino full: ({:.6%}, {:.6%}\n".format(
            reports['sortino_full'], reports['sortino_full_semi_std'])

        outputs += "Sortino partial: ({:.6%}, {:.6%}\n".format(
            reports['sortino_partial'], reports['sortino_partial_semi_std'])

        outputs += "mdd: {:.2%}, mad:{:.4f}\n".format(
            reports['max_drawdown'], reports['max_abs_drawdown'])

        outputs += "SPA_ pvalue: [{:.4%}, {:.4%}, {:.4%}]\n".format(
            reports['SPA_l_pvalue'], reports['SPA_c_pvalue'],
            reports['SPA_u_pvalue'])

        return outputs, reports


class ValidPortfolioParameterMixin(object):
    @staticmethod
    def valid_trans_fee(trans_fee):
        """
        Parameter:
        -------------
        trans_fee: float
        """
        if not 0 <= trans_fee <= 1:
            raise ValueError("wrong trans_fee: {}".format(trans_fee))

    @staticmethod
    def valid_dimension(dim1_name, dim1, dim2):
        """
        Parameters:
        -------------
        dim1, dim2: positive integer
        dim1_name, str
        """
        if dim1 != dim2:
            raise ValueError("mismatch {} dimension: {}, {}".format(
                dim1_name, dim1, dim2))

    @staticmethod
    def valid_trans_date(start_date, end_date):
        """
        Parameters:
        --------------
        start_date, end_date: datetime.date
        """
        if start_date >= end_date:
            raise ValueError("wrong transaction interval, start:{}, "
                             "end:{})".format(start_date, end_date))


class SPTradingPortfolio(ValidPortfolioParameterMixin,
                         PortfolioReportMixin):
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
        risk_rois: pandas.DataFrame, shape: (n_period, n_stock)
        risk_free_rois: pandas.series, shape: (n_exp_period, )
        initial_risk_wealth: pandas.series, shape: (n_stock,)
        initial_risk_free_wealth: float
        buy_trans_fee: float, 0<=value < 1,
            the fee will not change in the simulation
        sell_trans_fee: float, 0<=value < 1, the same as above
        start_date: datetime.date, first date of simulation
        end_date: datetime.date, last date of simulation
        window_length: integer, historical periods for estimated parameters
        verbose: boolean

        Data
        --------------
        risk_wealth_df: pandas.DataFrame, shape: (n_exp_period, n_stock)
        risk_free_wealth_df: pandas.Series, shape: (n_exp_period,)
        buy_amount_df: pandas.DataFrame, shape: (n_exp_period, n_stock)
        sell_amount_df: pandas.DataFrame, shape: (n_exp_period, n_stock)
        estimated_risk_roi_error: pandas.Series, shape: (n_exp_period,)
            - The scenarios generating function may be failed to generate
              scenarios in some periods, we record these periods if fails.
        trans_fee_loss: float, the cumulative loss of transaction fee.

        """
        # valid number of symbols
        self.valid_dimension("n_symbol", len(symbols), risk_rois.shape[1])
        self.symbols = symbols
        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        # valid number of symbols
        self.valid_dimension("n_symbol", len(symbols), len(initial_risk_wealth))
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
        self.window_length = window_length
        self.start_date_idx = self.risk_rois.index.get_loc(
            self.exp_risk_rois.index[0])

        if self.start_date_idx < window_length:
            raise ValueError('There is no enough data for estimating '
                             'parameters.')

        # valid specific parameters added by users
        self.valid_specific_parameters()

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

        # estimating error Series, shape: (n_period,)
        # The scenarios generating function may be failed to generate
        # scenarios in some periods, we record these periods if fails.
        self.estimated_risk_roi_error = pd.Series(np.zeros(
            self.n_exp_period).astype(np.bool),
          index=self.exp_risk_rois.index)

        # cumulative loss in transaction fee in the simulation
        self.trans_fee_loss = 0

    def valid_specific_parameters(self, *args, **kwargs):
        """
        implemented by user
        the function will be called in the __init__
        """
        pass

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        estimating next period risky assets rois,
        implemented by user

        Returns:
        ----------------------------
        estimated_risk_rois: pandas.DataFrame, shape: (n_stock, n_scenario)
        """
        raise NotImplementedError('get_estimated_rois')

    def get_estimated_risk_free_roi(self, *arg, **kwargs):
        """
        estimating next period risk free asset rois,
        implemented by user, and it should return a float number.

        Returns:
        ------------------------------
        risk_free_roi: float
        """
        raise NotImplementedError('get_estimated_risk_free_roi')

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """
        stochastic programming for determining current period
        buy amounts and sell amounts by using historical data.

        implemented by user, and it must return a dict contains
        at least the following two elements:
        {
            "buy_amounts": buy_amounts, pandas.Series, shape: (n_stock, )
            "sell_amounts": sell_amounts, , pandas.Series, shape: (n_stock, )
        }
        """
        raise NotImplementedError('get_current_buy_sell_amounts')

    def get_trading_func_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        func_name: str, Function name of the class
        """
        raise NotImplementedError('get_trading_func_name')

    def set_specific_period_action(self, *args, **kwargs):
        pass

    def add_results_to_reports(self, reports):
        """ add Additional results to reports after a simulation """
        return reports

    def run(self):
        """
        run recourse programming simulation

        Returns:
        ----------------
        standard report
        """
        t0 = time()

        # get function name
        func_name = self.get_trading_func_name()

        # current wealth of each stock in the portfolio
        allocated_risk_wealth = self.initial_risk_wealth
        allocated_risk_free_wealth = self.initial_risk_free_wealth

        # count of generating scenario error
        estimated_risk_roi_error_count = 0
        for tdx in xrange(self.n_exp_period):
            t1 = time()
            # estimating next period rois, shape: (n_stock, n_scenario)
            try:
                estimated_risk_rois = self.get_estimated_risk_rois(tdx=tdx)
            except ValueError as e:
                print ("generating scenario error: {}, {}".format(
                    self.exp_risk_rois.index[tdx], e))
                self.estimated_risk_roi_error[tdx] = True

            estimated_risk_free_roi = self.get_estimated_risk_free_roi(tdx=tdx)

            # generating scenarios success
            if self.estimated_risk_roi_error[tdx] is False:

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

                # buy and sell according results, shape: (n_stock, )
                buy_amounts = results["buy_amounts"]
                sell_amounts = results["sell_amounts"]

            # generating scenarios failed
            else:
                # buy and sell nothing, shape:(n_stock, )
                buy_amounts = pd.Series(np.zeros(self.n_stock),
                                        index=self.symbols)
                sell_amounts = pd.Series(np.zeros(self.n_stock),
                                         index=self.symbols)
                estimated_risk_roi_error_count += 1

            # record buy and sell amounts
            self.buy_amounts_df.iloc[tdx] = buy_amounts
            self.sell_amounts_df.iloc[tdx] = sell_amounts

            # record the transaction loss
            buy_amounts_sum = buy_amounts.sum()
            sell_amounts_sum = sell_amounts.sum()
            self.trans_fee_loss += (
                buy_amounts_sum * self.buy_trans_fee +
               sell_amounts_sum * self.sell_trans_fee
            )

            # buy and sell amounts consider the transaction cost
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

            print ("[{}/{}] {} {} OK, estimated error count:{} " \
                  "current_wealth:{}, {:.3f} secs".format(
                    tdx + 1, self.n_exp_period,
                    func_name,
                    self.exp_risk_rois.index[tdx],
                    estimated_risk_roi_error_count,
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
        reports['buy_amounts_df'] = self.buy_amounts_df
        reports['sell_amounts_df'] = self.sell_amounts_df
        reports['estimated_risk_roi_error'] = self.estimated_risk_roi_error
        reports['estimated_risk_roi_error_count'] = \
            self.estimated_risk_roi_error.sum()

        # add simulation time
        reports['simulation_time'] = time() - t0

        # user specified  additional elements to reports
        reports = self.add_results_to_reports(reports)

        if self.verbose:
            print (output)
        print ("{} OK n_stock:{}, [{}-{}], {:.4f}.secs".format(
            func_name, self.n_stock,
            self.exp_risk_rois.index[0],
            self.exp_risk_rois.index[-1],
            time() - t0))

        return reports
