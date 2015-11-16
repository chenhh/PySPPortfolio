# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

import os
from datetime import date
from time import time
import numpy as np
import pandas as pd
import statsmodels.tsa as smtsa
import statsmodels.stats.stattools as smstattools

from arch.bootstrap import (SPA, )
from arch.unitroot import (DFGLS, PhillipsPerron, KPSS)


def sharpe(series):
    """
    Sharpe ratio
    note: the numpy std() function is the population estimator

    Parameters:
    ---------------
    series: list or numpy.array, ROI series
    """
    s = np.asarray(series)
    try:
        val = s.mean() / s.std()
    except FloatingPointError:
        # set 0 when standard deviation is zero
        val = 0
    return val


def sortino_full(series, mar=0):
    """
    Sortino ratio, using all periods of the series

    Parameters:
    ---------------
    series: list or numpy.array, ROI series
    mar: float, minimum acceptable return, usually set to 0
    """
    s = np.asarray(series)
    mean = s.mean()
    semi_std = np.sqrt(((s * ((s - mar) < 0)) ** 2).mean())
    try:
        val = mean / semi_std
    except FloatingPointError:
         # set 0 when semi-standard deviation is zero
        val = 0
    return val, semi_std


def sortino_partial(series, mar=0):
    """
    Sortino ratio, using only negative roi periods of the series

    Parameters:
    ---------------
    series: list or numpy.array, ROI series
    mar: float, minimum acceptable return, usually set to 0
    """
    s = np.asarray(series)
    mean = s.mean()
    n_neg_period = ((s - mar) < 0).sum()
    try:
        semi_std = np.sqrt(((s * ((s - mar) < 0)) ** 2).sum() / n_neg_period)
        val = mean / semi_std
    except FloatingPointError:
        # set 0 when semi-standard deviation or negative period is zero
        val, semi_std = 0, 0
    return val, semi_std


def maximum_drawdown(series):
    """
    https://en.wikipedia.org/wiki/Drawdown_(economics)
    the peak may be zero
    e.g.
    s= [0, -0.4, -0.2, 0.2]
    peak = [0, 0, 0, 0.2]
    therefore we don't provide relative percentage of mdd

    Parameters:
    ---------------
    series: list or numpy.array, ROI series
    """
    s = np.asarray(series)
    peak = pd.expanding_max(s)

    # absolute drawdown
    ad = np.maximum(peak - s, 0)
    mad = np.max(ad)

    return mad


def roi_series_statistics(symbol, start_date, end_date):
    """
    :param symbol: string, symbol for computing return statistics
    :param start_date: datetime.date
    :param end_date: datetime.date
    :return: {cumulative_return, annualized_return, daily_mean_return,
    daily_stdev, daily_skewness, daily_ex_kurtosis, daily_shapre,
    daily_sortino_full, daily_sortino_partial, JB, ADF)
    """
    # read pkl
    from ipro.dev import (STOCK_PKL_DIR, )

    pkl_path = os.path.join(STOCK_PKL_DIR, 'panel_largest50stocks.pkl')
    panel = pd.read_pickle(pkl_path)
    df = panel.loc[start_date:end_date, str(symbol), :].T

    # the roi is shown in percentage
    rois = df['adj_roi'] / 100.
    open_prices = df['open_price']
    high_prices = df['high_price']
    low_prices = df['low_price']
    close_prices = df['close_price']

    reports = {}

    # cumulative return
    cum_rois = (rois + 1).cumprod()
    reports['cum_roi'] = cum_rois[-1] - 1

    # annualized return
    years = (end_date.year - start_date.year) + 1
    ann_roi = np.power(reports['cum_roi'] + 1, 1. / years) - 1
    reports['ann_roi'] = ann_roi

    # daily return mean, stdev, skewness, ex_kurtosis
    reports['daily_mean_roi'] = rois.mean()
    reports['daily_std_roi'] = rois.std()

    # skew and kurtosis are not affected by the unit (%)
    reports['daily_skew_roi'] = rois.skew()
    reports['daily_kurt_roi'] = rois.kurt()

    # sharpe, sortino_full, sortino_parital
    reports['sharpe'] = sharpe(rois)
    reports['sortino_full'] = sortino_full(rois)[0]
    reports['sortino_full_semistd'] = sortino_full(rois)[1]
    reports['sortino_partial'] = sortino_partial(rois)[0]
    reports['sortino_partial_semistd'] = sortino_partial(rois)[1]

    # mdd, mad
    reports['max_dropdown_open_price'], reports['max_abs_drawdown_open_price'] = \
        maximum_drawdown(open_prices)
    reports['max_dropdown_high_price'], reports['max_abs_drawdown_high_price'] = \
        maximum_drawdown(high_prices)
    reports['max_dropdown_low_price'], reports['max_abs_drawdown_low_price'] = \
        maximum_drawdown(low_prices)
    reports['max_dropdown_close_price'], reports[
        'max_abs_drawdown_close_price'] = \
        maximum_drawdown(close_prices)

    # Jarque Bera test
    res = smstattools.jarque_bera(rois)
    reports['JB'] = res[0]
    reports['JB_pvalue'] = res[1]

    # Augumented DF test
    res = smtsa.stattools.adfuller(rois, autolag="AIC")
    reports['ADF'] = res[0]
    reports['ADF_pvalue'] = res[1]


    # DF-GLS, an improvement of ADF
    res = DFGLS(rois)
    reports['DFGLS'] = res.stat
    reports['DFGLS_pvalue'] = res.pvalue

    # PhillipsPerron
    res = PhillipsPerron(rois)
    reports['PP'] = res.stat
    reports['PP_pvalue'] = res.pvalue

    # KPSS
    res = KPSS(rois)
    reports['KPSS'] = res.stat
    reports['KPSS_pvalue'] = res.pvalue

    # SPA test
    spa = SPA(rois, np.zeros(rois.size), reps=1000)
    spa.seed(np.random.randint(0, 2 ** 31 - 1))
    spa.compute()
    reports['SPA_c_pvalue'] = spa.pvalues[1]

    return reports



