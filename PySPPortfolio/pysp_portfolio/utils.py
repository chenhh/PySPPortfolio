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
    ROI series
    the numpy std() function is the population estimator
    """
    s = np.asarray(series)
    try:
        val = s.mean() / s.std()
    except FloatingPointError:
        val = 0
    return val


def sortino_full(series, mar=0):
    """
    ROI series
    :mar:, minimum acceptable return
    """
    s = np.asarray(series)
    mean = s.mean()
    semi_std = np.sqrt(((s * ((s - mar) < 0)) ** 2).mean())
    try:
        val = mean / semi_std
    except FloatingPointError:
        val = 0
    return val, semi_std


def sortino_partial(series, mar=0):
    """
    ROI series
    :mar:, minimum acceptable return
    """
    s = np.asarray(series)
    mean = s.mean()
    n_neg_period = ((s - mar) < 0).sum()
    try:
        semi_std = np.sqrt(((s * ((s - mar) < 0)) ** 2).sum() / n_neg_period)
        val = mean / semi_std
    except FloatingPointError:
        val, semi_std = 0, 0
    return val, semi_std


def maximum_drawdown(series):
    """
    https://en.wikipedia.org/wiki/Drawdown_(economics)

    :param series: numpy.array
    :return:
    """
    s = np.asarray(series)
    peak = pd.expanding_max(s)

    # absolute drawdown
    ad = np.maximum(peak - s, 0)
    mad = np.max(ad)

    # drawdown, if peak == 0, return 0
    with np.errstate(divide='ignore'):
        res = s / peak
        res[peak==0] = 1
        dd = np.maximum(1. - res, 0)
        mdd = np.max(dd)

    return mdd, mad


def KL_divergence(vec1, vec2):
    '''
    KL divergence as distance function of two vectors
    :param vec1: numpy.array
    :param vec2: numpy.array
    :return: float
    '''
    return (vec1 * np.log(vec1/vec2)).sum()


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


def roi_stats_to_csv(year=None):
    t0 = time()
    from ipro.dev import (EXP_SYMBOLS, )

    if not year:
        start_date = date(2005, 1, 1)
        end_date = date(2014, 12, 31)
    else:
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)

    fname = "exp_return_stats_{}-{}.csv".format(start_date.strftime("%Y%m%d"),
                                                end_date.strftime("%Y%m%d"))
    fpath = os.path.join("e:", fname)

    with open(fpath, "wb") as fout:
        head = "symbol, cum, annual, daily mean, std, skew, ex_kurt,"
        head += "mdd open, mdd high, mdd low, mdd close,"
        head += "mad open, mad high, mad low, mad close,"
        head += "sharpe, sortino_f, semistd, sortino_p, semistd, "
        head += "JB(p), ADF(p), DFGLS(p), PP(p), KPSS(p), SPA_C(p) \n"
        fout.write(head)

    with open(fpath, 'ab') as fout:
        for idx, symbol in enumerate(EXP_SYMBOLS):
            t1 = time()
            reports = roi_series_statistics(symbol, start_date, end_date)

            data = "{},{},{},{},{},{},{},".format(symbol,
                                                  reports['cum_roi'],
                                                  reports['ann_roi'],
                                                  reports['daily_mean_roi'],
                                                  reports['daily_std_roi'],
                                                  reports['daily_skew_roi'],
                                                  reports['daily_kurt_roi']
                                                  )
            data += "{},{},{},{},".format(
                reports["max_dropdown_open_price"],
                reports["max_dropdown_high_price"],
                reports["max_dropdown_low_price"],
                reports["max_dropdown_close_price"]
            )
            data += "{},{},{},{},".format(
                reports["max_abs_drawdown_open_price"],
                reports["max_abs_drawdown_high_price"],
                reports["max_abs_drawdown_low_price"],
                reports["max_abs_drawdown_close_price"]
            )
            data += "{},{},{},{},{},".format(reports['sharpe'],
                                             reports['sortino_full'],
                                             reports['sortino_full_semistd'],
                                             reports['sortino_partial'],
                                             reports['sortino_partial_semistd']
                                             )
            data += "{},{},{},{},{},{}\n".format(
                reports['JB_pvalue'],
                reports['ADF_pvalue'],
                reports['DFGLS_pvalue'],
                reports['PP_pvalue'],
                reports['KPSS_pvalue'],
                reports['SPA_c_pvalue'])
            fout.write(data)
            print "[{}/{}]{} return stats OK, {:.3f} secs".format(
                idx + 1, len(EXP_SYMBOLS), symbol, time() - t1)
    print "return stats to csv OK, {:.3f} secs".format(time() - t0)


def test_generating_relative_prices_df():
    from ipro.dev import (EXP_SYMBOLS, )

    symbols = EXP_SYMBOLS
    df = generate_relative_prices_df(symbols)
    start_date = date(2015, 1, 2)
    end_date = date(2015, 1, 18)
    rpt = get_relative_price_parameters(df, start_date, end_date)
    print rpt['exp_relative_prices']


def test_mdd():
    s = np.random.rand(100)
    print maximum_drawdown(s)


def test_roi_series_statistics():
    symbol = 2330
    start_date = date(2005, 1, 1)
    end_date = date(2014, 12, 31)
    roi_series_statistics(symbol, start_date, end_date)


if __name__ == '__main__':
    import sys

    sys.path.append(os.path.join(os.path.abspath('..'), '..'))


    # test_generating_relative_prices_df()
    # test_mdd()
    test_roi_series_statistics()

    #
    # return_stats_to_csv()
    # for year in xrange(2005, 2014):
    #     return_stats_to_csv(year)
