# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2

extract-transform-load of data
"""

from datetime import date
from time import time
from glob import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PySPPortfolio.pysp_portfolio import *
from statsmodels.stats.stattools import (jarque_bera, )
from statsmodels.tsa.stattools import (adfuller, )
from utils import (sharpe, sortino_full, sortino_partial, maximum_drawdown)
from arch.bootstrap.multiple_comparrison import (SPA,)
from arch.unitroot.unitroot import (DFGLS, PhillipsPerron, KPSS)

def cp950_to_utf8(data):
    ''' utility function in parsing csv '''
    return data.strip().decode('cp950')


def data_strip(data):
    ''' utility function in parsing csv '''
    return data.strip()


def csv_to_pkl(symbols=EXP_SYMBOLS):
    """
    extract data from csv

    Returns:
    --------------
    pandas.DataFrame, corresponding to the csv
    """
    t0 = time()
    csvs = glob(os.path.join(SYMBOLS_CSV_DIR, '*.csv'))
    for rdx, csv in enumerate(csvs):
        symbol = csv[csv.rfind(os.sep) + 1:csv.rfind('.')]
        df = pd.read_csv(open(csv),
                         index_col=("year_month_day",),
                         parse_dates=True,
                         dtype={
                             'symbol': str,
                             'abbreviation': str,
                             'year_month_day': date,
                             'open_price': np.float,
                             'high_price': np.float,
                             'low_price': np.float,
                             'close_price': np.float,
                             'volume_1000_shares': np.float,
                             'value_1000_dollars': np.float,
                             'simple_roi_%': np.float,
                             'turnover_ratio_%': np.float,
                             'market_value_million_dollars': np.float,
                             'continuous_roi_%': np.float,
                         },
                         converters={
                             'symbol': data_strip,
                             'abbreviation': cp950_to_utf8,
                         },
                         )

        # output data file path
        fout_path = os.path.join(SYMBOLS_PKL_DIR, '{}_df.pkl'.format(symbol))
        df.to_pickle(fout_path)

        print ("[{}/{}]{}.csv to dataframe OK, {:.3f} secs".format(
            rdx + 1, len(csvs), symbol, time() - t0))

    print ("csv_to_pkl OK, {:.3f} secs".format(time() - t0))


def dataframe_to_panel(symbols=EXP_SYMBOLS):
    """
    aggregating and trimming data to a panel file
    """
    t0 = time()
    start_date = date(2004, 1, 1)
    end_date = END_DATE

    # load first df to read the periods
    fin_path = os.path.join(SYMBOLS_PKL_DIR, "{}_df.pkl".format(symbols[0]))
    df = pd.read_pickle(fin_path)

    # get trans_dates and columns
    trans_dates = df[start_date:end_date].index
    trans_dates.name = 'trans_dates'
    minor_indices = ['close_price', 'simple_roi']

    # setting panel
    pnl = pd.Panel(
        np.zeros((len(trans_dates), len(symbols), len(minor_indices))),
        items=trans_dates,
        major_axis=symbols,
        minor_axis=minor_indices)

    for sdx, symbol in enumerate(symbols):
        t1 = time()
        # read df
        fin_path = os.path.join(SYMBOLS_PKL_DIR, "{}_df.pkl".format(symbol))
        trimmed_df = pd.read_pickle(fin_path).loc[start_date:end_date]

        # rename columns
        trimmed_df['simple_roi_%'] /= 100.
        trimmed_df.rename(columns={r'simple_roi_%': 'simple_roi'},
                          inplace=True)

        # note: pnl.loc[:, symbol, :], shape: (columns, n_exp_period)
        pnl.loc[:, symbol, :] = trimmed_df.ix[:, ('close_price',
                                                  'simple_roi')].T

        print ("[{}/{}] {} load to panel OK, {:.3f} secs".format(
            sdx, len(symbols), symbol, time() - t1))

    #  # fill na with 0
    pnl = pnl.fillna(0)

    # output data file path
    fout_path = os.path.join(SYMBOLS_PKL_DIR,
                             'TAIEX_2005_largest50cap_panel.pkl')
    pnl.to_pickle(fout_path)

    print ("all exp_symbols load to panel OK, {:.3f} secs".format(time()-t0))


def plot_exp_symbol_roi(n_row=5, n_col=5, plot_kind='line'):
    """
    plot the line and distribution charts

    http://matplotlib.org/api/legend_api.html
    legend location:
        'best'         : 0, (only implemented for axes legends)
        'upper right'  : 1,
        'upper left'   : 2,
        'lower left'   : 3,
        'lower right'  : 4,
        'right'        : 5,
        'center left'  : 6,
        'center right' : 7,
        'lower center' : 8,
        'upper center' : 9,
        'center'       : 10,
    """
    fin_path = os.path.join(SYMBOLS_PKL_DIR,
                             'TAIEX_2005_largest50cap_panel.pkl')
    panel = pd.read_pickle(fin_path)

    # shape: (5,5) * 2
    symbols = EXP_SYMBOLS
    if len(symbols) % (n_col * n_row) == 0:
         n_figure = len(symbols) / (n_col * n_row)
    else:
         n_figure = len(symbols) / (n_col * n_row) + 1

    plt.clf()
    for fdx in xrange(n_figure):
        sdx = fdx * n_row * n_col
        edx = (fdx+1) * n_row * n_col
        df = panel.ix[:, symbols[sdx:edx], 'simple_roi'].T

        if plot_kind == "line":
            axes_arr = df.plot(
                kind=plot_kind,
                subplots=True, layout=(n_row, n_col), figsize=(48, 36),
                color='green', legend=False, sharex=True, sharey=True)

        elif plot_kind == "hist":
            axes_arr = df.plot(
                kind=plot_kind, bins=50,
                subplots=True, layout=(n_row, n_col), figsize=(48, 36),
                color='green', legend=False, sharex=False, sharey=False)

        for rdx in xrange(n_row):
            for cdx in xrange(n_col):
                axes_arr[rdx,cdx].legend(loc='upper center',
                                         bbox_to_anchor=(0.5, 1.0))

        img_path = os.path.join(DATA_DIR, 'roi_plot',
                                'roi_{}_{}.pdf'.format(plot_kind, fdx))
        plt.savefig(img_path)

    plt.show()
    plt.close()


def exp_symbols_statistics(fout_path=os.path.join(
                            DATA_DIR, 'exp_symbols_statistics.xlsx')):
    """
    statistics of experiment symbols
    output the results to xlsx
    """
    t0 = time()
    fin_path = os.path.join(SYMBOLS_PKL_DIR,
                             'TAIEX_2005_largest50cap_panel.pkl')
    # shape: (n_exp_period, n_stock, ('simple_roi', 'close_price'))
    panel = pd.read_pickle(fin_path)

    assert panel.major_axis.tolist() == EXP_SYMBOLS
    panel = panel.loc[date(2005,1,3):date(2014,12,31)]


    # the roi in the first experiment date is zero
    panel.loc[date(2005,1,3), :, 'simple_roi'] = 0.

    stat_indices=(
        # basic information
        'start_date', 'end_date',
        'n_exp_period', 'n_period_up', 'n_period_down',
        # roi
        'cum_roi', 'daily_roi', 'daily_mean_roi',
        'daily_std_roi', 'daily_skew_roi', 'daily_kurt_roi',
        # roi/risk indices
        'sharpe', 'sortino_full', 'sortino_full_semi_std',
        'sortino_partial', 'sortino_partial_semi_std',
        'max_drawdown', 'max_abs_drawdown',
        # normal tests
        'JB_pvalue',
        # uniroot tests
        'ADF_c_pvalue', 'ADF_ct_pvalue', 'ADF_ctt_pvalue', 'ADF_nc_pvalue',
        'DFGLS_c_pvalue', 'DFGLS_ct_pvalue',
        'PP_c_pvalue', 'PP_ct_pvalue', 'PP_nc_pvalue',
        'KPSS_c_pvalue', 'KPSS_ct_pvalue',
        # performance
        'SPA_l_pvalue', 'SPA_c_pvalue', 'SPA_u_pvalue'
    )

    stat_df = pd.DataFrame(np.zeros((len(stat_indices), len(EXP_SYMBOLS))),
                           index=stat_indices,
                           columns=EXP_SYMBOLS)

    for rdx, symbol in enumerate(EXP_SYMBOLS[:2]):
        t1 = time()
        rois = panel[:, symbol, 'simple_roi']
        # basic
        stat_df.loc['start_date', symbol] = rois.index[0]
        stat_df.loc['end_date', symbol] = rois.index[-1]
        stat_df.loc['n_exp_period', symbol] = len(rois)
        stat_df.loc['n_period_up', symbol] = (rois > 0).sum()
        stat_df.loc['n_period_down', symbol] = (rois < 0).sum()

        # roi
        stat_df.loc['cum_roi', symbol] = (rois+1).prod() - 1
        stat_df.loc['daily_roi', symbol] = np.power((rois+1).prod(),
                                                    1./len(rois))-1
        stat_df.loc['daily_mean_roi', symbol] = rois.mean()
        stat_df.loc['daily_std_roi', symbol] = rois.std()
        stat_df.loc['daily_skew_roi', symbol] = rois.skew()
        stat_df.loc['daily_kurt_roi', symbol] = rois.kurt() # excess

        # roi/risk indices
        stat_df.loc['sharpe', symbol] = sharpe(rois)
        (stat_df.loc['sortino_full', symbol],
         stat_df.loc['sortino_full_semi_std', symbol]) = sortino_full(rois)

        (stat_df.loc['sortino_partial', symbol],
         stat_df.loc['sortino_partial_semi_std', symbol]) = sortino_partial(rois)
        print maximum_drawdown(rois)
        (stat_df.loc['max_drawdown', symbol],
         stat_df.loc['max_abs_drawdown', symbol]) = maximum_drawdown(rois)

        # normal tests
        stat_df.loc['JB_pvalue', symbol] = jarque_bera(rois)[1]

        # uniroot tests
        stat_df.loc['ADF_c_pvalue', symbol] = \
            adfuller(rois, regression='c')[1]
        stat_df.loc['ADF_ct_pvalue', symbol] = \
            adfuller(rois, regression='ct')[1]
        stat_df.loc['ADF_ctt_pvalue', symbol] = \
            adfuller(rois, regression='ctt')[1]
        stat_df.loc['ADF_nc_pvalue', symbol] = \
            adfuller(rois, regression='nc')[1]

        stat_df.loc['DFGLS_c_pvalue', symbol] = DFGLS(rois, trend='c').pvalue
        stat_df.loc['DFGLS_ct_pvalue', symbol] = DFGLS(rois, trend='ct').pvalue
        stat_df.loc['PP_c_pvalue', symbol] = \
            PhillipsPerron(rois, trend='c').pvalue
        stat_df.loc['PP_ct_pvalue', symbol] = \
            PhillipsPerron(rois, trend='ct').pvalue
        stat_df.loc['PP_nc_pvalue', symbol] = \
            PhillipsPerron(rois, trend='nc').pvalue
        stat_df.loc['KPSS_c_pvalue', symbol] = KPSS(rois, trend='c').pvalue
        stat_df.loc['KPSS_ct_pvalue', symbol] = KPSS(rois, trend='ct').pvalue

        # performance
        spa = SPA(rois, np.zeros(len(rois)), reps=5000)
        spa.seed(np.random.randint(0, 2 ** 31 - 1))
        spa.compute()
        stat_df.loc['SPA_l_pvalue', symbol] = spa.pvalues[0]
        stat_df.loc['SPA_c_pvalue', symbol]= spa.pvalues[1]
        stat_df.loc['SPA_u_pvalue', symbol] = spa.pvalues[2]

        print ("[{}/{}] {} roi statistics OK, {:.3f} secs".format(
            rdx+1, len(EXP_SYMBOLS), symbol, time() - t1
        ))

    # write to excel
    writer = pd.ExcelWriter(fout_path, engine='xlsxwriter')

    stat_df.to_excel(writer, sheet_name='stats')

    # Get the xlsxwriter workbook and worksheet objects.
    workbook  = writer.book
    worksheet = writer.sheets['stats']

    # Add some cell formats.
    percent_fmt = workbook.add_format({'num_format': '0%'})
    date_fmt = workbook.add_format({'num_format': 'yy/mmm/dd'})

    worksheet.set_row('B2:AY3', 20, date_fmt)
    worksheet.set_row('B7:AY13', 20, percent_fmt)

    writer.save()

    print ("all roi statistics OK, {:.3f} secs".format(time() - t0))


if __name__ == '__main__':
    # csv_to_pkl()
    # dataframe_to_panel()
    # plot_exp_symbol_roi(plot_kind='line')
    # plot_exp_symbol_roi(plot_kind='hist')
    exp_symbols_statistics()
