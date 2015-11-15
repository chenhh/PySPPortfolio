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

    # output data file path
    fout_path = os.path.join(SYMBOLS_PKL_DIR,
                             'TAIEX_2005_largest50cap_panel.pkl')
    pnl.to_pickle(fout_path)

    print ("all exp_symbols load to panel OK, {:.3f} secs".format(time()-t0))


def plot_exp_symbol_roi():
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
    n_row, n_col = 5, 5
    if len(symbols) % (n_col * n_row) == 0:
         n_figure = len(symbols) / (n_col * n_row)
    else:
         n_figure = len(symbols) / (n_col * n_row) + 1

    plt.clf()
    for fdx in xrange(n_figure):
        sdx = fdx * n_row * n_col
        edx = (fdx+1) * n_row * n_col
        df = panel.ix[:20, symbols[sdx:edx], 'simple_roi'].T
        axes_arr = df.plot(subplots=True, layout=(n_row, n_col),
                           color='green', legend=False, sharex=True,
                           sharey=True)

        for rdx in xrange(n_row):
            for cdx in xrange(n_col):
                axes_arr[rdx,cdx].legend(loc='upper center',
                                         bbox_to_anchor=(0.5, 1.2))

    plt.show()









if __name__ == '__main__':
    # csv_to_pkl()
    # dataframe_to_panel()
    plot_exp_symbol_roi()
