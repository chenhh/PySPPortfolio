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
import csv
from PySPPortfolio.pysp_portfolio import *


def cp950_to_utf8(data):
    ''' utility function in parsing csv '''
    return data.strip().decode('cp950')


def data_strip(data):
    ''' utility function in parsing csv '''
    return data.strip()


def csv_to_pkl(symbols = EXP_SYMBOLS):
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
        fout_path = os.path.join(SYMBOLS_PKL_DIR, '{}.pkl'.format(symbol))
        df.to_pickle(fout_path)

        print ("[{}/{}]{}.csv to dataframe OK, {:.3f} secs".format(
            rdx+1, len(csvs), symbol, time()-t0))

    print ("csv_to_pkl OK, {:.3f} secs".format(time()-t0))

if __name__ == '__main__':
    csv_to_pkl()
