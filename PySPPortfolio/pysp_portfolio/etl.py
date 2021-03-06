# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2

extract-transform-load of data
"""
from datetime import date
from time import time
from glob import glob
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PySPPortfolio.pysp_portfolio import *
from statsmodels.stats.stattools import (jarque_bera, )
from statsmodels.tsa.stattools import (adfuller, )
from utils import (sharpe, sortino_full, sortino_partial, maximum_drawdown)
from arch.bootstrap.multiple_comparrison import (SPA, )
from arch.unitroot.unitroot import (DFGLS, PhillipsPerron, KPSS)
from PySPPortfolio.pysp_portfolio.scenario.c_moment_matching import (
    heuristic_moment_matching as c_HMM,)

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


def verify_symbol_csv():
    """ test if any nan data in csv """
    csvs = glob(os.path.join(SYMBOLS_CSV_DIR, '*.csv'))
    for rdx, csv_path in enumerate(csvs):
        reader = csv.DictReader(open(csv_path), [
            's', 'abbr', 'date', 'o', 'h', 'l', 'c',
            'vol', 'val', 'r', 'to', 'cap', 'cr'])
        reader.next()
        for idx, row in enumerate(reader):
            try:
                roi = float(row['r'])
            except ValueError as e:
                print e
                print csv_path, idx, row['date'], float(row['r'])


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

    # # fill na with 0
    # pnl = pnl.fillna(0)

    # output data file path
    fout_path = os.path.join(SYMBOLS_PKL_DIR,
                             'TAIEX_2005_largest50cap_panel.pkl')
    pnl.to_pickle(fout_path)

    print ("all exp_symbols load to panel OK, {:.3f} secs".format(time() - t0))


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

    assert panel.major_axis.tolist() == EXP_SYMBOLS
    panel = panel.loc[date(2005, 1, 3):date(2014, 12, 31)]

    # the roi in the first experiment date is zero
    panel.loc[date(2005, 1, 3), :, 'simple_roi'] = 0.

    # shape: (5,5) * 2
    symbols = EXP_SYMBOLS
    if len(symbols) % (n_col * n_row) == 0:
        n_figure = len(symbols) / (n_col * n_row)
    else:
        n_figure = len(symbols) / (n_col * n_row) + 1

    plt.clf()
    for fdx in xrange(n_figure):
        sdx = fdx * n_row * n_col
        edx = (fdx + 1) * n_row * n_col
        df = panel.ix[:, symbols[sdx:edx], 'simple_roi'].T

        if plot_kind == "line":
            axes_arr = df.plot(
                kind=plot_kind,
                subplots=True, layout=(n_row, n_col), figsize=(48, 36),
                color='green', legend=False, sharex=False, sharey=False)

        elif plot_kind == "hist":
            axes_arr = df.plot(
                kind=plot_kind, bins=50,
                subplots=True, layout=(n_row, n_col), figsize=(48, 36),
                color='green', legend=False, sharex=False, sharey=False)

        elif plot_kind == "kde":
            axes_arr = df.plot(
                kind=plot_kind,
                subplots=True, layout=(n_row, n_col), figsize=(48, 36),
                color='green', legend=False, sharex=False, sharey=False)

        for rdx in xrange(n_row):
            for cdx in xrange(n_col):
                axes_arr[rdx, cdx].legend(loc='upper center',
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
    panel = panel.loc[date(2005, 1, 3):date(2014, 12, 31)]

    # the roi in the first experiment date is zero
    panel.loc[date(2005, 1, 3), :, 'simple_roi'] = 0.

    stat_indices = (
        # basic information
        'start_date', 'end_date',
        'n_exp_period', 'n_period_up', 'n_period_down',

        # roi
        'cum_roi', 'daily_roi', 'daily_mean_roi',
        'daily_std_roi', 'daily_skew_roi', 'daily_kurt_roi',

        # roi/risk indices
        'sharpe', 'sortino_full', 'sortino_full_semi_std',
        'sortino_partial', 'sortino_partial_semi_std',
        'max_abs_drawdown',

        # normal tests
        'JB', 'JB_pvalue',

        # uni-root tests
        'ADF_c',
        'ADF_c_pvalue',
        'ADF_ct',
        'ADF_ct_pvalue',
        'ADF_ctt',
        'ADF_ctt_pvalue',
        'ADF_nc',
        'ADF_nc_pvalue',
        'DFGLS_c',
        'DFGLS_c_pvalue',
        'DFGLS_ct',
        'DFGLS_ct_pvalue',
        'PP_c',
        'PP_c_pvalue',
        'PP_ct',
        'PP_ct_pvalue',
        'PP_nc',
        'PP_nc_pvalue',
        'KPSS_c',
        'KPSS_c_pvalue',
        'KPSS_ct',
        'KPSS_ct_pvalue',

        # performance
        'SPA_l_pvalue', 'SPA_c_pvalue', 'SPA_u_pvalue'
    )

    stat_df = pd.DataFrame(np.zeros((len(stat_indices), len(EXP_SYMBOLS))),
                           index=stat_indices,
                           columns=EXP_SYMBOLS)

    for rdx, symbol in enumerate(EXP_SYMBOLS):
        t1 = time()
        rois = panel[:, symbol, 'simple_roi']
        # basic
        stat_df.loc['start_date', symbol] = rois.index[0].strftime("%Y/%b/%d")
        stat_df.loc['end_date', symbol] = rois.index[-1].strftime("%Y/%b/%d")
        stat_df.loc['n_exp_period', symbol] = len(rois)
        stat_df.loc['n_period_up', symbol] = (rois > 0).sum()
        stat_df.loc['n_period_down', symbol] = (rois < 0).sum()

        # roi
        stat_df.loc['cum_roi', symbol] = (rois + 1.).prod() - 1
        stat_df.loc['daily_roi', symbol] = np.power((rois + 1.).prod(),
                                                    1. / len(rois)) - 1
        stat_df.loc['daily_mean_roi', symbol] = rois.mean()
        stat_df.loc['daily_std_roi', symbol] = rois.std()
        stat_df.loc['daily_skew_roi', symbol] = rois.skew()
        stat_df.loc['daily_kurt_roi', symbol] = rois.kurt()  # excess

        # roi/risk indices
        stat_df.loc['sharpe', symbol] = sharpe(rois)
        (stat_df.loc['sortino_full', symbol],
         stat_df.loc['sortino_full_semi_std', symbol]) = sortino_full(rois)

        (stat_df.loc['sortino_partial', symbol],
         stat_df.loc['sortino_partial_semi_std', symbol]) = sortino_partial(
            rois)

        stat_df.loc['max_abs_drawdown', symbol] = maximum_drawdown(rois)

        # normal tests
        jb = jarque_bera(rois)
        stat_df.loc['JB', symbol] = jb[0]
        stat_df.loc['JB_pvalue', symbol] = jb[1]

        # uniroot tests
        adf_c = adfuller(rois, regression='c')
        stat_df.loc['ADF_c', symbol] = adf_c[0]
        stat_df.loc['ADF_c_pvalue', symbol] = adf_c[1]

        adf_ct = adfuller(rois, regression='ct')
        stat_df.loc['ADF_ct', symbol] = adf_ct[0]
        stat_df.loc['ADF_ct_pvalue', symbol] = adf_ct[1]

        adf_ctt = adfuller(rois, regression='ctt')
        stat_df.loc['ADF_ctt', symbol] = adf_ctt[0]
        stat_df.loc['ADF_ctt_pvalue', symbol] = adf_ctt[1]

        adf_nc = adfuller(rois, regression='nc')
        stat_df.loc['ADF_nc', symbol] = adf_nc[0]
        stat_df.loc['ADF_nc_pvalue', symbol] = adf_nc[1]

        dfgls_c_instance = DFGLS(rois, trend='c')
        dfgls_c, dfgls_c_pvalue = (dfgls_c_instance.stat,
                                   dfgls_c_instance.pvalue)
        stat_df.loc['DFGLS_c', symbol] = dfgls_c
        stat_df.loc['DFGLS_c_pvalue', symbol] = dfgls_c_pvalue

        dfgls_ct_instance = DFGLS(rois, trend='ct')
        dfgls_ct, dfgls_ct_pvalue = (dfgls_ct_instance.stat,
                                     dfgls_ct_instance.pvalue)
        stat_df.loc['DFGLS_ct', symbol] = dfgls_ct
        stat_df.loc['DFGLS_ct_pvalue', symbol] = dfgls_ct_pvalue

        pp_c_instance = PhillipsPerron(rois, trend='c')
        pp_c, pp_c_pvalue = (pp_c_instance.stat, pp_c_instance.pvalue)
        stat_df.loc['PP_c', symbol] = pp_c
        stat_df.loc['PP_c_pvalue', symbol] = pp_c_pvalue

        pp_ct_instance = PhillipsPerron(rois, trend='ct')
        pp_ct, pp_ct_pvalue = (pp_ct_instance.stat, pp_ct_instance.pvalue)
        stat_df.loc['PP_ct', symbol] = pp_ct
        stat_df.loc['PP_ct_pvalue', symbol] = pp_ct_pvalue

        pp_nc_instance = PhillipsPerron(rois, trend='nc')
        pp_nc, pp_nc_pvalue = (pp_nc_instance.stat, pp_nc_instance.pvalue)
        stat_df.loc['PP_nc', symbol] = pp_nc
        stat_df.loc['PP_nc_pvalue', symbol] = pp_nc_pvalue

        kpss_c_instance = KPSS(rois, trend='c')
        kpss_c, kpss_c_pvalue = (kpss_c_instance.stat, kpss_c_instance.pvalue)
        stat_df.loc['KPSS_c', symbol] = kpss_c
        stat_df.loc['KPSS_c_pvalue', symbol] = kpss_c_pvalue

        kpss_ct_instance = KPSS(rois, trend='ct')
        kpss_ct, kpss_ct_pvalue = (kpss_ct_instance.stat,
                                   kpss_ct_instance.pvalue)
        stat_df.loc['KPSS_ct', symbol] = kpss_ct
        stat_df.loc['KPSS_ct_pvalue', symbol] = kpss_ct_pvalue

        # performance
        spa = SPA(rois, np.zeros(len(rois)), reps=5000)
        spa.seed(np.random.randint(0, 2 ** 31 - 1))
        spa.compute()
        stat_df.loc['SPA_l_pvalue', symbol] = spa.pvalues[0]
        stat_df.loc['SPA_c_pvalue', symbol] = spa.pvalues[1]
        stat_df.loc['SPA_u_pvalue', symbol] = spa.pvalues[2]

        print ("[{}/{}] {} roi statistics OK, {:.3f} secs".format(
            rdx + 1, len(EXP_SYMBOLS), symbol, time() - t1
        ))

    # write to excel
    writer = pd.ExcelWriter(fout_path, engine='xlsxwriter')
    stat_df = stat_df.T
    stat_df.to_excel(writer, sheet_name='stats')

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['stats']

    # basic formats.
    # set header
    header_fmt = workbook.add_format()
    header_fmt.set_text_wrap()
    worksheet.set_row(0, 15, header_fmt)

    # set date
    date_fmt = workbook.add_format({'num_format': 'yy/mmm/dd'})
    date_fmt.set_align('right')
    worksheet.set_column('B:C', 12, date_fmt)

    # set percentage
    percent_fmt = workbook.add_format({'num_format': '0.00%'})

    worksheet.set_column('G:J', 8, percent_fmt)
    worksheet.set_column('M:Q', 8, percent_fmt)

    worksheet.set_column('T:T', 8, percent_fmt)
    worksheet.set_column('V:V', 8, percent_fmt)
    worksheet.set_column('X:X', 8, percent_fmt)
    worksheet.set_column('Z:Z', 8, percent_fmt)
    worksheet.set_column('AB:AB', 8, percent_fmt)
    worksheet.set_column('AD:AD', 8, percent_fmt)
    worksheet.set_column('AF:AF', 8, percent_fmt)
    worksheet.set_column('AH:AH', 8, percent_fmt)
    worksheet.set_column('AJ:AJ', 8, percent_fmt)
    worksheet.set_column('AL:AL', 8, percent_fmt)
    worksheet.set_column('AN:AN', 8, percent_fmt)
    worksheet.set_column('AP:AP', 8, percent_fmt)
    worksheet.set_column('AQ:AS', 8, percent_fmt)

    writer.save()

    print ("all roi statistics OK, {:.3f} secs".format(time() - t0))


def generating_scenarios(n_stock, win_length, n_scenario=200, bias=False,
                         scenario_error_retry=3):
    """
    generating scenarios at once

    Parameters:
    ------------------
    n_stock: integer, number of stocks in the EXP_SYMBOLS
    win_length: integer, number of historical periods
    n_scenario: integer, number of scenarios to generating
    bias: boolean,
        - False: unbiased estimator of moments
        - True: biased estimator of moments
    scenario_error_retry: integer, maximum retry of scenarios
    """
    t0 = time()
    fin_path = os.path.join(SYMBOLS_PKL_DIR,
                            'TAIEX_2005_largest50cap_panel.pkl')

    # shape: (n_period, n_stock, ('simple_roi', 'close_price'))
    panel = pd.read_pickle(fin_path)

    # symbols
    symbols = EXP_SYMBOLS[:n_stock]

    # all trans_date
    trans_dates = panel.items

    # experiment trans_dates
    exp_start_date, exp_end_date = START_DATE, END_DATE
    exp_start_idx = trans_dates.get_loc(exp_start_date)
    exp_end_idx = trans_dates.get_loc(exp_end_date)
    exp_trans_dates = trans_dates[exp_start_idx: exp_end_idx+1]
    n_exp_period = len(exp_trans_dates)

    # estimating moments and correlation matrix
    est_moments = pd.DataFrame(np.zeros((n_stock, 4)), index=symbols)

    parameters = "m{}_w{}_s{}_{}".format(n_stock, win_length, n_scenario,
                                         "biased" if bias else "unbiased")

    # output scenario panel
    scenario_panel = pd.Panel(np.zeros((n_exp_period, n_stock, n_scenario)),
                              items= exp_trans_dates,
                              major_axis=symbols)

    for tdx, exp_date in enumerate(exp_trans_dates):
        t1 = time()

        # rolling historical window indices, containing today
        est_start_idx = exp_start_idx + tdx - win_length + 1
        est_end_idx = exp_start_idx + tdx + 1
        hist_interval = trans_dates[est_start_idx:est_end_idx]

        assert len(hist_interval) == win_length
        assert hist_interval[-1] == exp_date

        # hist_data, shape: (n_stock, win_length)
        hist_data = panel.loc[hist_interval, symbols, 'simple_roi']
        # est moments and corrs
        est_moments.iloc[:, 0] = hist_data.mean(axis=1)
        if bias:
            est_moments.iloc[:, 1] = hist_data.std(axis=1, ddof=0)
            est_moments.iloc[:, 2] = hist_data.skew(axis=1, bias=True)
            est_moments.iloc[:, 3] = hist_data.kurt(axis=1, bias=True)
        else:
            est_moments.iloc[:, 1] = hist_data.std(axis=1, ddof=1)
            est_moments.iloc[:, 2] = hist_data.skew(axis=1, bias=False)
            est_moments.iloc[:, 3] = hist_data.kurt(axis=1, bias=False)
        est_corrs = (hist_data.T).corr("pearson")

        # generating unbiased scenario
        for error_count in xrange(scenario_error_retry):
            try:
                for error_exponent in xrange(-3, 0):
                    try:
                        # default moment and corr errors (1e-3, 1e-3)
                        # df shape: (n_stock, n_scenario)
                        max_moment_err = 10 **(error_exponent)
                        max_corr_err = 10 **(error_exponent)
                        scenario_df = c_HMM(est_moments.as_matrix(),
                                            est_corrs.as_matrix(),
                                            n_scenario, bias,
                                            max_moment_err,
                                            max_corr_err)
                    except ValueError as e:
                        print ("relaxing max err: {}_{}_max_mom_err:{}, "
                               "max_corr_err{}".format( exp_date, parameters,
                                max_moment_err, max_corr_err))
                    else:
                        # generating scenarios success
                        break

            except Exception as e:
                # catch any other exception
                if error_count == scenario_error_retry - 1:
                    raise Exception(e)
            else:
                # generating scenarios success
                break

        # store scenarios
        scenario_panel.loc[exp_date, :, :] = scenario_df

        # clear est data
        print ("[{}/{}][{}_{}] {}: {} scenarios OK, {:.3f} secs".format(
            tdx+1, n_exp_period,
            exp_start_date.strftime("%y%m%d"),
            exp_end_date.strftime("%y%m%d"),
            exp_date.strftime("%Y-%m-%d"),
            parameters, time() - t1))

    # scenario dir
    scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios')
    if not os.path.exists(scenario_path):
        os.makedirs(scenario_path)

    # check file name
    for file_cnt in xrange(1, MAX_SCENARIO_FILE_CNT+1):
        file_name = "{}_{}_{}_{}.pkl".format(
            exp_start_date.strftime('%Y%m%d'),
            exp_end_date.strftime('%Y%m%d'), parameters, file_cnt)
        file_path = os.path.join(scenario_path, file_name)
        if os.path.exists(file_path):
            if file_cnt == MAX_SCENARIO_FILE_CNT:
                raise ValueError('maximum file count limited, {}'.format(
                    file_path))
        else:
            # store file
            scenario_panel.to_pickle(file_path)
            break

    print ("generating scenarios {}-{}, {} OK, {:.3f} secs \n {}".format(
        exp_start_date.strftime('%Y%m%d'),
        exp_end_date.strftime('%Y%m%d'),
        parameters, time() - t0, file_path))


if __name__ == '__main__':
    pass
    # csv_to_pkl()
    # dataframe_to_panel()
    # plot_exp_symbol_roi(plot_kind='line')
    # plot_exp_symbol_roi(plot_kind='hist')
    # plot_exp_symbol_roi(plot_kind='kde')
    # exp_symbols_statistics()
    # verify_symbol_csv()
    generating_scenarios(10, 60)
