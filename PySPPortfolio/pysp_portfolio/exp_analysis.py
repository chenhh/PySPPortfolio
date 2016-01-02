# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
import os
from time import time
from datetime import date
import pandas as pd
import numpy as np
import scipy.stats as spstats
import statsmodels.stats.stattools as stat_tools
import statsmodels.tsa.stattools as tsa_tools
from PySPPortfolio.pysp_portfolio import *
import utils
from arch.bootstrap.multiple_comparrison import (SPA, )
from gen_results import (all_experiment_parameters,)

def load_rois(symbol=None):
    if symbol is None:
        symbol = "TAIEX_2005_largest50cap_panel"

    return pd.read_pickle(
            os.path.join(SYMBOLS_PKL_DIR, "{}.pkl".format(symbol)))

def load_results(prob_type, n_stock, win_length=0, n_scenario=200,
                     bias=False, scenario_cnt=1, alpha=0.95):
    """
    Parameters:
    -----------------------------------------------------
    prob_type: str, {min_cvar_sp, min_cvar_sip}
    n_stock: integer
    win_length: integer
    n_scenario: integer
    bias: boolean
    scenario_cnt: integer
    alpha: float

    """
    if prob_type in ("min_cvar_sp", "min_ms_cvar_sp", "min_cvar_eev"):
        param = "{}_{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
            n_stock, win_length, n_scenario, "biased" if bias else "unbiased",
            scenario_cnt, alpha)
    elif prob_type == "min_cvar_sip":
        param = "{}_{}_all{}_m{}_w{}_s{}_{}_{}_a{:.2f}".format(
            START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
            len(EXP_SYMBOLS), n_stock, win_length, n_scenario,
            "biased" if bias else "unbiased", scenario_cnt, alpha)
    elif prob_type == "bah":
        param = "{}_{}_m{}".format(
            START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
            n_stock)
    else:
        raise ValueError('unknown prob_type: {}'.format(prob_type))

    # read results
    file_name = '{}_{}.pkl'.format(prob_type, param)
    file_path = os.path.join(EXP_SP_PORTFOLIO_DIR, prob_type, file_name)
    if not os.path.exists(file_path):
        print ("results {} not exists.".format(file_path))
        return None

    # cum_roi to df
    results = pd.read_pickle(file_path)
    return results

def all_results_to_sheet_xlsx(prob_type="min_cvar_sip", sheet="alpha",
                              max_scenario_cnts=MAX_SCENARIO_FILE_CNT):
    """
    n_stock: {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}: length 10
    win_length: {50, 60, ..., 240}, length: 20
    n_scenario:  200
    biased: {unbiased,}
    cnt: {1,2,3}
    alpha: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                                  0.85, 0.9, 0.95): length: 10

    the sheet can be {n_stock, win_length, alpha}

    """
    if not sheet in ("n_stock", "win_length", "alpha"):
        raise ValueError('{} cannot be sheet'.format(sheet))

    n_stocks = range(5, 50 + 5, 5)
    win_lengths = range(50, 240 + 10, 10)
    cnts = range(1, max_scenario_cnts+1)
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')
    columns = ['n_stock', 'win_length', 'alpha', "scenario_cnt",
              'start_date', 'end_date', 'n_exp_period',
              'trans_fee_loss',
              'cum_roi', 'daily_roi', 'daily_mean_roi', 'daily_std_roi',
               'daily_skew_roi',
              'daily_kurt_roi', 'sharpe', 'sortino_full', 'sortino_partial',
              'max_abs_drawdown', 'SPA_l_pvalue', 'SPA_c_pvalue',
              'SPA_u_pvalue', 'simulation_time']

    if prob_type == "min_cvar_sip":
        columns.insert(1, "max_portfolio_size")


    if sheet == "n_stock":
        name = ["w{}_s200_unbiased_{}_a{}".format(win_length, cnt, alpha)
                    for win_length in win_lengths
                    for cnt in cnts
                    for alpha in alphas]

        # panel: shape:(n_stock,  win_length * alpha * cnt, n_result_columns)
        results_panel = pd.Panel(
                            np.zeros((len(n_stocks), len(name), len(columns))),
                            items=n_stocks, major_axis=name,
                            minor_axis=columns)

    elif sheet == "win_length":
        name = ["m{}_s200_unbiased_{}_a{}".format(n_stock, cnt, alpha)
                    for n_stock in n_stocks
                    for cnt in cnts
                    for alpha in alphas]

        # panel: shape:(win_length,  n_stock * alpha * cnt, n_result_columns)
        results_panel = pd.Panel(
                            np.zeros((len(win_lengths), len(name),
                                      len(columns))),
                            items=win_lengths, major_axis=name,
                            minor_axis=columns)

    elif sheet == "alpha":
        name = ["m{}_w{}_s200_unbiased_{}".format(n_stock, win_length, cnt)
                    for cnt in cnts
                    for n_stock in n_stocks
                    for win_length in win_lengths]

        # panel: shape:(alpha, n_stock * win_length * cnt, n_result_columns)
        results_panel = pd.Panel(
                            np.zeros((len(alphas), len(name), len(columns))),
                            items=alphas, major_axis=name, minor_axis=columns)

    params = all_experiment_parameters(prob_type, max_scenario_cnts)
    n_param = len(params)
    for rdx, (m, w, n, b, c, a) in enumerate(params):
        t1 = time()
        bias = True if b == "biased" else False
        results = load_results(prob_type, m, w, n, bias, c, float(a))
        if results:
            if sheet == "n_stock":
                major = "w{}_s200_unbiased_{}_a{}".format(w, c, a)
                item_key = m

            elif sheet == "win_length":
                major = "m{}_s200_unbiased_{}_a{}".format(m, c, a)
                item_key = w

            elif sheet == "alpha":
                major = "m{}_w{}_s200_unbiased_{}".format(m,w,c)
                item_key = a

            for col_key in columns:
                if col_key not in ('win_length', 'scenario_cnt'):
                    results_panel.loc[item_key, major, col_key] = results[
                        col_key]
                else:
                    results_panel.loc[item_key, major, 'win_length'] = w
                    results_panel.loc[item_key, major, 'scenario_cnt'] = c

            print ("[{}/{}] {} {}: {}_{} OK, {:.3f} secs".format(
                rdx +1, n_param, prob_type, sheet , major, a, time()-t1))

    results_panel.to_excel(os.path.join(TMP_DIR,
                                '{}_{}.xlsx'.format(prob_type, sheet)))


def all_results_to_xlsx(prob_type="min_cvar_sp",
                        max_scenario_cnts=MAX_SCENARIO_FILE_CNT):
    """ output results to a single sheet """
    n_stocks = range(5, 50 + 5, 5)
    win_lengths = range(50, 240 + 10, 10)
    cnts = range(1, max_scenario_cnts+1)
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')
    columns = ['n_stock', 'win_length', 'alpha', "scenario_cnt",
              'start_date', 'end_date', 'n_exp_period',
              'trans_fee_loss',
              'cum_roi', 'daily_roi', 'daily_mean_roi', 'daily_std_roi',
               'daily_skew_roi',
              'daily_kurt_roi', 'sharpe', 'sortino_full', 'sortino_partial',
              'max_abs_drawdown', 'SPA_l_pvalue', 'SPA_c_pvalue',
              'SPA_u_pvalue', 'simulation_time']

    if prob_type == "min_cvar_sip":
        columns.append("max_portfolio_size")

    # output all combination to a sheet
    names = ["m{}_w{}_s200_unbiased_{}_a{}".format(
                n_stock, win_length, cnt, alpha)
                for n_stock in n_stocks
                for win_length in win_lengths
                for cnt in cnts
                for alpha in alphas]

    result_df = pd.DataFrame(
                np.zeros((len(names), len(columns))),
                index=names, columns=columns)

    params = all_experiment_parameters(prob_type, max_scenario_cnts)
    n_param = len(params)

    for rdx, (m, w, n, b, c, a) in enumerate(params):
        bias = True if b == "biased" else False
        results = load_results(prob_type, m, w, n, bias, c, float(a))
        if results:
            key = "m{}_w{}_s200_unbiased_{}_a{}".format(m,w,c,a)

            for col_key in columns:
                if col_key not in ('win_length', 'scenario_cnt'):
                    result_df.loc[key, col_key] = results[col_key]
                else:
                    result_df.loc[key, 'win_length'] = w
                    result_df.loc[key, 'scenario_cnt'] = c

            print ("[{}/{}] {} {} OK".format(rdx +1, n_param, prob_type, key))
    print ("{} OK".format(prob_type))
    result_df.to_excel(os.path.join(TMP_DIR,
                                '{}_results_all.xlsx'.format(prob_type)))
    pd.to_pickle(result_df, os.path.join(TMP_DIR,
                                '{}_results_all.pkl'.format(prob_type)))

def all_results_to_4dpanel(prob_type="min_cvar_sp",
                           max_scenario_cnts=MAX_SCENARIO_FILE_CNT):
    """
    axis_0: n_stock
    axis_1: win_length
    axis_2: alpha
    axis_3: cnt + columns
    """
    cnts = range(1, max_scenario_cnts+1)
    n_stocks = ["m{}".format(str(v)) for v in range(5, 50 + 5, 5)]
    win_lengths = ["w{}".format(str(v)) for v in range(50, 240 + 10, 10)]
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')
    columns = ['n_stock', 'win_length', 'alpha', 'scenario_cnt',
              'start_date', 'end_date', 'n_exp_period',
              'trans_fee_loss',
              'cum_roi', 'daily_roi', 'daily_mean_roi', 'daily_std_roi',
              'daily_skew_roi',
              'daily_kurt_roi', 'sharpe', 'sortino_full', 'sortino_partial',
              'max_abs_drawdown', 'SPA_l_pvalue', 'SPA_c_pvalue',
              'SPA_u_pvalue', 'simulation_time']

    params = all_experiment_parameters(prob_type, max_scenario_cnts)
    n_param = len(params)

    # 3 4d-panel
    panels = [pd.Panel4D(np.zeros((len(n_stocks),len(win_lengths),
                                    len(alphas), len(columns))),
                           labels=n_stocks, items=win_lengths,
                           major_axis=alphas, minor_axis=columns)
              for _ in cnts]

    for rdx, (m, w, n, b, c, a) in enumerate(params):
        bias = True if b == "biased" else False
        results = load_results(prob_type, m, w, n, bias, c, float(a))
        stock_key = "m{}".format(m)
        win_key = "w{}".format(w)
        for col_key in columns:
            if col_key in ('win_length', 'scenario_cnt'):
                # additioanl columns
                panels[c-1].loc[stock_key, win_key, a, 'win_length'] = w
                panels[c-1].loc[stock_key, win_key, a, 'scenario_cnt'] = c
            else:
                panels[c-1].loc[stock_key, win_key, a, col_key] = results[
                    col_key]

        print ("[{}/{}] {} OK".format(rdx+1, n_param, results['func_name']))

    for cnt in xrange(max_scenario_cnts):
        file_name = "{}_exp_results_{}.pkl".format(prob_type, cnt+1)
        file_path = os.path.join(TMP_DIR, file_name)
        results = {}
        results['labels'] = panels[cnt].labels
        results['items'] = panels[cnt].items
        results['major_axis'] = panels[cnt].major_axis
        results['minor_axis'] = panels[cnt].minor_axis
        results['data'] = panels[cnt].as_matrix()
        pd.to_pickle(results, file_path)


def bah_results_to_xlsx():
    """ buy and hold """
    n_stocks = range(5, 50+5, 5)
    columns = ['n_stock', 'start_date', 'end_date', 'n_exp_period',
               'trans_fee_loss', 'cum_roi', 'daily_roi', 'daily_mean_roi',
               'daily_std_roi',  'daily_skew_roi', 'daily_kurt_roi',
               'sharpe', 'sortino_full',
               'sortino_partial', 'max_abs_drawdown', 'SPA_l_pvalue',
               'SPA_c_pvalue', 'SPA_u_pvalue', 'simulation_time']

    df = pd.DataFrame(np.zeros((len(n_stocks), len(columns))),
                      index=n_stocks, columns=columns)

    for n_stock in n_stocks:
        results = load_results("bah", n_stock)
        print results['func_name']
        for col in columns:
            df.loc[n_stock, col] = results[col]

    df.to_excel(os.path.join(TMP_DIR, 'BAH.xlsx'))

def significant_star(val):
    if val <= 0.01:
        star = "***"
    elif val <= 0.05:
        star = "**"
    elif val <= 0.1:
        star = "*"
    else:
        star = ""
    return star

def bah_results_to_latex():
    """ buy and hold """
    n_stocks = range(5, 50+5, 5)
    columns = ['n_stock', 'R_cum', 'R_ann', 'mu',
               'sigma', 'skew', 'kurt', 'Sortino',
               'sharpe',  'JB', 'ADF',
               'SPA']
    with open(os.path.join(TMP_DIR, 'bah_stat_txt.txt'), 'wb') as texfile:
        texfile.write("{} \\ \hline \n".format(" & ".join(columns)))

        for n_stock in n_stocks:
            results = load_results("bah", n_stock)
            wealth_arr = (results['wealth_df'].sum(axis=1) +
                         results['risk_free_wealth'])
            rois = wealth_arr.pct_change()
            rois[0] = 0
            # print rois

            R_c = (1+rois).prod() -1
            R_a = np.power(R_c+1, 1./10) -1
            sharpe = utils.sharpe(rois)
            sortino = utils.sortino_full(rois)[0]
            jb = stat_tools.jarque_bera(rois)[1]
            jb_str = "{:>4.2f}".format(jb*100)

            adf_c = tsa_tools.adfuller(rois, regression='c')[1]
            adf_ct = tsa_tools.adfuller(rois, regression='ct')[1]
            adf_ctt = tsa_tools.adfuller(rois, regression='ctt')[1]
            adf_nc = tsa_tools.adfuller(rois, regression='nc')[1]
            adf = max(adf_c, adf_ct, adf_ctt, adf_nc)
            spa_value = 0
            for _ in xrange(10):
                spa = SPA(rois, np.zeros(wealth_arr.size), reps=5000)
                spa.seed(np.random.randint(0, 2 ** 31 - 1))
                spa.compute()
                if spa.pvalues[1] > spa_value :
                    spa_value = spa.pvalues[1]

            row = ["{:>3}".format(n_stock),
                   "{:>6.2f}".format(R_c*100),
                   "{:>4.2f}".format(R_a*100),
                   "{:>6.4f}".format(rois.mean()*100),
                   "{:>6.4f}".format(rois.std(ddof=1)*100),
                   "{:>5.2f}".format(spstats.skew(rois, bias=False)),
                   "{:>4.2f}".format(spstats.kurtosis(rois, bias=False)),
                   "{:>4.2f}".format(sharpe*100),
                   "{:>4.2f}".format(sortino*100),
                   "{:<3} {:>4.2f}".format(significant_star(jb), jb*100),
                   "{:<3} {:>4.2f}".format(significant_star(adf), adf*100),
                   "{:<3} {:>4.2f}".format(significant_star(spa_value),
                                           spa_value * 100),
                   ]
            texfile.write("{} \\\\ \hline \n".format(" & ".join(row)))
            print "BAH: {} OK".format(n_stock)

def plot_results(prob_type="min_cvar_sp", scenario_cnt=1):
    """
    :param prob_type:
    :return:
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if not prob_type in ('min_cvar_sp', 'min_cvar_sip'):
        raise ValueError('unknown problem type:{}'.format(prob_type))

    file_path = os.path.join(EXP_SP_PORTFOLIO_DIR, "reports",
                             "{}_exp_results_{}.pkl".format(prob_type,
                                                            scenario_cnt))
    results = pd.read_pickle(file_path)
    panel =  pd.Panel4D(results['data'], labels=results['labels'],
                        items=results['items'],
                        major_axis=results['major_axis'],
                        minor_axis=results['minor_axis'])
    # n_stock, win_length, alpha, columns
    stock ="m45"
    win = 'w230'
    alpha="0.90"
    roi_df = panel.loc["m5", : , :, 'cum_roi']
    print roi_df.columns, roi_df.index
    ax = roi_df.plot( kind='bar', title="{}-s{}".format(stock, scenario_cnt),
                      legend=True, ylim=(0.8, 2.8), yerr=np.random.randn(10))

    ax.legend(loc=1, bbox_to_anchor=(1.05, 1.0))

    plt.show()


def all_results_roi_stats(prob_type="min_cvar_sp"):
    """
    cnts = range(1, 3+1)
    n_stocks = ["m{}".format(str(v)) for v in range(5, 50 + 5, 5)]
    win_lengths = ["w{}".format(str(v)) for v in range(50, 240 + 10, 10)]
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')
    columns = ['n_stock', 'win_length', 'alpha', 'scenario_cnt',
              'start_date', 'end_date', 'n_exp_period',
              'trans_fee_loss',
              'cum_roi', 'daily_roi', 'daily_mean_roi', 'daily_std_roi',
              'daily_kurt_roi', 'sharpe', 'sortino_full', 'sortino_partial',
              'max_abs_drawdown', 'SPA_l_pvalue', 'SPA_c_pvalue',
              'SPA_u_pvalue', 'simulation_time']
    """
    fin = os.path.join(EXP_SP_PORTFOLIO_REPORT_DIR,
                       '{}_results_all.pkl'.format(prob_type))
    df = pd.read_pickle(fin)
    grouped = df.groupby(['n_stock', 'win_length', 'alpha'])
    stats_df = pd.DataFrame([grouped.mean()['cum_roi'],
                             grouped.std()['cum_roi']],
                            index=("cum_roi_mean", "cum_roi_std"))
    stats_df = stats_df.T
    stats_df.reset_index(inplace=True)
    print stats_df
    stats_df.to_excel(os.path.join(
        TMP_DIR, '{}_results_roi_stats.xlsx'.format(prob_type)))


def plot_3d_results(prob_type="min_cvar_sp", z_dim='cum_roi'):
    """
    axis-0: n_stock
    axis-1: win_length
    axis-2: cum_roi (annualized roi)
    for each alpha

    df columns:
    Index([u'n_stock', u'win_length', u'alpha', u'scenario_cnt', u'start_date',
       u'end_date', u'n_exp_period', u'trans_fee_loss', u'cum_roi',
       u'daily_roi', u'daily_mean_roi', u'daily_std_roi', u'daily_kurt_roi',
       u'sharpe', u'sortino_full', u'sortino_partial', u'max_abs_drawdown',
       u'SPA_l_pvalue', u'SPA_c_pvalue', u'SPA_u_pvalue', u'simulation_time'],
      dtype='object')

    colormap:
    http://matplotlib.org/examples/color/colormaps_reference.html

    z_dim: {"cum_roi", "ann_roi", "std_roi"}

    """
    if not prob_type in ("min_cvar_sp", "min_cvar_sip", "min_cvar_eev"):
        raise ValueError("unknown problem type: {}".format(prob_type))

    data_path = os.path.join(EXP_SP_PORTFOLIO_REPORT_DIR,
                       '{}_results_all.pkl'.format(prob_type))
    df = pd.read_pickle(data_path)

    # set alpha column to str
    for rdx in xrange(df.index.size):
        df.ix[rdx, 'alpha'] = "{:.2f}".format(df.ix[rdx, 'alpha'])

    #axes
    stocks = np.arange(5, 50+5, 5)
    if prob_type == "min_cvar_sip":
        lengths = np.arange(60, 240 + 10, 10)
    else:
        lengths = np.arange(60, 240 + 10, 10)
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    # figure size in inches
    fig = plt.figure( figsize=(64,48), facecolor='white')

    # normalized color bar
    if z_dim == "cum_roi":
        LOWER_BOUND, UPPER_BOUND, STEP = -100, 280, 50
        COLOR_STEP = 20
    if z_dim == "cum_roi_diff":
        LOWER_BOUND, UPPER_BOUND, STEP = -200, 400, 50
        COLOR_STEP = 20
    elif z_dim == "ann_roi":
        LOWER_BOUND, UPPER_BOUND, STEP = -10, 20, 4
        COLOR_STEP = 2
    elif z_dim in ("sharpe",):
        LOWER_BOUND, UPPER_BOUND, STEP = -5, 8, 2
        COLOR_STEP = 2
    elif z_dim == "sortino_full":
        LOWER_BOUND, UPPER_BOUND, STEP = -5, 10, 2
        COLOR_STEP = 2
    elif z_dim == "SPA_c_pvalue":
        LOWER_BOUND, UPPER_BOUND, STEP = 0, 12, 2
        COLOR_STEP = 2

    cm_norm = mpl.colors.Normalize(vmin=LOWER_BOUND, vmax=UPPER_BOUND,
                                   clip=False)

    for adx, alpha in enumerate(alphas):
        ax = fig.add_subplot(2, 5, adx+1, projection='3d',
                             xlim=(50, 5), ylim=(40, 240))
        ax.set_zlim(LOWER_BOUND, UPPER_BOUND)

        if z_dim == "cum_roi":
            ax.set_zlabel(r'Average cumulative returns (%)', fontsize=22,
                      fontname="Times New Roman", linespacing=4.5)

        elif z_dim == "ann_roi":
            ax.set_zlabel(r'Average annaualized returns (%)', fontsize=22,
                      fontname="Times New Roman", linespacing=4.5)
        elif z_dim == "sharpe":
            ax.set_zlabel(r'Average Sharpe ratio (%)', fontsize=22,
                      fontname="Times New Roman", linespacing=4.5)
        elif z_dim == "sortino_full":
            ax.set_zlabel(r'Average Sortino ratio (%)', fontsize=22,
                      fontname="Times New Roman", linespacing=4.5)
        elif z_dim == "SPA_c_pvalue":
            ax.set_zlabel(r'P values (%)', fontsize=22,
                      fontname="Times New Roman", linespacing=4.5)

        ax.set_title(r'$\alpha = {}\%$'.format(int(float(alpha)*100.)),
                     y=1.02, fontsize=30)
        ax.set_xlabel(r'$M$', fontsize=24)
        ax.set_ylabel(r'$h$', fontsize=24)
        ax.tick_params(labelsize=10, pad=0, )
        ax.set_xticklabels(np.arange(5, 50+5, 5), fontsize=12,
                           fontname="Times New Roman")
        ax.set_yticklabels(np.arange(50, 240 + 10, 50), fontsize=12,
                           # rotation=-30,
                           fontname="Times New Roman")
        ax.set_zticklabels(np.arange(LOWER_BOUND, UPPER_BOUND, STEP),
                           rotation=90, fontsize=12,
                           fontname="Times New Roman")

        Xs, Ys = np.meshgrid(stocks, lengths)
        Zs = np.zeros_like(Xs, dtype=np.float)

        n_row, n_col = Xs.shape
        for rdx in xrange(n_row):
            for cdx in xrange(n_col):
                n_stock, win_length = Xs[rdx, cdx], Ys[rdx, cdx]
                if prob_type in ("min_cvar_sp", "min_cvar_eev"):
                    if z_dim in ('cum_roi', 'ann_roi'):
                        cum_rois =  df.loc[(df.loc[:,'n_stock']==n_stock) &
                                      (df.loc[:, 'win_length'] == win_length) &
                                      (df.loc[:, 'alpha'] == alpha),
                                      'cum_roi']
                    elif z_dim in ("sharpe", "sortino_full", "SPA_c_pvalue"):
                        values=  df.loc[(df.loc[:,'n_stock']==n_stock) &
                                      (df.loc[:, 'win_length'] == win_length) &
                                      (df.loc[:, 'alpha'] == alpha),
                                      z_dim]

                elif prob_type == "min_cvar_sip":
                    if z_dim in ('cum_roi', 'ann_roi'):
                        cum_rois =  df.loc[
                                (df.loc[:, 'max_portfolio_size'] == n_stock) &
                                (df.loc[:, 'win_length'] == win_length) &
                                (df.loc[:, 'alpha'] == alpha),
                                'cum_roi']
                    elif z_dim in ("sharpe", "sortino_full", "SPA_c_pvalue"):
                        values =  df.loc[
                                (df.loc[:,'max_portfolio_size']==n_stock) &
                                (df.loc[:, 'win_length'] == win_length) &
                                (df.loc[:, 'alpha'] == alpha),
                                z_dim]


                if z_dim == "cum_roi":
                    mean = cum_rois.mean()
                elif z_dim == "ann_roi":
                    # 2005~2014
                    mean = (np.power(cum_rois+1, 1./10) - 1).mean()
                elif  z_dim in ("sharpe", "sortino_full", "SPA_c_pvalue"):
                    mean = values.mean()

                Zs[rdx, cdx] = 0 if np.isnan(mean) else mean * 100

        if prob_type == "min_cvar_sp":
            # n_stock = 50, window = 50
            Zs[-1, 0] = np.nan
        if z_dim == "SPA_c_pvalue":
            Zs[Zs > UPPER_BOUND] = UPPER_BOUND

        print alpha, Zs, UPPER_BOUND

        # contour, projected on z
        # cset = ax.contour(Xs, Ys, Zs, zdir='z', offset=LOWER_BOUND,
        #                    alpha=1, cmap=plt.cm.coolwarm, norm=cm_norm,
        #                    zorder=1)
        # cset = ax.contourf(Xs, Ys, Zs, zdir='z', offset=LOWER_BOUND,
        #                    alpha=1, cmap=plt.cm.coolwarm, norm=cm_norm,
        #                    zorder=1)

        # surface
        p = ax.plot_surface(Xs, Ys, Zs, rstride=1, cstride=1, alpha=1,
                        cmap=plt.cm.coolwarm, norm=cm_norm, zorder=0,
                        antialiased=True)

    # share color bar
    cbar_ax = fig.add_axes([0.96, 0.125, 0.01, 0.75])
    fig.colorbar(p, ax=fig.get_axes(), cax=cbar_ax,
                 ticks=np.arange(LOWER_BOUND, UPPER_BOUND+COLOR_STEP/2,
                                 COLOR_STEP))

    fig.subplots_adjust(left=0.01, bottom=0.02, right=0.95, top=0.98,
                        wspace=0.01, hspace=0.01)
    # plt.tight_layout()
    # plt.savefig(os.path.join(TMP_DIR, '{}_{}.pdf'.format(prob_type, z_dim)),
    # format="pdf", dpi=600)
    plt.show()

def plot_2d_contour(prob_type="min_cvar_sp", z_dim="cum_roi"):
    if not prob_type in ("min_cvar_sp", "min_cvar_sip", "min_cvar_eev"):
        raise ValueError("unknown problem type: {}".format(prob_type))

    data_path = os.path.join(EXP_SP_PORTFOLIO_REPORT_DIR,
                       '{}_results_all.pkl'.format(prob_type))
    df = pd.read_pickle(data_path)

    # set alpha column to str
    for rdx in xrange(df.index.size):
        df.ix[rdx, 'alpha'] = "{:.2f}".format(df.ix[rdx, 'alpha'])

    #axes
    stocks = np.arange(5, 50+5, 5)
    lengths = np.arange(50, 240 + 10, 10)
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # figure size in inches
    fig = plt.figure( figsize=(64,48), facecolor='white')

    cm_norm = mpl.colors.Normalize(vmin=-100, vmax=300,
                                   clip=False)
    for adx, alpha in enumerate(alphas):
        ax = fig.add_subplot(2, 5, adx+1,
                             xlim=(50, 5), ylim=(50, 240))

        ax.set_title(r'$\alpha = {}\%$'.format(int(float(alpha)*100.)),
                     y=1.02, fontsize=24)
        ax.set_xlabel(r'$M$', fontsize=20)
        ax.set_ylabel(r'$h$', fontsize=20)
        ax.tick_params(labelsize=10, pad=2, )
        ax.set_xticklabels(np.arange(5, 50+5, 5), fontsize=12,
                           fontname="Times New Roman")
        ax.set_yticks(lengths)
        ax.set_yticklabels(lengths, fontsize=12,
                           rotation=0,
                           fontname="Times New Roman")
        # ax.set_zticklabels(np.arange(LOWER_BOUND, UPPER_BOUND, STEP),
        #                    rotation=90, fontsize=12,
        #                    fontname="Times New Roman")

        Xs, Ys = np.meshgrid(stocks, lengths)
        Zs = np.zeros_like(Xs, dtype=np.float)

        n_row, n_col = Xs.shape
        for rdx in xrange(n_row):
            for cdx in xrange(n_col):
                n_stock, win_length = Xs[rdx, cdx], Ys[rdx, cdx]
                if prob_type in ("min_cvar_sp", "min_cvar_eev"):
                    if z_dim in ('cum_roi', 'ann_roi'):
                        cum_rois =  df.loc[(df.loc[:,'n_stock']==n_stock) &
                                      (df.loc[:, 'win_length'] == win_length) &
                                      (df.loc[:, 'alpha'] == alpha),
                                      'cum_roi']
                    elif z_dim in ("sharpe", "sortino_full", "SPA_c_pvalue"):
                        values=  df.loc[(df.loc[:,'n_stock']==n_stock) &
                                      (df.loc[:, 'win_length'] == win_length) &
                                      (df.loc[:, 'alpha'] == alpha),
                                      z_dim]

                elif prob_type == "min_cvar_sip":
                    if z_dim in ('cum_roi', 'ann_roi'):
                        cum_rois =  df.loc[
                                (df.loc[:, 'max_portfolio_size'] == n_stock) &
                                (df.loc[:, 'win_length'] == win_length) &
                                (df.loc[:, 'alpha'] == alpha),
                                'cum_roi']
                    elif z_dim in ("sharpe", "sortino_full", "SPA_c_pvalue"):
                        values =  df.loc[
                                (df.loc[:,'max_portfolio_size']==n_stock) &
                                (df.loc[:, 'win_length'] == win_length) &
                                (df.loc[:, 'alpha'] == alpha),
                                z_dim]


                if z_dim == "cum_roi":
                    mean = cum_rois.mean()
                elif z_dim == "ann_roi":
                    # 2005~2014
                    mean = (np.power(cum_rois+1, 1./10) - 1).mean()
                elif  z_dim in ("sharpe", "sortino_full", "SPA_c_pvalue"):
                    mean = values.mean()

                Zs[rdx, cdx] = 0 if np.isnan(mean) else mean * 100

        if prob_type == "min_cvar_sp":
            # n_stock = 50, window = 50
            Zs[-1, 0] = 0


        print alpha, Zs

        # contour, projected on z
        cset = ax.contourf(Xs, Ys, Zs, cmap=plt.cm.coolwarm, norm=cm_norm)
        # surface
        # p = ax.plot_surface(Xs, Ys, Zs, rstride=1, cstride=1, alpha=1,
        #                 cmap=plt.cm.coolwarm, norm=cm_norm, zorder=0,
        #                 antialiased=True)

    # share color bar
    cbar_ax = fig.add_axes([0.96, 0.125, 0.01, 0.75])
    # color_range = np.arange(-100, 320, 40)
    cbar = fig.colorbar(cset, ax=fig.get_axes(), cax=cbar_ax,
                        )
    # cbar.set_ticks([-100, 0, 300])
    # cbar.set_ticklabels(color_range)
    # cbar.update_ticks()

    # fig.subplots_adjust(left=0.02, bottom=0.02, right=0.95, top=0.98,
    #                     wspace=0.1, hspace=0.1)
    # plt.tight_layout()
    # plt.savefig(os.path.join(TMP_DIR, '{}_{}.pdf'.format(prob_type, z_dim)),
    # format="pdf", dpi=600)
    plt.show()



def plot_3d_VSS(prob_type="min_cvar_sp"):
    data_path = os.path.join(EXP_SP_PORTFOLIO_REPORT_DIR,
                       '{}_results_all.pkl'.format(prob_type))
    df = pd.read_pickle(data_path)

    data_path2 = os.path.join(EXP_SP_PORTFOLIO_REPORT_DIR,
                       'min_cvar_eev_results_all.pkl')
    df2 = pd.read_pickle(data_path2)

    # set alpha column to str
    for rdx in xrange(df.index.size):
        df.ix[rdx, 'alpha'] = "{:.2f}".format(df.ix[rdx, 'alpha'])
        df2.ix[rdx, 'alpha'] = "{:.2f}".format(df2.ix[rdx, 'alpha'])

    #axes
    stocks = np.arange(5, 50+5, 5)
    lengths = np.arange(60, 240 + 10, 10)
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    # figure size in inches
    fig = plt.figure( figsize=(64,48), facecolor='white')

    # normalized color bar
    LOWER_BOUND, UPPER_BOUND, STEP = -100, 280, 50
    COLOR_STEP = 20
    cm_norm = mpl.colors.Normalize(vmin=LOWER_BOUND, vmax=UPPER_BOUND,
                                   clip=False)

    for adx, alpha in enumerate(alphas):
        ax = fig.add_subplot(2, 5, adx+1, projection='3d',
                             xlim=(50, 5), ylim=(40, 240))
        ax.set_zlim(LOWER_BOUND, UPPER_BOUND)
        ax.set_zlabel(r'Average VSS', fontsize=22,
                      fontname="Times New Roman", linespacing=4.5)

        ax.set_title(r'$\alpha = {}\%$'.format(int(float(alpha)*100.)),
                     y=1.02, fontsize=30)
        ax.set_xlabel(r'$M$', fontsize=24)
        ax.set_ylabel(r'$h$', fontsize=24)
        ax.tick_params(labelsize=10, pad=0, )
        ax.set_xticklabels(np.arange(5, 50+5, 5), fontsize=12,
                           fontname="Times New Roman")
        ax.set_yticklabels(np.arange(50, 240 + 10, 50), fontsize=12,
                           # rotation=-30,
                           fontname="Times New Roman")
        ax.set_zticklabels(np.arange(LOWER_BOUND, UPPER_BOUND, STEP),
                           rotation=90, fontsize=12,
                           fontname="Times New Roman")

        Xs, Ys = np.meshgrid(stocks, lengths)
        Zs = np.zeros_like(Xs, dtype=np.float)

        n_row, n_col = Xs.shape
        for rdx in xrange(n_row):
            for cdx in xrange(n_col):
                n_stock, win_length = Xs[rdx, cdx], Ys[rdx, cdx]
                if prob_type in ("min_cvar_sp", ):
                    cum_rois =  df.loc[(df.loc[:,'n_stock']==n_stock) &
                                  (df.loc[:, 'win_length'] == win_length) &
                                  (df.loc[:, 'alpha'] == alpha),
                                  'cum_roi']

                elif prob_type == "min_cvar_sip":
                    cum_rois =  df.loc[
                            (df.loc[:, 'max_portfolio_size'] == n_stock) &
                            (df.loc[:, 'win_length'] == win_length) &
                            (df.loc[:, 'alpha'] == alpha),
                                'cum_roi']

                mean = cum_rois.mean()

                Zs[rdx, cdx] = 0 if np.isnan(mean) else mean * 100

        if prob_type == "min_cvar_sp":
            # n_stock = 50, window = 50
            Zs[-1, 0] = np.nan

        print alpha, Zs, UPPER_BOUND

        # surface
        p = ax.plot_surface(Xs, Ys, Zs, rstride=1, cstride=1, alpha=1,
                        cmap=plt.cm.coolwarm, norm=cm_norm, zorder=0,
                        antialiased=True)

    # share color bar
    cbar_ax = fig.add_axes([0.96, 0.125, 0.01, 0.75])
    fig.colorbar(p, ax=fig.get_axes(), cax=cbar_ax,
                 ticks=np.arange(LOWER_BOUND, UPPER_BOUND+COLOR_STEP/2,
                                 COLOR_STEP))

    fig.subplots_adjust(left=0.01, bottom=0.02, right=0.95, top=0.98,
                        wspace=0.01, hspace=0.01)
    # plt.tight_layout()
    # plt.savefig(os.path.join(TMP_DIR, '{}_{}.pdf'.format(prob_type, z_dim)),
    # format="pdf", dpi=600)
    plt.show()




def plot_3d_eev(z_dim="cum_roi"):
    data_path = os.path.join(EXP_SP_PORTFOLIO_REPORT_DIR,
                       '{}_results_all.pkl'.format("min_cvar_eev"))
    df = pd.read_pickle(data_path)
    alpha = "0.50"
    # set alpha column to str
    for rdx in xrange(df.index.size):
        df.ix[rdx, 'alpha'] = "{:.2f}".format(df.ix[rdx, 'alpha'])

      #axes
    stocks = np.arange(5, 50+5, 5)
    lengths = np.arange(60, 240 + 10, 10)
    # alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
    #                '0.85', '0.90', '0.95')

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    # figure size in inches
    fig = plt.figure( figsize=(64,48), facecolor='white')

    # normalized color bar
    if z_dim == "cum_roi":
        LOWER_BOUND, UPPER_BOUND, STEP = -100, 600, 100
        COLOR_STEP = 50
    elif z_dim == "ann_roi":
        LOWER_BOUND, UPPER_BOUND, STEP = -10, 20, 4
        COLOR_STEP = 2
    elif z_dim in ("sharpe",):
        LOWER_BOUND, UPPER_BOUND, STEP = -5, 8, 2
        COLOR_STEP = 2
    elif z_dim == "sortino_full":
        LOWER_BOUND, UPPER_BOUND, STEP = -5, 10, 2
        COLOR_STEP = 2
    elif z_dim == "SPA_c_pvalue":
        LOWER_BOUND, UPPER_BOUND, STEP = 0, 100, 10
        COLOR_STEP = 10

    cm_norm = mpl.colors.Normalize(vmin=LOWER_BOUND, vmax=UPPER_BOUND,
                                   clip=False)

    ax = fig.add_subplot(1,1,1,projection='3d', xlim=(50, 5), ylim=(40, 240))
    ax.set_zlim(LOWER_BOUND, UPPER_BOUND)

    if z_dim == "cum_roi":
        ax.set_zlabel(r'Average cumulative returns (%)', fontsize=22,
                  fontname="Times New Roman", linespacing=4.5)

    elif z_dim == "ann_roi":
        ax.set_zlabel(r'Average annaualized returns (%)', fontsize=22,
                  fontname="Times New Roman", linespacing=4.5)
    elif z_dim == "sharpe":
        ax.set_zlabel(r'Average Sharpe ratio (%)', fontsize=22,
                  fontname="Times New Roman", linespacing=4.5)
    elif z_dim == "sortino_full":
        ax.set_zlabel(r'Average Sortino ratio (%)', fontsize=22,
                  fontname="Times New Roman", linespacing=4.5)
    elif z_dim == "SPA_c_pvalue":
        ax.set_zlabel(r'Average P values of SPA test (%)', fontsize=22,
                  fontname="Times New Roman", linespacing=4.5)

    # ax.set_title(r'$\alpha = {}\%$'.format(int(float(alpha)*100.)),
    #              y=1.02, fontsize=30)
    ax.set_xlabel(r'$M$', fontsize=24)
    ax.set_ylabel(r'$h$', fontsize=24)
    ax.tick_params(labelsize=10, pad=0, )
    ax.set_xticklabels(np.arange(5, 50+5, 5), fontsize=12,
                       fontname="Times New Roman")
    ax.set_yticklabels(np.arange(50, 240 + 10, 50), fontsize=12,
                       # rotation=-30,
                       fontname="Times New Roman")
    ax.set_zticklabels(np.arange(LOWER_BOUND, UPPER_BOUND, STEP),
                       rotation=90, fontsize=12,
                       fontname="Times New Roman")

    Xs, Ys = np.meshgrid(stocks, lengths)
    Zs = np.zeros_like(Xs, dtype=np.float)

    n_row, n_col = Xs.shape
    for rdx in xrange(n_row):
        for cdx in xrange(n_col):
            n_stock, win_length = Xs[rdx, cdx], Ys[rdx, cdx]
            if z_dim in ('cum_roi', 'ann_roi'):
                cum_rois =  df.loc[(df.loc[:,'n_stock']==n_stock) &
                              (df.loc[:, 'win_length'] == win_length) &
                              (df.loc[:, 'alpha'] == alpha),
                              'cum_roi']
            elif z_dim in ("sharpe", "sortino_full", "SPA_c_pvalue"):
                values=  df.loc[(df.loc[:,'n_stock']==n_stock) &
                              (df.loc[:, 'win_length'] == win_length) &
                              (df.loc[:, 'alpha'] == alpha),
                              z_dim]

            if z_dim == "cum_roi":
                mean = cum_rois.mean()
            elif z_dim == "ann_roi":
                # 2005~2014
                mean = (np.power(cum_rois+1, 1./10) - 1).mean()
            elif  z_dim in ("sharpe", "sortino_full", "SPA_c_pvalue"):
                mean = values.mean()
            print rdx, cdx, mean
            Zs[rdx, cdx] = 0 if np.isnan(mean) else mean * 100


    print alpha, Zs

    # contour, projected on z
    # cset = ax.contour(Xs, Ys, Zs, zdir='z', offset=LOWER_BOUND,
    #                    alpha=1, cmap=plt.cm.coolwarm, norm=cm_norm,
    #                    zorder=1)
    # cset = ax.contourf(Xs, Ys, Zs, zdir='z', offset=LOWER_BOUND,
    #                    alpha=1, cmap=plt.cm.coolwarm, norm=cm_norm,
    #                    zorder=1)

    # surface
    p = ax.plot_surface(Xs, Ys, Zs, rstride=1, cstride=1, alpha=1,
                    cmap=plt.cm.coolwarm, norm=cm_norm, zorder=0,
                    antialiased=True)

    # share color bar
    cbar_ax = fig.add_axes([0.96, 0.125, 0.01, 0.75])
    fig.colorbar(p, ax=fig.get_axes(), cax=cbar_ax,
                 ticks=np.arange(LOWER_BOUND, UPPER_BOUND+COLOR_STEP/2,
                                 COLOR_STEP))

    fig.subplots_adjust(left=0.01, bottom=0.02, right=0.95, top=0.98,
                        wspace=0.01, hspace=0.01)
    # plt.tight_layout()
    plt.savefig(os.path.join(TMP_DIR, 'cumulative_roi.pdf'),
    format="pdf", dpi=600)
    plt.show()

def stock_statistics(latex=True):
    import csv
    symbols = EXP_SYMBOLS
    panel = load_rois()

    with open(os.path.join(TMP_DIR, 'stat.csv'), 'wb') as csvfile, \
         open(os.path.join(TMP_DIR, 'stat_txt.txt'), 'wb') as texfile:
        fieldnames = ["rank",'name', "R_c", "R_a", "mu", "std", "skew", "kurt",
                      "sharpe", "sortino", "jb", "adf_c", "adf_ct",
                      "adf_ctt", "adf_nc", "spa"]
        texnames = ["rank",'name', "R_c(%)", "R_a(%)", "mu(%)", "std(%)",
                    "skew", "kurt", "sharpe(%)", "sortino(%)", "jb",
                    "adf", "SPA"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        texfile.write("{} \\ \hline \n".format(" & ".join(texnames)))

        for sdx, symbol in enumerate(symbols):
            rois = panel[START_DATE:END_DATE, symbol, 'simple_roi']
            rois[0] = 0
            R_c = (1+rois).prod() -1
            R_a = np.power(R_c+1, 1./10) -1
            sharpe = utils.sharpe(rois)
            sortino = utils.sortino_full(rois)[0]
            jb = stat_tools.jarque_bera(rois)[1]
            adf_c = tsa_tools.adfuller(rois, regression='c')[1]
            adf_ct = tsa_tools.adfuller(rois, regression='ct')[1]
            adf_ctt = tsa_tools.adfuller(rois, regression='ctt')[1]
            adf_nc = tsa_tools.adfuller(rois, regression='nc')[1]
            adf = max(adf_c, adf_ct, adf_ctt, adf_nc)

            spa_value = 0
            for _ in xrange(10):
                spa = SPA(rois, np.zeros(rois.size), reps=5000)
                spa.seed(np.random.randint(0, 2 ** 31 - 1))
                spa.compute()
                if spa.pvalues[1] > spa_value :
                    spa_value = spa.pvalues[1]

            writer.writerow({
                "rank": sdx+1,
                "name": symbol,
                "R_c": R_c,
                "R_a": R_a,
                "mu": rois.mean(),
                "std": rois.std(ddof=1),
                "skew": spstats.skew(rois, bias=False),
                "kurt": spstats.kurtosis(rois, bias=False),
                "sharpe": sharpe,
                "sortino": sortino,
                "jb": jb,
                "adf_c": adf_c,
                "adf_ct": adf_ct,
                "adf_ctt": adf_ctt,
                "adf_nc": adf_nc,
                "spa": spa_value,
            })
            row = ["{:>3}".format(sdx+1), symbol,
                   "{:>6.2f}".format(R_c*100),
                   "{:>6.2f}".format(R_a*100),
                   "{:>7.4f}".format(rois.mean()*100),
                   "{:>7.4f}".format(rois.std(ddof=1)*100),
                   "{:>5.2f}".format(spstats.skew(rois, bias=False)),
                   "{:>4.2f}".format(spstats.kurtosis(rois, bias=False)),
                   "{:>5.2f}".format(sharpe*100),
                   "{:>5.2f}".format(sortino*100),
                   "{:<3} {:>4.2f}".format(significant_star(jb), jb*100 ),
                   "{:<3} {:>4.2f}".format(significant_star(adf), adf*100),
                   "{:<3} {:>4.2f}".format(significant_star(spa_value),
                                           spa_value * 100),
                   ]
            texfile.write("{} \\\\ \hline \n".format(" & ".join(row)))
            print sdx+1, symbol, R_c, jb


def plot_best_parameters(prob_type='min_cvar_sp'):
    """
    according to the mean cum_roi
    """
    import csv

    sample_dates = [date(2005,1,3), date(2006, 1, 2), date(2007, 1, 2),
                    date(2008, 1, 2), date(2009, 1, 5), date(2010,1,4),
                    date(2011, 1, 3), date(2012,1,2), date(2013, 1, 2),
                    date(2014,1,2), date(2014, 12, 31)]
    if prob_type == "min_cvar_sp":
        params = [(5, 150, 0.8), (10, 90, 0.5), (15, 100, 0.65),
                  (20, 110, 0.6), (25, 120, 0.55), (30, 190, 0.7),
                  (35, 120, 0.55), (40, 100, 0.5), (45, 120, 0.55),
                  (50, 120, 0.55)]
    elif prob_type == "min_cvar_sip":
        params = [(5, 200, 0.5), (10, 130, 0.5), (15, 120, 0.5),
                  (20, 120, 0.5), (25, 120, 0.55), (30, 120, 0.55),
                  (35, 120, 0.55), (40, 120, 0.55), (45, 120, 0.55),
                  (50, 120, 0.55)]
    elif prob_type == "bah":
        params = range(5, 55, 5)

    file_name = os.path.join(TMP_DIR, 'best_{}_process.csv'.format(prob_type))
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile)

        heads = [d for d in sample_dates]
        heads.insert(0, "param")
        heads.append("JB")
        heads.append('ADF')
        writer.writerow(heads)
        for p in params:
            print p
            if prob_type in ('min_cvar_sp', 'min_cvar_sip'):
                wealths = None
                JB, ADF = 1,1
                for s_cnt in xrange(1, 4):
                    res = load_results(prob_type, p[0], p[1],
                                       scenario_cnt=s_cnt, alpha=p[2])
                    wealth_proc = (res['wealth_df'].sum(axis=1) + res[
                            'risk_free_wealth'])
                    if wealths is None:
                        wealths = wealth_proc
                    else:
                        wealths +=  wealth_proc

                    rois =  wealth_proc.pct_change()
                    rois[0] = 0

                    jb = stat_tools.jarque_bera(rois)[1]
                    adf_c = tsa_tools.adfuller(rois, regression='c')[1]
                    adf_ct = tsa_tools.adfuller(rois, regression='ct')[1]
                    adf_ctt = tsa_tools.adfuller(rois, regression='ctt')[1]
                    adf_nc = tsa_tools.adfuller(rois, regression='nc')[1]
                    adf = max(adf_c, adf_ct, adf_ctt, adf_nc)
                    if jb < JB:
                        JB = jb
                    if adf < ADF:
                        ADF = adf

                vals = [wealths.loc[s_date]/3/1e6 for s_date in sample_dates]
                vals.insert(0, "({},{},{:.0%})".format(
                        p[0],p[1],p[2]))
                vals.append(JB)
                vals.append(ADF)
                writer.writerow(vals)

            elif prob_type == "bah":
                res = load_results(prob_type, p)
                wealths = res['wealth_df'].sum(axis=1) + res['risk_free_wealth']
                vals = [wealths.loc[s_date]/1e6 for s_date in sample_dates]
                vals.insert(0, p)
                writer.writerow(vals)


def table_best_parameter(prob_type='min_cvar_sp'):
    """
    res columns
    Index([u'n_stock', u'win_length', u'alpha', u'scenario_cnt',
    u'start_date', u'end_date', u'n_exp_period', u'trans_fee_loss',
    u'cum_roi',
       u'daily_roi', u'daily_mean_roi', u'daily_std_roi', u'daily_skew_roi',
       u'daily_kurt_roi', u'sharpe', u'sortino_full', u'sortino_partial',
       u'max_abs_drawdown', u'SPA_l_pvalue', u'SPA_c_pvalue', u'SPA_u_pvalue',
       u'simulation_time'],
      dtype='object')
    """

    if prob_type == "min_cvar_sp":
        df = pd.read_pickle(os.path.join(EXP_SP_PORTFOLIO_REPORT_DIR,
                                "min_cvar_sp_results_all.pkl"))
    elif prob_type == "min_cvar_sip":
        df = pd.read_pickle(os.path.join(EXP_SP_PORTFOLIO_REPORT_DIR,
                                "min_cvar_sip_results_all.pkl"))
     # set alpha column to str
    for rdx in xrange(df.index.size):
        df.ix[rdx, 'alpha'] = "{:.2f}".format(df.ix[rdx, 'alpha'])

    n_stocks = range(5, 55, 5)
    win_lengths = range(50, 240 + 10, 10)
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')

    texnames = ["n_stock", 'win_length', 'alpha' , "R_c", "R_a", "mu", "std",
                    "skew", "kurt", "sharpe", "sortino",  "SPA"]
    res_df = pd.DataFrame(
            np.zeros((len(n_stocks) * len(win_lengths) * len(alphas),
                      len(texnames))),
            columns=texnames
    )
    # the best parameter by portfolio size
    res_best_df = pd.DataFrame(np.zeros((len(n_stocks), len(texnames))),
                               columns=texnames)

    count = 0
    for n_stock in n_stocks:
        for win_length in win_lengths:
            for alpha in alphas:
                if prob_type == "min_cvar_sp":
                    values =  df.loc[(df.loc[:,'n_stock']==n_stock) &
                                          (df.loc[:, 'win_length'] == win_length) &
                                          (df.loc[:, 'alpha'] == alpha), :]
                elif prob_type == "min_cvar_sip":
                    values =  df.loc[(df.loc[:,'max_portfolio_size']==n_stock) &
                                          (df.loc[:, 'win_length'] == win_length) &
                                          (df.loc[:, 'alpha'] == alpha), :]
                res_df.ix[count, 'n_stock'] = n_stock
                res_df.ix[count, 'win_length'] = win_length
                res_df.ix[count, 'alpha'] = float(alpha)

                cum_roi = values['cum_roi'].mean()
                res_df.ix[count, 'R_c'] = cum_roi

                ann_roi = np.power(cum_roi+1, 1./10) -1
                res_df.ix[count, 'R_a'] = ann_roi
                res_df.ix[count, 'mu'] =values['daily_mean_roi'].mean()
                res_df.ix[count, 'std'] =values['daily_std_roi'].mean()
                res_df.ix[count, 'skew'] =values['daily_skew_roi'].mean()
                res_df.ix[count, 'kurt'] =values['daily_kurt_roi'].mean()
                res_df.ix[count, 'sharpe'] =values['sharpe'].mean()
                res_df.ix[count, 'sortino'] =values['sortino_full'].mean()
                if len(values['SPA_c_pvalue']) > 0:
                    res_df.ix[count, 'SPA'] =max(values['SPA_c_pvalue'])

                print n_stock, win_length, alpha, "OK"

                # update
                count += 1
    # print res_df
    res_df.to_excel(os.path.join(TMP_DIR, 'best_{}.xlsx').format(prob_type))

    for ndx, n_stock in enumerate(n_stocks):
        tmp_df = res_df.loc[res_df.loc[:,'n_stock']==n_stock, :]
        best_rec = tmp_df.sort_values('R_c', ascending=False).iloc[0]
        res_best_df.iloc[ndx] = best_rec


    res_best_df.to_excel(os.path.join(
            TMP_DIR, 'best_mean_stock_{}.xlsx').format(prob_type))
    res_best_df.to_pickle(os.path.join(
            TMP_DIR, 'best_mean_stock_{}.pkl').format(prob_type))

def best_mean_stock_latex(prob_type='min_cvar_sip'):
    pkl = os.path.join(TMP_DIR, "best_mean_stock_{}.pkl".format(prob_type))
    df = pd.read_pickle(pkl)
    print df.columns
    with open(os.path.join(TMP_DIR, "best_mean_stock_{}.txt".format(prob_type)), 'wb') as \
            texfile:
        for rdx in xrange(df.index.size):
            param = "({:.0f}, {:.0f}, {:.0f}\%)".format(df.ix[rdx, 'n_stock'],
                                            df.ix[rdx, 'win_length'],
                                            df.ix[rdx, 'alpha']*100)
            row = ["{:>14}".format(param),
                   "{:>6.2f}".format(df.ix[rdx, 'R_c']*100),
                   "{:>4.2f}".format(df.ix[rdx, 'R_a']*100),
                   "{:>6.4f}".format(df.ix[rdx, 'mu']*100),
                   "{:>6.4f}".format(df.ix[rdx, 'std']*100),
                   "{:>5.2f}".format(df.ix[rdx, 'skew']),
                   "{:>4.2f}".format(df.ix[rdx, 'kurt']),
                   "{:>4.2f}".format(df.ix[rdx, 'sharpe']*100),
                   "{:>4.2f}".format(df.ix[rdx, 'sortino']*100),
                   "{:<3} {:>4.2f}".format(significant_star(0), 0*100),
                   "{:<3} {:>4.2f}".format(significant_star(0), 0*100),
                   "{:<3} {:>4.2f}".format(significant_star(df.ix[rdx, 'SPA']),
                                           df.ix[rdx, 'SPA'] * 100),
                   ]
            texfile.write("{} \\\\ \hline \n".format(" & ".join(row)))
            print param



if __name__ == '__main__':
    # all_results_to_sheet_xlsx("min_cvar_sp", "n_stock")
    # all_results_to_sheet_xlsx("min_cvar_sp", "win_length")
    # all_results_to_sheet_xlsx("min_cvar_sp", "alpha")

    # all_results_to_sheet_xlsx("min_cvar_sip", "n_stock")
    # all_results_to_sheet_xlsx("min_cvar_sip", "win_length")
    # all_results_to_sheet_xlsx("min_cvar_sip", "alpha")

    # all_results_to_4dpanel(prob_type="min_cvar_sp")
    # all_results_to_4dpanel(prob_type="min_cvar_sip")

    # all_results_to_xlsx("min_cvar_sp")
    # all_results_to_xlsx("min_cvar_sip")
    # all_results_to_xlsx("min_cvar_eev")

    # all_results_roi_stats("min_cvar_sp")
    # all_results_roi_stats("min_cvar_sip")
    # plot_3d_results("min_cvar_sip")
    # plot_3d_results("min_cvar_eev", z_dim='cum_roi')
    plot_2d_contour("min_cvar_sp")
    # plot_3d_eev()
    # plot_3d_results("min_cvar_sp", z_dim='ann_roi')
    # plot_3d_results("min_cvar_sip", z_dim='sortino_full')
    # plot_3d_results("min_cvar_sp", z_dim='SPA_c_pvalue')
    # plot_3d_results("min_cvar_sip", z_dim='SPA_c_pvalue')
    # bah_results_to_xlsx()
    # plot_best_parameters("min_cvar_sp")
    # plot_best_parameters("min_cvar_sip")
    # plot_best_parameters("bah")
    # stock_statistics()
    # bah_results_to_latex()
    # table_best_parameter("min_cvar_sp")
    # table_best_parameter("min_cvar_sip")
    # best_mean_stock_latex()
    pass