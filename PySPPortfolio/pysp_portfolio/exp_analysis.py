# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
import os
import pandas as pd
import numpy as np
from PySPPortfolio.pysp_portfolio import *
from gen_results import (all_experiment_parameters,)

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
    if prob_type in ("min_cvar_sp", "ms_min_cvar_sp", "min_cvar_eev"):
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

def all_results_to_sheet_xlsx(prob_type="min_cvar_sip", sheet="alpha"):
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
    cnts = range(1, 3+1)
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')
    columns = ['n_stock', 'win_length', 'alpha', "scenario_cnt",
              'start_date', 'end_date', 'n_exp_period',
              'trans_fee_loss',
              'cum_roi', 'daily_roi', 'daily_mean_roi', 'daily_std_roi',
              'daily_kurt_roi', 'sharpe', 'sortino_full', 'sortino_partial',
              'max_abs_drawdown', 'SPA_l_pvalue', 'SPA_c_pvalue',
              'SPA_u_pvalue', 'simulation_time']


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

    params = all_experiment_parameters(prob_type)
    n_param = len(params)
    for rdx, (m, w, n, b, c, a) in enumerate(params):
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

            print ("[{}/{}] {} {}: {}_{} OK".format(
                rdx +1, n_param, prob_type, sheet , major, a))

    results_panel.to_excel(os.path.join(TMP_DIR,
                                '{}_{}.xlsx'.format(prob_type, sheet)))


def all_results_to_xlsx(prob_type="min_cvar_sp"):
    """ output results to a single sheet """
    n_stocks = range(5, 50 + 5, 5)
    win_lengths = range(50, 240 + 10, 10)
    cnts = range(1, 3+1)
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')
    columns = ['n_stock', 'win_length', 'alpha', "scenario_cnt",
              'start_date', 'end_date', 'n_exp_period',
              'trans_fee_loss',
              'cum_roi', 'daily_roi', 'daily_mean_roi', 'daily_std_roi',
              'daily_kurt_roi', 'sharpe', 'sortino_full', 'sortino_partial',
              'max_abs_drawdown', 'SPA_l_pvalue', 'SPA_c_pvalue',
              'SPA_u_pvalue', 'simulation_time']

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

    params = all_experiment_parameters(prob_type)
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

            print ("[{}/{}] {} {} OK".format(rdx +1, n_param, key))
    print ("{} OK".format(prob_type))
    result_df.to_excel(os.path.join(TMP_DIR,
                                '{}_results_all.xlsx'.format(prob_type)))
    pd.to_pickle(result_df, os.path.join(TMP_DIR,
                                '{}_results_all.pkl'.format(prob_type)))

def all_results_to_4dpanel(prob_type="min_cvar_sp"):
    """
    axis_0: n_stock
    axis_1: win_length
    axis_2: alpha
    axis_3: cnt + columns
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

    params = all_experiment_parameters()
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

    for cnt in xrange(3):
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
    n_stocks = range(5, 50+5, 5)
    columns = ['n_stock', 'start_date', 'end_date', 'n_exp_period',
               'trans_fee_loss', 'cum_roi', 'daily_roi', 'daily_mean_roi',
               'daily_std_roi', 'daily_kurt_roi', 'sharpe', 'sortino_full',
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


def all_results_roi_stats():
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
    fin = os.path.join(EXP_SP_PORTFOLIO_DIR, 'reports',
                       'min_cvar_sp_results_all.pkl')
    df = pd.read_pickle(fin)
    grouped = df.groupby(['n_stock', 'win_length', 'alpha'])
    stats_df = pd.DataFrame([grouped.mean()['cum_roi'],
                             grouped.std()['cum_roi']],
                            index=("cum_roi_mean", "cum_roi_std"))
    stats_df = stats_df.T
    stats_df.reset_index(inplace=True)
    print stats_df
    stats_df.to_excel(os.path.join(TMP_DIR,
                                       'min_cvar_sp_results_roi_stats.xlsx'))


def plot_4d_results(prob_type="min_cvar_sp", dim_z="alpha"):
    """
    axis-0: n_stock
    axis-1: win_length
    axis-2: alpha
    axis-3: cum_roi (annualized roi)

    df columns:
    Index([u'n_stock', u'win_length', u'alpha', u'scenario_cnt', u'start_date',
       u'end_date', u'n_exp_period', u'trans_fee_loss', u'cum_roi',
       u'daily_roi', u'daily_mean_roi', u'daily_std_roi', u'daily_kurt_roi',
       u'sharpe', u'sortino_full', u'sortino_partial', u'max_abs_drawdown',
       u'SPA_l_pvalue', u'SPA_c_pvalue', u'SPA_u_pvalue', u'simulation_time'],
      dtype='object')


    colormap:
    http://matplotlib.org/examples/color/colormaps_reference.html

    """
    if not prob_type in ("min_cvar_sp",):
        raise ValueError("unknown problem type: {}".format(prob_type))

    data_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'reports',
                       '{}_results_all.pkl'.format(prob_type))
    df = pd.read_pickle(data_path)
    # print df.loc[(df.loc[:, 'win_length']==100),
    #              'cum_roi']
    # print df.loc[(df.loc[:,'n_stock']==5) & (df.loc[:,'win_length']==100),
    #              'cum_roi']

    # axes
    stocks = np.arange(5, 50+5, 5)  # 10
    lengths = np.arange(50, 240 + 10, 10)   #20
    alphas = ('0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80',
                   '0.85', '0.90', '0.95')

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()
    fig.figsize(64,48)
    if dim_z == "n_stock":
        for mdx, n_stock in enumerate(stocks):
            ax = fig.add_subplot(2, 5, mdx+1, projection='3d')
            ax.set_title('n_stock: {}'.format(n_stock))
            v_alphas = [float(v) for v in alphas]
            Xs, Ys = np.meshgrid(lengths, v_alphas)
            Zs = np.zeros_like(Xs, dtype=np.float)

            n_row, n_col = Xs.shape
            for rdx in xrange(n_row):
                for cdx in xrange(n_col):
                    win_length, alpha = Xs[rdx, cdx], Ys[rdx, cdx]
                    cum_rois =  df.loc[(df.loc[:,'n_stock']==n_stock) &
                                  (df.loc[:, 'win_length'] == win_length) &
                                  (df.loc[:, 'alpha'] == alpha),
                                  'cum_roi']
                    mean = cum_rois.mean()
                    Zs[rdx, cdx] = 0 if np.isnan(mean) else mean

            print Zs
             # projected on z
            cset = ax.contour(Xs, Ys, Zs, zdir='z', offset=-1,
                              cmap=plt.cm.coolwarm)
            ax.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2, alpha=0.4,
                            cmap=plt.cm.coolwarm)
            ax.set_xlabel('window length')
            ax.set_xlim(40, 240)
            ax.set_ylabel('alpha')
            ax.set_ylim(0.5, 1)
            ax.set_zlabel('cumulative ROI')
            ax.set_zlim(-1, 3)

    if dim_z == "win_length":
        for wdx, win_length in enumerate(lengths):
            ax = fig.add_subplot(4, 5, wdx+1, projection='3d')
            v_alphas = [float(v) for v in alphas]
            Xs, Ys = np.meshgrid(stocks, v_alphas)
            Zs = np.zeros_like(Xs, dtype=np.float)

            n_row, n_col = Xs.shape
            for rdx in xrange(n_row):
                for cdx in xrange(n_col):
                    n_stock, alpha = Xs[rdx, cdx], Ys[rdx, cdx]
                    cum_rois =  df.loc[(df.loc[:,'n_stock']==n_stock) &
                                  (df.loc[:, 'win_length'] == win_length) &
                                  (df.loc[:, 'alpha'] == alpha),
                                  'cum_roi']
                    mean = cum_rois.mean()
                    Zs[rdx, cdx] = 0 if np.isnan(mean) else mean

            print Zs
             # projected on z
            cset = ax.contour(Xs, Ys, Zs, zdir='z', offset=-1,
                              cmap=plt.cm.coolwarm)
            ax.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2, alpha=0.4,
                            cmap=plt.cm.coolwarm)
            ax.set_xlabel('n_stock')
            ax.set_xlim(0, 50)
            ax.set_ylabel('alpha')
            ax.set_ylim(0.5, 1)
            ax.set_zlabel('cumulative ROI')
            ax.set_zlim(-1, 3)

    if dim_z == "alpha":
        # position=fig.add_axes([0.93, 0.1, 0.02, 0.35])
        # cbar = plt.colorbar(cset, cax=position)

        # color normalization
        cm_norm = mpl.colors.Normalize(vmin=-1, vmax=3, clip=False)


        for adx, value in enumerate(alphas):
            alpha = float(value)
            ax = fig.add_subplot(2, 5, adx+1, projection='3d')
            ax.set_title('alpha: {:.0%}'.format(alpha))
            Xs, Ys = np.meshgrid(stocks, lengths)
            Zs = np.zeros_like(Xs, dtype=np.float)
            # stds = np.zeros_like(Xs, dtype=np.float)

            n_row, n_col = Xs.shape
            for rdx in xrange(n_row):
                for cdx in xrange(n_col):
                    n_stock, win_length = Xs[rdx, cdx], Ys[rdx, cdx]
                    cum_rois =  df.loc[(df.loc[:,'n_stock']==n_stock) &
                                  (df.loc[:, 'win_length'] == win_length) &
                                  (df.loc[:, 'alpha'] == alpha),
                                  'cum_roi']
                    # annualized_rois = np.power(cum_rois+1, 1./10) -1
                    mean = cum_rois.mean()
                    Zs[rdx, cdx] = 0 if np.isnan(mean) else mean

            print adx, Zs
            p = ax.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2, alpha=0.6,
                            cmap=plt.cm.coolwarm, norm=cm_norm
                            )
            # fig.colorbar(p)
            ax.set_xlabel('portfolio size', fontsize=14)
            # ax.tick_params(axis='both', labelsize=8)

            ax.set_xlim(0, 50)
            ax.set_ylabel('window length', fontsize=14)
            ax.set_ylim(40, 240)
            ax.set_zlabel('cumulative ROI', fontsize=14)
            ax.set_zlim(-1, 2.8)

            # projected on z
            cset = ax.contourf(Xs, Ys, Zs, zdir='z', offset=-1, alpha=0.6,
                              cmap=plt.cm.coolwarm, norm=cm_norm)
            # color bar
            # cbaxes = ax.add_axes([0.8, 0.1, 0.03, 0.8])
            # cbar = fig.colorbar(cset)
            # cbar.ax.set_ylabel('verbosity coefficient')

        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(p, cax=cbar_ax)

    plt.show()



if __name__ == '__main__':
    # all_results_to_sheet_xlsx("min_cvar_sip", "n_stock")
    # all_results_to_sheet_xlsx("min_cvar_sip", "win_length")
    # all_results_to_sheet_xlsx("min_cvar_sip", "alpha")
    # all_results_to_4dpanel(prob_type="min_cvar_sp")
    # all_results_to_xlsx()
    # all_results_roi_stats()
    plot_4d_results(dim_z="alpha")
    # plot_results()
    # reports = load_results("bah", 5)
    # print reports
    # bah_results_to_xlsx()
    # wdf = reports['wealth_df']
    # wfree = reports['risk_free_wealth']
    # warr = wdf.sum(axis=1) + wfree
    # warr[0] = 0
    # print warr
    # print warr.pct_change()
