# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""



def run_min_cvar_sp_simulation(n_stock, window_length, alpha, n_scenario=200):
    """
    :return: reports
    """
    from ipro.dev import (EXP_SYMBOLS, DROPBOX_UP_EXPERIMENT_DIR)

    n_stock = int(n_stock)
    window_length = int(window_length)
    alpha = float(alpha)

    symbols = EXP_SYMBOLS[:n_stock]
    risk_rois = generate_rois_df(symbols)
    start_date = date(2005, 1, 1)
    end_date = date(2015, 4, 30)

    exp_risk_rois = risk_rois.loc[start_date:end_date]
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6

    obj = MinCVaRSPPortfolio(symbols, risk_rois, risk_free_rois,
                           initial_risk_wealth,
                           initial_risk_free_wealth, start_date=start_date,
                           end_date=end_date, window_length=window_length,
                           alpha=alpha, n_scenario=n_scenario, verbose=False)

    reports = obj.run()
    print reports

    file_name = '{}_SP_{}-{}_m{}_w{}_a{:.2f}_s{}.pkl'.format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        exp_risk_rois.index[0].strftime("%Y%m%d"),
        exp_risk_rois.index[-1].strftime("%Y%m%d"),
        len(symbols),
        window_length,
        alpha,
        n_scenario)

    file_dir = os.path.join(DROPBOX_UP_EXPERIMENT_DIR, 'cvar_sp')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))


    return reports

def run_min_cvar_sip_simulation(max_portfolio_size, window_length, alpha,
                             n_scenario=200):
    """
    :return: reports
    """
    from ipro.dev import (EXP_SYMBOLS, DROPBOX_UP_EXPERIMENT_DIR)

    max_portfolio_size = int(max_portfolio_size)
    window_length = int(window_length)
    alpha = float(alpha)

    symbols = EXP_SYMBOLS
    n_stock = len(symbols)
    risk_rois = generate_rois_df(symbols)
    start_date = date(2005, 1, 1)
    end_date = date(2015, 4, 30)

    exp_risk_rois = risk_rois.loc[start_date:end_date]
    n_period = exp_risk_rois.shape[0]
    risk_free_rois = pd.Series(np.zeros(n_period), index=exp_risk_rois.index)
    initial_risk_wealth = pd.Series(np.zeros(n_stock), index=symbols)
    initial_risk_free_wealth = 1e6

    obj = MinCVaRSIPPortfolio(symbols, max_portfolio_size,
                            risk_rois, risk_free_rois,
                            initial_risk_wealth,
                            initial_risk_free_wealth, start_date=start_date,
                            end_date=end_date, window_length=window_length,
                            alpha=alpha, n_scenario=n_scenario, verbose=False)

    reports = obj.run()
    print reports

    file_name = '{}_SIP_{}-{}_m{}_mc{}_w{}_a{:.2f}_s{}.pkl'.format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        exp_risk_rois.index[0].strftime("%Y%m%d"),
        exp_risk_rois.index[-1].strftime("%Y%m%d"),
        max_portfolio_size,
        len(symbols),
        window_length,
        alpha,
        n_scenario)

    file_dir = os.path.join(DROPBOX_UP_EXPERIMENT_DIR, 'cvar_sip')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))

    return reports



if __name__ == '__main__':
    import sys
    import argparse

    sys.path.append(os.path.join(os.path.abspath('..'), '..'))
    parser = argparse.ArgumentParser()

    test_min_cvar_sp_portfolio()

    # parser.add_argument("-m", "--n_stock", required=True, type=int,
    #                     choices=range(5, 55, 5))
    # parser.add_argument("-w", "--win_length", required=True, type=int)
    # parser.add_argument("-a", "--alpha", required=True)
    # args = parser.parse_args()
    #
    # run_min_cvar_sp_simulation(args.n_stock, args.win_length, args.alpha)