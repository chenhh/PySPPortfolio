# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

import platform
import pandas as pd
from datetime import date
import atexit
import glob
import os
from PySPPortfolio.pysp_portfolio import *
from PySPPortfolio.pysp_portfolio.etl import generating_scenarios


def all_parameters_combination():
    """
    n_stock: {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
    win_length: {50, 60, ..., 240}
    n_scenario:  200
    biased: {unbiased, bias}
    cnt: {1,2,3}

    """
    exp_start_date, exp_end_date = START_DATE, END_DATE
    all_params = ["{}_{}_m{}_w{}_s{}_{}_{}".format(
                exp_start_date.strftime("%Y%m%d"),
                exp_end_date.strftime("%Y%m%d"),
                n_stock, win_length, n_scenario,bias, cnt)
                  for cnt in xrange(1, 4)
                  for bias in ("unbiased",)
                  for n_scenario in (200,)
                  for win_length in xrange(50, 240 + 10, 10)
                  for n_stock in xrange(5, 50 + 5, 5)
                  ]
    return set(all_params)

def checking_generated_scenarios(scenario_path=None):
    """
    checking unfinished experiment parameters.
    """
    if scenario_path is None:
        scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios')

    # get all params
    all_params = all_parameters_combination()

    pkls = glob.glob(os.path.join(scenario_path, "*.pkl"))
    for pkl in pkls:
        param = pkl[pkl.rfind(os.sep)+1: pkl.rfind('.')]

        if param in all_params:
            all_params.remove(param)
            print ("{} has finished.".format(param))
        else:
            print ("{} not in exp parameters.".format(param))

    # unfinished params
    return all_params

def checking_working_parameters(scenario_path=None, log_file=None):
    """
    if a parameter is under working and not write to pkl,
    it is recorded to a file
    """
    if scenario_path is None:
        scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios')

    if log_file is None:
        log_file = 'working.pkl'

    # get all params
    all_params = all_parameters_combination()

    # storing a dict, key: param, value: platform_name
    file_path = os.path.join(scenario_path, log_file)
    if not os.path.exists(file_path):
        # no working parameters
        return all_params

    data = pd.read_pickle(file_path)
    for param, node in data.items():
        if param in all_params:
            all_params.remove(param)
            print ("{} under processing on {}.".format(param, node))
        else:
            print ("{} not in exp parameters.".format(param))

    # unfinished params
    return all_params


def dispatch_scenario_parameters(scenario_path=None, log_file=None):

    if scenario_path is None:
        scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios')

    if log_file is None:
        # storing a dict, {key: param, value: platform_name}
        log_file = 'working.pkl'

    unfinished_params = all_parameters_combination()

    # reading working pkl
    log_path = os.path.join(scenario_path, log_file)

    while len(unfinished_params) > 0:
        params1 = checking_generated_scenarios(scenario_path)
        params2 = checking_working_parameters(scenario_path, log_file)
        unfinished_params.remove(params1)
        unfinished_params.remove(params2)

        param = unfinished_params.pop()
        _, _, stock, win, scenario, biased, _ = param.split('_')
        n_stock =int(stock[stock.rfind('m')+1:])
        win_length = int(win[win.rfind('w')+1:])
        n_scenario = int(scenario[scenario.rfind('s')+1:])
        bias = True if biased == "biased" else False

        # log  parameter to file
        if not os.path.exists(log_path):
            working_dict = {}
        else:
            working_dict = pd.read_pickle(log_path)

        working_dict[param] = platform.node()
        pd.to_pickle(working_dict, log_path)

        # generating scenarios
        try:
            generating_scenarios(n_stock, win_length, n_scenario, bias)
        except Exception as e:
            print param, e
        finally:
            working_dict = pd.read_pickle(log_path)
            del working_dict[param]
            pd.to_pickle(working_dict, log_path)

if __name__ == '__main__':
    dispatch_scenario_parameters()
