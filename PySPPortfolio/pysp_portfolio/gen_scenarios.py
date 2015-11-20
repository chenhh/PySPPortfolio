# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

import platform
import pandas as pd
from datetime import date
import time
import glob
import os
from PySPPortfolio.pysp_portfolio import *
from PySPPortfolio.pysp_portfolio.etl import generating_scenarios


def all_parameters_combination_name():
    """
    file_name of all experiment parameters
    n_stock: {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
    win_length: {50, 60, ..., 240}
    n_scenario:  200
    biased: {unbiased, bias}
    cnt: {1,2,3}
    combinations: 10 * 20 * 3 = 600 (only unbiased)
    """
    exp_start_date, exp_end_date = START_DATE, END_DATE
    all_params = ["{}_{}_m{}_w{}_s{}_{}_{}".format(
                exp_start_date.strftime("%Y%m%d"),
                exp_end_date.strftime("%Y%m%d"),
                n_stock, win_length, n_scenario,bias, cnt)
                  for cnt in xrange(1, 3+1)
                  for bias in ("unbiased",)
                  for n_scenario in (200,)
                  for win_length in xrange(50, 240 + 10, 10)
                  for n_stock in xrange(5, 50 + 5, 5)
                  ]
    return set(all_params)

def checking_generated_scenarios(scenario_path=None):
    """
    return unfinished experiment parameters.
    """
    if scenario_path is None:
        scenario_path = EXP_SCENARIO_DIR

    # get all params
    all_params = all_parameters_combination_name()

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
        scenario_path = EXP_SCENARIO_DIR

    if log_file is None:
        log_file = 'working.pkl'

    # get all params
    all_params = all_parameters_combination_name()

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
        scenario_path = EXP_SCENARIO_DIR

    if log_file is None:
        # storing a dict, {key: param, value: platform_name}
        log_file = 'working.pkl'

    # reading working pkl
    log_path = os.path.join(scenario_path, log_file)

    params1 = checking_generated_scenarios(scenario_path)
    params2 = checking_working_parameters(scenario_path, log_file)
    unfinished_params = params1.intersection(params2)

    print ("initial unfinished params: {}".format(len(unfinished_params)))

    while len(unfinished_params) > 0:
        # each loop we have to
        params1 = checking_generated_scenarios(scenario_path)
        params2 = checking_working_parameters(scenario_path, log_file)
        unfinished_params = params1.intersection(params2)

        print ("current unfinished params: {}".format(len(unfinished_params)))
        if len(unfinished_params) <= 100:
            for u_param in unfinished_params:
                print ("unfinished: {}".format(u_param))

        param = unfinished_params.pop()
        _, _, stock, win, scenario, biased, _ = param.split('_')
        n_stock =int(stock[stock.rfind('m')+1:])
        win_length = int(win[win.rfind('w')+1:])
        n_scenario = int(scenario[scenario.rfind('s')+1:])
        bias = True if biased == "biased" else False

        # don't do this parameter, because don't have enought win_length
        if n_stock == 50 and win_length == 50:
            continue

        # log  parameter to file
        if not os.path.exists(log_path):
            working_dict = {}
        else:
            working_dict = pd.read_pickle(log_path)

        working_dict[param] = platform.node()
        for retry in xrange(3):
            try:
                # preventing multi-process write file at the same time
                pd.to_pickle(working_dict, log_path)
            except IOError as e:
                if retry == 2:
                    raise Exception(e)
                else:
                    print ("working retry: {}, {}".format(retry+1, e))
                    time.sleep(2)

        # generating scenarios
        try:
            print ("gen scenario: {}".format(param))
            generating_scenarios(n_stock, win_length, n_scenario, bias)
        except Exception as e:
            print param, e
        finally:
            working_dict = pd.read_pickle(log_path)
            if param in working_dict.keys():
                del working_dict[param]
            else:
                print ("can't find {} in working dict.".format(param))
            for retry in xrange(3):
                try:
                    # preventing multi-process write file at the same time
                    pd.to_pickle(working_dict, log_path)
                except IOError as e:
                    if retry == 2:
                        raise Exception(e)
                    else:
                        print ("finally retry: {}, {}".format(retry+1, e))
                        time.sleep(2)


def read_working_parameters():
    scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios')
    log_file = 'working.pkl'
    file_path = os.path.join(scenario_path, log_file)

    if not os.path.exists(file_path):
        print ("{} not exists.".format(file_path))
    else:
        working_dict = pd.read_pickle(file_path)
        for param, node in working_dict.items():
            print param, node

if __name__ == '__main__':
    dispatch_scenario_parameters()
    # read_working_parameters()
