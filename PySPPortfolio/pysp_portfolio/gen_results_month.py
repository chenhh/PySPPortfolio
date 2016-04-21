# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
import platform
import numpy as np
import pandas as pd
from datetime import date
import time
import glob
import os
from PySPPortfolio.pysp_portfolio import *
from exp_cvar import (run_min_ms_cvar_eventsp_simulation, )
from gen_results import (retry_read_pickle, retry_write_pickle)


def get_results_dir(prob_type):
    """
    Parameter:
    ----------------
    prob_type: str, {min_cvar_sp, min_cvar_sip}

    Returns:
    ----------------
    return the results directory of given problem type
    """
    if prob_type in ("min_ms_cvar_eventsp", "min_cvar_sp2",):
        return os.path.join(EXP_SP_PORTFOLIO_DIR, prob_type)

    else:
        raise ValueError("unknown prob_type: {}".format(prob_type))


def get_month_pairs():
    month_pairs = []
    for year in xrange(2005, 2014 + 1):
        for month in xrange(1, 12 + 1):
            if month in (1, 3, 5, 7, 8, 10, 12):
                month_pairs.append((date(year, month, 1),
                                    date(year, month, 31)))
            elif month in (4, 6, 9, 11):
                month_pairs.append((date(year, month, 1),
                                    date(year, month, 30)))
            else:
                month_pairs.append((date(year, 2, 1),
                                    date(year, 2, 28)))
    return month_pairs


def all_experiment_parameters(prob_type, max_scenario_cnts):
    """
    file_name of all experiment parameters
    (n_stock, win_length, n_scenario, bias, cnt, alpha, start_date, end_date)

    """
    all_params = []
    for pair in get_month_pairs():
        for cnt in xrange(1, max_scenario_cnts + 1):
            all_params.append((5, 120, 200, "unbiased", cnt, "0.50",
                               pair[0], pair[1]))
    return set(all_params)


def checking_finished_parameters(prob_type, max_scenario_cnts):
    """
    return unfinished experiment parameters.
    example:
    min_cvar_sip_20050103_20141231_all50_m5_w100_s200_unbiased_1_a0.95
    ['min', 'cvar', 'sp', '20050103', '20141231', 'm5', 'w50', 's200',
     'unbiased', '1', 'a0.95']

    min_cvar_sp_20050103_20141231_m5_w50_s200_unbiased_1_a0.95
    ['min', 'cvar', 'sip', '20050103', '20141231', 'all50', 'm5', 'w100',
     's200', 'unbiased', '1', 'a0.95']
    """
    dir_path = get_results_dir(prob_type)

    # get all params
    all_params = all_experiment_parameters(prob_type, max_scenario_cnts)

    if prob_type in ("min_ms_cvar_eventsp"):
        # min_ms_cvar_eventsp_20050103_20050105_m5_w70_s200_unbiased_1_a0.95
        pkls = glob.glob(os.path.join(dir_path, "{}_20*.pkl".format(
            prob_type)))

    for pkl in pkls:
        name = pkl[pkl.rfind(os.sep) + 1: pkl.rfind('.')]
        exp_params = name.split('_')
        if prob_type in ("min_ms_cvar_eventsp",):
            d1, d2 = exp_params[4], exp_params[5]
            params = exp_params[6:]
            start_date = date(int(d1[:4]), int(d1[4:6]), int(d1[6:8]))
            end_date = date(int(d2[:4]), int(d2[4:6]), int(d2[6:8]))

        n_stock = int(params[0][params[0].rfind('m') + 1:])
        win_length = int(params[1][params[1].rfind('w') + 1:])
        n_scenario = int(params[2][params[2].rfind('s') + 1:])
        bias = params[3]
        scenario_cnt = int(params[4])
        alpha = params[5][params[5].rfind('a') + 1:]

        data_param = (n_stock, win_length, n_scenario, bias, scenario_cnt,
                      alpha, start_date, end_date)

        if data_param in all_params:
            all_params.remove(data_param)

    # return unfinished params
    return all_params


def checking_working_parameters(prob_type, max_scenario_cnts, retry_cnt=5):
    """
    if a parameter is under working and not write to pkl,
    it is recorded to a file

    working dict:
        - key: str,  "|".join(str(v) for v in param)
        - value: machine node
    """
    dir_path = get_results_dir(prob_type)
    log_file = '{}_working.pkl'.format(prob_type)

    # get all params
    all_params = all_experiment_parameters(prob_type, max_scenario_cnts)

    # storing a dict, key: param, value: platform_name
    file_path = os.path.join(dir_path, log_file)
    if not os.path.exists(file_path):
        # no working parameters
        print ("{} no working log file.".format(prob_type))
        return all_params

    data = retry_read_pickle(file_path)
    for param_key, node in data.items():
        # key:  (n_stock, win_length, _scenario, biased, cnt, alpha, start_date,
        # end_date)
        keys = param_key.split('|')

        param = (int(keys[0]), int(keys[1]), int(keys[2]), keys[3],
                 int(keys[4]), keys[5],
                 date(*map(lambda x:int(x), keys[6].split('-'))),
                 date(*map(lambda x: int(x), keys[7].split('-'))))

        if param in all_params:
            all_params.remove(param)
            print ("working {}: {} is running on {}.".format(
                prob_type, param, node))
    print ("#. of running parameters: {}".format(len(data)))

    # unfinished params
    return all_params


def dispatch_experiment_parameters(prob_type, max_scenario_cnts):
    dir_path = get_results_dir(prob_type)

    # storing a dict, {key: param, value: platform_name}
    log_file = '{}_working.pkl'.format(prob_type)

    # reading working pkl
    log_path = os.path.join(dir_path, log_file)
    params1 = checking_finished_parameters(prob_type, max_scenario_cnts)
    params2 = checking_working_parameters(prob_type, max_scenario_cnts)
    unfinished_params = params1.intersection(params2)

    print ("dispatch: {}, #. unfinished parameters: {}".format(
        prob_type, len(unfinished_params)))

    while len(unfinished_params) > 0:
        # each loop we have to
        params1 = checking_finished_parameters(prob_type, max_scenario_cnts)
        params2 = checking_working_parameters(prob_type, max_scenario_cnts)
        unfinished_params = params1.intersection(params2)

        print ("dispatch: {},  current #. of unfinished parameters: {}".format(
            prob_type, len(unfinished_params)))

        if len(unfinished_params) <= 100:
            for u_param in unfinished_params:
                print ("unfinished: {}".format(u_param))

        param = unfinished_params.pop()
        n_stock, win_length, n_scenario, bias, cnt, alpha, start_date, \
        end_date = param
        alpha = float(alpha)
        bias = True if bias == "biased" else False

        # log  parameter to file
        if not os.path.exists(log_path):
            working_dict = {}
        else:
            working_dict = retry_read_pickle(log_path)

        param_key = "|".join(str(v) for v in param)
        working_dict[param_key] = platform.node()
        retry_write_pickle(working_dict, log_path)

        # run experiment
        try:
            if prob_type == "min_ms_cvar_eventsp":
                run_min_ms_cvar_eventsp_simulation(
                    n_stock, win_length, n_scenario, bias, cnt, alpha,
                    start_date=start_date, end_date=end_date,
                    solver_io='lp')
        except Exception as e:
            print ("dispatch: run experiment:", param, e)
        finally:
            working_dict = retry_read_pickle(log_path)
            if param_key in working_dict.keys():
                del working_dict[param_key]
            else:
                print ("dispatch: can't find {} in working dict.".format(
                    param_key))
            retry_write_pickle(working_dict, log_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--prob_type", required=True, type=str)
    parser.add_argument("-c", "--max_scenario_cnt", required=True, type=int)
    args = parser.parse_args()
    dispatch_experiment_parameters(args.prob_type, args.max_scenario_cnt)
