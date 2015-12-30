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
from exp_cvar import (run_min_cvar_sip_simulation, run_min_cvar_sp_simulation,
                  run_min_cvar_eev_simulation, run_min_ms_cvar_sp_simulation)

def get_results_dir(prob_type):
    """
    Parameter:
    ----------------
    prob_type: str, {min_cvar_sp, min_cvar_sip}
    """
    if prob_type in ("min_cvar_sp", "min_cvar_sip", "min_cvar_eev",
                     "min_ms_cvar_sp", "min_cvar_eev_objective"):
        return os.path.join(EXP_SP_PORTFOLIO_DIR, prob_type)
    else:
        raise ValueError("unknown prob_type: {}".format(prob_type))



def all_experiment_parameters(prob_type, max_scenario_cnts):
    """
    file_name of all experiment parameters
    n_stock: {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
    win_length: {50, 60, ..., 240}
    n_scenario:  200
    biased: {unbiased,}
    cnt: {1,2,3}
    alpha: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

    """
    # combinations: 10 * 20 * 3 * 10 = 6000
    all_params = []
    for n_stock in xrange(5, 50 + 5, 5):
        for win_length in xrange(50, 240 + 10, 10):
            if prob_type in ( "min_cvar_sp", 'min_cvar_eev', "min_ms_cvar_sp",
                              "min_cvar_eev_objective"):
                if n_stock == 50 and win_length == 50:
                    # preclude m50_w50
                    continue
            elif prob_type == "min_cvar_sip":
                # because the candidate set is 50, therefore all
                # experiments of windows length = 50 is excluded.
                if win_length == 50:
                    continue

            for n_scenario in (200,):
                for bias in ("unbiased",):
                    for cnt in xrange(1, max_scenario_cnts+1):
                        if prob_type in ("min_cvar_eev_objective",):
                            all_params.append(n_stock, win_length, n_scenario,
                                              bias, cnt)
                        else:
                            for alpha in ('0.50', '0.55', '0.60', '0.65',
                                          '0.70', '0.75', '0.80', '0.85',
                                          '0.90', '0.95'):
                                all_params.append(
                                   (n_stock, win_length,
                                    n_scenario,bias, cnt, alpha))

    return set(all_params)



def special_experiment_parameters(prob_type, max_scenario_cnts):
    """
    file_name of all experiment parameters
    n_stock: {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
    win_length: {50, 60, ..., 240}
    n_scenario:  200
    biased: {unbiased,}
    cnt: {1,2,3}
    alpha: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

    """
    # combinations: 10 * 20 * 3 * 10 = 6000
    all_params = []
    if prob_type == "min_cvar_eev":
        all_params.append((5, 190, 200, "unbiased", 1, 0.85))
        all_params.append((10, 110, 200, "unbiased", 3, 0.6))
        all_params.append((15, 100, 200, "unbiased", 5, 0.65))
        all_params.append((20, 190, 200, "unbiased", 1, 0.85))

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

    if prob_type in ("min_cvar_sp", "min_cvar_eev", "min_ms_cvar_sp",
                     "min_cvar_eev_objective",):
        pkls = glob.glob(os.path.join(dir_path,
                    "{}_20050103_20141231_*.pkl".format(prob_type)))
    elif prob_type == "min_cvar_sip":
        pkls = glob.glob(os.path.join(dir_path,
                    "{}_20050103_20141231_all50_*.pkl".format(prob_type)))

    for pkl in pkls:
        name = pkl[pkl.rfind(os.sep)+1: pkl.rfind('.')]
        exp_params = name.split('_')
        if prob_type in ("min_cvar_sp", "min_cvar_eev"):
            params = exp_params[5:]
        elif prob_type in ("min_cvar_sip", "min_ms_cvar_sp",
                           "min_cvar_eev_objective",):
            params = exp_params[6:]
        n_stock = int(params[0][params[0].rfind('m')+1:])
        win_length = int(params[1][params[1].rfind('w')+1:])
        n_scenario = int(params[2][params[2].rfind('s')+1:])
        bias = params[3]
        scenario_cnt = int(params[4])
        alpha = params[5][params[5].rfind('a')+1:]

        data_param = (n_stock, win_length, n_scenario, bias, scenario_cnt,
                      alpha)

        if data_param in all_params:
            all_params.remove(data_param)

    # return unfinished params
    return all_params


def checking_working_parameters(prob_type, max_scenario_cnts):
    """
    if a parameter is under working and not write to pkl,
    it is recorded to a file

    working dict:
        - key: str,  "|".join(str(v) for v in param)
        - value: machine node
    """
    dir_path = get_results_dir(prob_type)
    log_file = '{}_working.pkl'.format(prob_type)
    retry_cnt = 5
    # get all params
    all_params = all_experiment_parameters(prob_type, max_scenario_cnts)

    # storing a dict, key: param, value: platform_name
    file_path = os.path.join(dir_path, log_file)
    if not os.path.exists(file_path):
        # no working parameters
        print ("checking_working: {} no working parameters".format(prob_type))
        return all_params

    data = retry_read_pickle(file_path)
    for param_key, node in data.items():
        # key:  n_stock, win_length, _scenario, biased, cnt, alpha
        keys =  param_key.split('|')
        param = (int(keys[0]), int(keys[1]), int(keys[2]), keys[3],
                 int(keys[4]), keys[5])
        if param in all_params:
            all_params.remove(param)
            print ("checking_working {}: {} under processing on {}.".format(
                prob_type, param, node))
    print ("checking_working params: {}".format(len(data)))

    # unfinished params
    return all_params

def retry_read_pickle(file_path, retry_cnt=5):
    for retry in xrange(retry_cnt):
        try:
            working_dict = pd.read_pickle(file_path)
            return working_dict

        except IOError as e:
            if retry == retry_cnt -1:
                raise Exception(e)
            else:
                print ("dispatch: reading retry: {}, {}".format(
                    retry+1, e))
                time.sleep(np.random.rand() * 5)


def retry_write_pickle(data, file_path, retry_cnt=5):
    for retry in xrange(retry_cnt):
        try:
            pd.to_pickle(data, file_path)
        except IOError as e:
            if retry == retry_cnt -1:
                raise Exception(e)
            else:
                print ("dispatch: reading retry: {}, {}".format(
                    retry+1, e))
                time.sleep(np.random.rand() * 5)

def dispatch_experiment_parameters(prob_type, max_scenario_cnts):

    dir_path = get_results_dir(prob_type)

    # storing a dict, {key: param, value: platform_name}
    log_file = '{}_working.pkl'.format(prob_type)

    # reading working pkl
    log_path = os.path.join(dir_path, log_file)

    params1 = checking_finished_parameters(prob_type, max_scenario_cnts)
    params2 = checking_working_parameters(prob_type, max_scenario_cnts)
    unfinished_params = params1.intersection(params2)

    print ("dispatch: {} initial unfinished params: {}".format(
        prob_type, len(unfinished_params)))

    while len(unfinished_params) > 0:
        # each loop we have to
        params1 = checking_finished_parameters(prob_type, max_scenario_cnts)
        params2 = checking_working_parameters(prob_type, max_scenario_cnts)
        unfinished_params = params1.intersection(params2)

        print ("dispatch: {} current unfinished params: {}".format(
            prob_type, len(unfinished_params)))

        if len(unfinished_params) <= 100:
            for u_param in unfinished_params:
                print ("unfinished: {}".format(u_param))

        param = unfinished_params.pop()
        if prob_type not in ("min_cvar_eev_objective",):
            n_stock, win_length, n_scenario, bias, cnt, alpha = param
            alpha = float(alpha)
        else:
            n_stock, win_length, n_scenario, bias, cnt = param

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
            if prob_type == 'min_cvar_sp':
                run_min_cvar_sp_simulation(n_stock, win_length, n_scenario,
                               bias, cnt, alpha)
            elif prob_type == "min_cvar_sip":
                run_min_cvar_sip_simulation(n_stock, win_length,
                                n_scenario, bias, cnt, alpha)
            elif prob_type == "min_cvar_eev":
                run_min_cvar_eev_simulation(n_stock, win_length, n_scenario,
                               bias, cnt, alpha, eev_objectve=False)
            elif prob_type == "min_cvar_eev_objective":
                run_min_cvar_eev_simulation(n_stock, win_length, n_scenario,
                               bias, cnt, 0.5, eev_objectve=True)
            elif prob_type == "min_ms_cvar_sp":
                # after run one alpha, it will runs all alphas
                # and write it to corresponding files, and the cehcking
                # finished_parameter will delete all related alphas
                run_min_ms_cvar_sp_simulation(n_stock, win_length, n_scenario,
                               bias, cnt, alphas=[0.5, 0.55, 0.6, 0.65, 0.7,
                                                  0.75, 0.8, 0.85, 0.9, 0.95])
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