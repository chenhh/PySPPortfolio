# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
import pandas as pd
import os
import numpy as np
from PySPPortfolio.pysp_portfolio import *


def roi_stats(n_stock, win_length, n_scenario=200, bias="unbiased"):


    for cnt in xrange(1, 4):
        scenario_name = "{}_{}_m{}_w{}_s{}_{}_{}.pkl".format(
        START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d"),
            n_stock, win_length, n_scenario, bias, cnt)
        scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios',
                                 scenario_name)

        if not os.path.exists(scenario_path):
            print ("{} not exists.".format(scenario_name))
            continue

        panel = pd.read_pickle(scenario_path)

        # verify all zeros at a specific period



    # df = pd.read_pickle(scenario_path)
    # print df

if __name__ == '__main__':
    roi_stats(5, 50)