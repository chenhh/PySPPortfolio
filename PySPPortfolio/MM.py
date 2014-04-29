# -*- coding: utf-8 -*-
'''
Created on 2014/4/29

@author: Hung-Hsin Chen
'''

from __future__ import division
from datetime import date
import os
import pandas as pd
import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt

PklBasicFeaturesDir = os.path.join(os.getcwd(),'pkl', 'BasicFeatures')
symbols = ('2330', '1216')
startDate, endDate = date(2005, 1, 1), date(2013, 12, 31)

def readDF(symbols):
    dfs = []
    for symbol in symbols:
        fin = os.path.join(PklBasicFeaturesDir, "%s.pkl"%(symbol))
        dfs.append(pd.read_pickle(fin))
    
    return dfs

def plotDistribution(dfs, today, histPeriods = 20):
    assert len(dfs) == 2
    fig = plt.figure()
    x_axis = plt.axis()
    y_axis = plt.axis()
    
    for df in dfs:
        ax = plt.axes()
    
    





if __name__ == '__main__':
    readDF(symbols[0])