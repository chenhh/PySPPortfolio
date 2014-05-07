# -*- coding: utf-8 -*-
'''
Created on 2013/10/22

@author: Hung-Hsin Chen
@email: chenhh@par.cse.nsysu.edu.tw
'''
from __future__ import division
import numpy as np


def Sharpe(series):
    '''ROI series
    the numpy std() function is the population estimator
    '''
    s = np.asarray(series)     
    try: 
        val =  s.mean()/s.std()
    except FloatingPointError:
        val = 0
    return val
    

def SortinoFull(series, MAR=0):
    '''ROI series
    MAR, minimum acceptable return
    '''
    s = np.asarray(series)
    mean = s.mean()
    semistd = np.sqrt( ((s*((s-MAR)<0))**2).mean() )
    try:
        val =  mean/semistd
    except FloatingPointError:
        val = 0
    return val


def SortinoPartial(series, MAR=0):
    '''ROI series
    MAR, minimum acceptable return
    '''
    s = np.asarray(series)
    mean = s.mean()
    neg_periods = (s-MAR) < 0
    n_neg_periods = neg_periods.sum()
    try:
        semistd = np.sqrt(((s * n_neg_periods)**2).sum()/n_neg_periods)
        val =  mean/semistd
    except FloatingPointError:
        val = 0
    return val


if __name__ == '__main__':
    pass