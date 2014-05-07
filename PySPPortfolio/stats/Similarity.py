# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

'''

import numpy as np
import time
import pandas as pd
import scipy.stats as spstats
import scipy.spatial.distance as spdist
import scipy.stats as spstats
from datetime import date
import os
import sys
import sklearn.preprocessing as skprep


def DTW(S1, S2, window=5):
    '''
    dynamic time warping with constraint window size
    the larger distance, the less similar between the series
    @param S1: numpy.array
    @param S2: numpy.array
    @param window: positive integer
    
    @return DTW distance
    '''
    S1, S2 = np.asarray(S1), np.asarray(S2)
    window = max(window, abs(S1.size-S2.size))
    mtx = np.ones((S1.size+1, S2.size+1)) * np.inf
    mtx[0, 0] = 0.
   
    for idx in xrange(S1.size):
        low, high = max(0, idx-window), min(S2.size, idx+window+1)
        for jdx in xrange(low, high):
            cost = abs(S1[idx] - S2[jdx])
            mtx[idx+1, jdx+1] = cost + min(mtx[idx, jdx+1], 
                                           mtx[idx+1, jdx],
                                           mtx[idx, jdx]    #match
                                           )
    return mtx[idx, jdx]
    

def correlation(S1, S2):
    '''
    the larger value, the more similar between the series
    Pearson, Spearman, Kendall correlation coefficient, 
    @param S1: numpy.array
    @param S2: numpy.array
    @return -1 <= coefficient <=1
    '''
    S1, S2 = np.asarray(S1), np.asarray(S2)
    assert S1.size == S2.size
    
    funcs = (spstats.pearsonr, spstats.spearmanr, spstats.kendalltau)
    corrs = []
    for func in funcs:
        try:
            corr, _ = func(S1, S2)
        except FloatingPointError:
            corr = None
        corrs.append(corr)
        
    return corrs


def distance_correlation(S1, S2):
    '''
    http://en.wikipedia.org/wiki/Distance_correlation
    http://cran.r-project.org/web/packages/energy/energy.pdf
    
    the same results as R package energy: dcor
    @param S1: numpy.array 
    @param S2: numpy.array
    
    if dimension of S1 or S2 >=2, 
    then row: data, column: feature
    
    @return 0<= coefficient <= 1
    '''
    S1, S2 = np.asarray(S1), np.asarray(S2)
    
    if not S1.shape[:-1]:
        #1D array
        S1 = S1[:, np.newaxis]
        
    if not S2.shape[:-1]:
        #1D array
        S2 = S2[:, np.newaxis] 
    
    assert S1.shape[0] == S2.shape[0]
    
    #pairwise distance
    a = spdist.squareform(spdist.pdist(S1))
    a_row_mean = a.mean(axis=0)
    a_mean = a.mean()
  
    b = spdist.squareform(spdist.pdist(S2))
    b_row_mean = b.mean(axis=0)
    b_mean = b.mean()
    
    #double centered distance
    A = a - a_row_mean - a_row_mean[:, np.newaxis] + a_mean
    B = b - b_row_mean - b_row_mean[:, np.newaxis] + b_mean
    
    #squared sample distance covariance and variances
    sq_dcov, sq_dvar1, sq_dvar2 = (A*B).mean(), (A*A).mean(), (B*B).mean()
    if sq_dvar1 == 0 or sq_dvar2 == 0:
        dcor= 0
        dcov, dvar1, dvar2  = np.sqrt((sq_dcov, sq_dvar1, sq_dvar2))
    else:
        sq_dcor = sq_dcov/np.sqrt(sq_dvar1*sq_dvar2)
        dcor, dcov, dvar1, dvar2 = np.sqrt((sq_dcor, sq_dcov, sq_dvar1, sq_dvar2))
    
    return {"dcor":dcor, "dcov": dcov, "dvar1": dvar1, "dvar2": dvar2}


def dcor_test(S1, S2):
    '''
    http://en.wikipedia.org/wiki/Distance_correlation
    http://cran.r-project.org/web/packages/energy/energy.pdf
    
    the same results as R package energy: dcor
    @param S1: numpy.array 
    @param S2: numpy.array
    
    if dimension of S1 or S2 >=2, 
    then row: data, column: feature
    
    @return coefficient
    '''
    S1, S2 = np.asarray(S1), np.asarray(S2)
    
    if S1.ndim == 1:
        #1D array
        S1 = S1[:, np.newaxis]
        
    if S2.ndim == 1:
        #1D array
        S2 = S2[:, np.newaxis] 
    
    assert S1.shape[0] == S2.shape[0]
    n_samples = S1.shape[0] 
    
    #pairwise distance (a_col_mean == a_row_mean)
    a = spdist.squareform(spdist.pdist(S1))
    a_row_mean = a.mean(axis=0)
    a_mean = a.mean()
  
    b = spdist.squareform(spdist.pdist(S2))
    b_row_mean = b.mean(axis=0)
    b_mean = b.mean()
    
    #double centered distance
    A = a - a_row_mean - a_row_mean[:, np.newaxis] + a_mean
    B = b - b_row_mean - b_row_mean[:, np.newaxis] + b_mean
    
    #squared sample distance covariance and variances
    sq_dcov, sq_dvar1, sq_dvar2 = np.mean(A*B), np.mean(A*A), np.mean(B*B)
    if sq_dvar1 == 0 or sq_dvar2 == 0:
        dcor = 0
    else:
        sq_dcor = sq_dcov/np.sqrt(sq_dvar1*sq_dvar2)
        dcor = np.sqrt(sq_dcor)
    
    #modified A, B
    if n_samples >= 4:
        c = n_samples/(n_samples-1.)
        
        Astar = c * (A - a/n_samples  )
        Astar.ravel()[::n_samples+1] = c * (a_row_mean - a_mean)
        Adiag = Astar.ravel()[::n_samples+1]
        
        Bstar = c * (B - b/n_samples  )
        Bstar.ravel()[::n_samples+1] = c * (b_row_mean - b_mean)
        Bdiag = Bstar.ravel()[::n_samples+1]
        
        c2 = n_samples/(n_samples-2.)
        c3 = 1./(n_samples * (n_samples-3.))
        V_xy = c3 * ((Astar*Bstar).sum() - c2 * (Adiag * Bdiag).sum())
        V_xx = c3 * ((Astar*Astar).sum() - c2 * (Adiag * Adiag).sum())
        V_yy = c3 * ((Bstar*Bstar).sum() - c2 * (Bdiag * Bdiag).sum())
        
        V_xxyy = V_xx *V_yy
        if V_xxyy > 0:
            m_dcor = V_xy/np.sqrt(V_xxyy)
        else:
            m_dcor = 0
        
        df=n_samples * (n_samples-3)/2 - 1
        try:
            t_stats = np.sqrt(df)*m_dcor/np.sqrt(1.-m_dcor*m_dcor)
            pval = 1.-spstats.t.cdf(t_stats, df=df)
        except FloatingPointError:
            #if m_dcor == 1 or -1
            t_stats = None
            pval = 0.
            
    else:
        m_dcor, df, t_stats, pval = None, 0, None, None
       
    return {"dcor": dcor, "modified_dcor":m_dcor, 
            "pvalue": pval, "t_stats":t_stats, "df": df} 
