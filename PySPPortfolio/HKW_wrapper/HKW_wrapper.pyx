# -*- coding: utf-8 -*-
'''
Created on 2014/3/11
@author: Hung-Hsin Chen

'''
from __future__ import division
cimport HKW_wrapper
cimport numpy as np
import numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t

cpdef HeuristicMomentMatching(np.ndarray tgtMoms,
                             np.ndarray tgtCorrs,
                             int n_scenario,
                             int MaxTrial,
                             int HKW_MaxIter,
                             double MaxErrMom,
                             double MaxErrCorr
                             ):
    cdef:
        int i, j
        int n_rv = tgtMoms.shape[0]
        int FormatOfTgMoms = 1
#         int MaxTrial = 50
#         int HKW_MaxIter = 50
#         double MaxErrMom = 1e-3
#         double MaxErrCorr = 1e-3
        int TestLevel = 2
        unsigned char UseStartDistrib = 0
        TMatrix p_TarMoms
        TMatrix p_TgCorrs
        TVector p_Probs
        TMatrix OutMat 
       
    p_TarMoms = TMatrix(0, 0, NULL) 
    p_TgCorrs = TMatrix(0, 0, NULL) 
    p_Probs = TVector(0, NULL)
    OutMat = TMatrix(0, 0, NULL) 

    Mat_Init(&p_TarMoms, 4, n_rv)
    Mat_Init(&p_TgCorrs, n_rv, n_rv)
    Vec_Init(&p_Probs, n_scenario)
    Mat_Init(&OutMat, n_rv, n_scenario)
    
    for i in xrange(n_rv):
        for j in xrange(4):
            p_TarMoms.val[j][i] = tgtMoms[i][j]

    for i in xrange(n_rv):
        for j in xrange(n_rv):
            p_TgCorrs.val[i][j] = tgtCorrs[i][j]

    for i in xrange(n_scenario):
        p_Probs.val[i] = 1./n_scenario

    Mat_Display(&p_TarMoms, "p_tarmoms")
    Mat_Display(&p_TgCorrs, "p_tgcorrs")
    #Vec_Display(&p_Probs, "p_Probs")

    code = HKW_ScenGen(FormatOfTgMoms, &p_TarMoms, &p_TgCorrs, &p_Probs, &OutMat,
                MaxErrMom, MaxErrCorr, TestLevel, MaxTrial, HKW_MaxIter,
                UseStartDistrib, NULL, NULL, NULL, NULL
            )
        
    #Mat_Display(&OutMat, "OutMat")
    
    scenarios = np.empty((n_rv, n_scenario))
    for i in xrange(n_rv):
        for j in xrange(n_scenario):
            scenarios[i][j] = OutMat.val[i][j]
    
    #De-allocate matrices
    Mat_Kill(&p_TarMoms)
    Mat_Kill(&p_TgCorrs)
    Mat_Kill(&OutMat)
    Vec_Kill(&p_Probs)
    
      
    return scenarios, code
            
