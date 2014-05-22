# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

kelly multivariate investment
'''

from __future__ import division
from coopr.pyomo import *
from time import time
from datetime import date
import numpy as np
import pandas as pd
import os
import time
from coopr.opt import  SolverFactory


def KellyCriterion(symbols, riskyRet, money=1e6, solver="cplex"):
    '''
    @riskyRet, shape: M*T
    
    maximize W*R - 1/2W^T \simga W
    '''
    t = time.time()
    
    mu = riskyRet.mean(axis=1)
    
    model = ConcreteModel()
    
    #Set
    model.symbols = range(len(symbols))
       
    #decision variables
    model.W = Var(model.symbols, within=NonNegativeReals)
    
    #constraint
    def CapitalConstraint_rule(model):
        allocation = sum(model.W[idx] for idx in model.symbols)
        return allocation == money
    
    model.CapitalConstraint = Constraint()
    
    
    #objective
    def KellyObjective_rule(model):
        profit = sum(model.W[idx]*mu[idx] for idx in model.symbols)
        risk = 0
        for idx in model.symbols:
            for jdx in model.symbols:
                    risk += model.W[idx] * model.W[jdx] * mu[idx]* mu[jdx] 
        
        return profit - 1./2 * risk
        
    model.KellyObjective = Objective(sense=maximize)
    
    # Create a solver
    opt = SolverFactory(solver)
    
    if solver =="cplex":
        opt.options["threads"] = 4
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    obj = results.Solution.Objective.__default_objective__['value']
    display(instance)
    
    print "Kelly elapsed %.3f secs"%(time.time()-t)

def testKelly():
    FileDir = os.path.abspath(os.path.curdir)
    PklBasicFeaturesDir = os.path.join(FileDir, '..', 'pkl', 'BasicFeatures')
    
    symbols = ['2330', '2317', '6505']
    n_period = 100
    ROIs = np.empty((len(symbols), n_period))
    for idx, symbol in enumerate(symbols):
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        roi =  df['adjROI'][:n_period]
        ROIs[idx] = roi
        
    KellyCriterion(symbols, ROIs, money=1e6, solver="cplex")

if __name__ == '__main__':
    testKelly()