# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

M. Kaut，"A copula-based heuristic for scenario generation," Computational M
anagement Science, pp. 1-14, 2013.
'''

from __future__ import division
import numpy as np
import time
from coopr.pyomo import *


def optimal2DCopulaSampling(data, n_scenario = 20):
    '''
    the optimal samples close to the empirical copula functions
    it can only to deal with bivariate samples
    
    @data, numpy.array, size: n_rv * 2
    '''
    assert data.shape[1] == 2
    n_rv = data.shape[0]
    
    tgt_copula = empiricalCopulaCDF(data)
    
    # Model
    model = ConcreteModel()
    
    #Set, dimension 1 and 2
    model.x = range(n_rv)
    model.y = range(n_rv)
    
    #decision variables
    model.X = Var(model.x, model.y, within=Binary)
    model.yp = Var(model.x, model.y, within=NonNegativeReals)
    model.yn = Var(model.x, model.y, within=NonNegativeReals)
    
    #constraint
    def rowConstraint_rule(model, x):
        '''to ensuring that each rank is used only once in each row'''
        val = sum( model.X[x][j] for j in model.y)
        return val == 1
        
    model.rowConstraint = Constraint(model.x)
    
    
    def columnConstraint_rule(model, y):
        '''to ensuring that each rank is used only once in each column'''
        val = sum( model.X[i][y] for i in model.x)
        return val == 1
      
    model.columnConstraint = Constraint(model.y)
    
    
    def copulaConstraint_rule(model, i, j):
        '''approximate constraint '''
        val = 0
        for kdx in xrange(i):
            for ldx in xrange(j):
                val += model.X[kdx][ldx]
        val = val - model.yp[i, j] + model.yn[i, j]
        return val == n_rv *  tgt_copula[i, j] 
            
    model.copulaConstraint = Constraint(model.x, model.y)
    
    #objective
    def minimizeBias_rule(model):
        '''minimize the bias between the sampling and given CDF'''
        val = 0
        for idx in model.x:
            for jdx in model.y:
                val += model.yp[idx][jdx] + model.yn[idx][jdx]
        return val
        
    model.minimizeBias = Objective()
    
    
    # Create a solver
    solver = "cplex"
    opt = SolverFactory(solver)
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    display(instance)
    

def empiricalCopula(data):
    '''
    empirical cumulative distribution function
    @data, numpy.array, size: n_rv * n_dim (N*D)
    time complexity: O(N**2 * D )
    '''
    #dominating sort
    n_rv, n_dim = data.shape
    
    #computing copula indices 
    #[:, :n_dim]為rank index, [:, n_dim]為dominating values 
    copula = np.ones((n_rv, n_dim +1))
    rankIdx = np.empty(n_rv)
    for col in xrange(n_dim):
        rankIdx[data[:, col].argsort()] = np.arange(n_rv)
        copula[:, col] = rankIdx
   
    #computing empirical copula
    for idx in xrange(n_rv):
        for jdx in xrange(idx+1 ,n_rv):
            if np.all(copula[idx, :n_dim] >= copula[jdx, :n_dim]): 
                copula[idx, n_dim] += 1
    copula[:, n_dim] = copula[:, n_dim].astype(np.float)/n_rv 
    print copula
  


def getCopulaValue(copula, indices):
    '''
    @copula, numpy.arrary, size: n_rv * (n_dim+1)
             the first n_dim columns are integer,
             and the last column is the copula value
    @indices, numpy.array, size: n_dim
    '''
    indices = np.asarray(indices)
    n_dim = copula.shape[1] - 1
    row, _ = np.where(copula[:, :n_dim] == indices)
    mask = np.ones(n_dim)*row[0]
    if row.size != indices.size or not (row == mask).all() :
        #1. some indices are not in the copula
        #2. not all indices are in the same row
        return 0
    else:
        return copula[row[0], n_dim]

def testEmpiricalCopula():
    n_rv, n_dim = 5, 2
    data = np.random.rand(n_rv, n_dim)
    print "data:\n", data
    empiricalCopula(data)


if __name__ == '__main__':
    testEmpiricalCopula()