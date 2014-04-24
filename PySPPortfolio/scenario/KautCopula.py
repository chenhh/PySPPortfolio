# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

M. Kaut，"A copula-based heuristic for scenario generation," Computational M
anagement Science, pp. 1-14, 2013.
'''

from __future__ import division
import numpy as np
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
    

def empirical2DCopulaCDF(data):
    '''
    empirical cumulative distribution function
    @data, numpy.array, size: n_rv * n_dim (N*D)
    time complexity: O(N**2 * D )
    '''
    #dominating sort
    n_rv, n_dim = data.shape
    assert n_dim == 2
    
    #computing copula indices 
        
#     copula = np.zeros((n_rv, n_rv))
#     for idx in xrange(n_rv):
#         dominated = 1
#         for jdx in xrange(idx+1, n_rv):
#             if np.all(data[idx] <= data[jdx]):
#                 dominated += 1
#         copula[idx] = dominated    
#     
#     copula = copula.astype(np.float)/n_rv 
    print copula
    return copula


def testEmpiricalCopulaCDF():
    n_rv, n_dim = 10, 2
    data = np.random.rand(n_rv, n_dim)
    print "data:\n", data
    empirical2DCopulaCDF(data)

if __name__ == '__main__':
    testEmpiricalCopulaCDF()