# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

M. Kaut，"A copula-based heuristic for scenario generation," Computational M
anagement Science, pp. 1-14, 2013.

the dimension of data is n_rv * n_dim
copula data structure:
    2D numpy.array, size n_rv * (n_dim +1)
    the first n_dim columns store the rank (ascending of each column) of each record
    the last column store the copula value of the record
'''

from __future__ import division
import numpy as np
import time
from coopr.pyomo import *
from coopr.opt import  SolverFactory
import matplotlib.pyplot as plt

def optimal2DCopulaSampling(data, n_scenario = 20):
    '''
    the optimal samples close to the empirical copula functions
    it can only to deal with bivariate samples
    
    @data, numpy.array, size: n_rv * 2
    '''
    assert data.shape[1] == 2
    n_rv = data.shape[0]
    
    tgt_copula = buildEmpiricalCopula(data)
    
    # Model
    model = ConcreteModel()
    
    #Set, dimension 1 and 2
    model.x = range(n_scenario)
    model.y = range(n_scenario)
    
    #decision variables
    model.X = Var(model.x, model.y, within=Binary)
    model.yp = Var(model.x, model.y, within=NonNegativeReals)
    model.yn = Var(model.x, model.y, within=NonNegativeReals)
    
    #constraint
    def rowConstraint_rule(model, x):
        '''to ensuring that each rank is used only once in each row'''
        val = sum( model.X[x, j] for j in model.y)
        return val == 1
        
    model.rowConstraint = Constraint(model.x)
    
    
    def columnConstraint_rule(model, y):
        '''to ensuring that each rank is used only once in each column'''
        val = sum( model.X[i, y] for i in model.x)
        return val == 1
      
    model.columnConstraint = Constraint(model.y)
    
    
    def copulaConstraint_rule(model, i, j):
        '''bias constraint '''
        val = 0
        for kdx in xrange(i):
            for ldx in xrange(j):
                val += model.X[kdx, ldx]
        val = val - model.yp[i, j] + model.yn[i, j]
        return val == n_rv *  getCopulaValue(tgt_copula, [i, j], n_scenario) 
            
    model.copulaConstraint = Constraint(model.x, model.y)
    
    #objective
    def minimizeBias_rule(model):
        '''minimize the bias between the sampling and given CDF'''
        val = 0
        for idx in model.x:
            for jdx in model.y:
                val += model.yp[idx, jdx] + model.yn[idx, jdx]
        return val
        
    model.minimizeBias = Objective()
    
    # Create a solver
#     solver = "glpk"
    solver= "cplex"
    opt = SolverFactory(solver)
#     opt.options["threads"]=4
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
#     display(instance)  
    
    
    results = {}
    
    for v in instance.active_components(Var):
        varobject = getattr(instance, v)
        if v == "X":
            results[v] = np.fromiter((varobject[idx, jdx].value 
                                    for jdx in np.arange(n_scenario) 
                                    for idx in np.arange(n_scenario)), np.float)
            results[v] = results[v].reshape((n_scenario, n_scenario))
            parsed = []
            for row in xrange(n_scenario):
                for col in xrange(n_scenario):
                    if results[v][row][col] >=0.9:
                        parsed.append((row, col))
            parsed = (np.asarray(parsed) + 1)/n_scenario
    
            results["parsed"] = parsed
            
        elif v == "yp":
            results[v] = np.fromiter((varobject[idx, jdx].value 
                                      for jdx in np.arange(n_scenario) 
                                      for idx in np.arange(n_scenario)), np.float)
            results[v] = results[v].reshape((n_scenario, n_scenario))
        elif v == "yn":
            results[v] = np.fromiter((varobject[idx, jdx].value 
                                      for jdx in np.arange(n_scenario) 
                                      for idx in np.arange(n_scenario)), np.float)
            results[v] = results[v].reshape((n_scenario, n_scenario))
    print results
    return results
    

def buildEmpiricalCopula(data):
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
        copula[:, col] = rankIdx + 1
   
    #computing empirical copula
    for idx in xrange(n_rv):
        for jdx in xrange(idx+1 ,n_rv):
            if np.all(copula[idx, :n_dim] >= copula[jdx, :n_dim]): 
                copula[idx, n_dim] += 1
            if np.all(copula[idx, :n_dim] <= copula[jdx, :n_dim]):
                copula[jdx, n_dim] += 1
                
    copula = copula.astype(np.float)/n_rv
#     print copula
#     print getCopulaValue(copula, [10, 20], 20)
    return copula
  


def getCopulaValue(copula, indices, maxVal):
    '''
    @copula, numpy.arrary, size: n_rv * (n_dim+1)
             the first n_dim columns are integer,
             and the last column is the copula value
    @probs, numpy.array, size: n_dim
    
    check how many copula points are dominated by the probs
    '''
    assert len(indices) == copula.shape[1] - 1    
    probs = np.asarray(indices, dtype=np.float)/maxVal

    n_rv, n_dim =copula.shape[0],  copula.shape[1] - 1
    
    dominating = sum(1 for row in xrange(n_rv) 
                     if np.all(probs >= copula[row, :n_dim]))   
       
    return float(dominating)/n_rv 

def testEmpiricalCopula():
    n_rv, n_dim = 5, 2
    data = np.random.rand(n_rv, n_dim)
    print "data:\n", data
    subplot = plt.subplot(2,1, 1)
    subplot.set_title('data')
    subplot.scatter(data[:, 0], data[:, 1])
    copula = buildEmpiricalCopula(data)
    print copula
    subplot = plt.subplot(2,1, 2)
    subplot.set_title('rank')
    subplot.scatter(copula[:, 0], copula[:, 1], color="green")
    plt.show()


def testOptimal2DCopulaSampling():
    n_rv, n_dim = 10, 2
    data = np.random.rand(n_rv, n_dim)
    t0 = time.time()
    copula = buildEmpiricalCopula(data)
    results = optimal2DCopulaSampling(data, n_scenario = 10)
    subplot = plt.subplot(3,1, 1)
    subplot.set_title('data')
    subplot.scatter(data[:, 0], data[:, 1])
    
    subplot = plt.subplot(3,1, 2)
    subplot.set_title('rank')
    subplot.scatter(copula[:, 0], copula[:, 1], color="pink")
    
    samples = results['parsed']
    subplot = plt.subplot(3,1, 3)
    subplot.set_title('samples')
    subplot.scatter(samples[:, 0], samples[:, 1], color="green")
    plt.show()
    print "elapsed %.3f secs"%(time.time()-t0)

if __name__ == '__main__':
#     testEmpiricalCopula()
    testOptimal2DCopulaSampling()
    