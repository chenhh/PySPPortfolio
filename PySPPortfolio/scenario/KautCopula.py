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

def optimal2DCopulaSampling(data, n_scenario = 20, solver="cplex"):
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
        '''to ensure that each rank is used only once in each row'''
        val = sum( model.X[x, j] for j in model.y)
        return val == 1
        
    model.rowConstraint = Constraint(model.x)
    
    
    def columnConstraint_rule(model, y):
        '''to ensure that each rank is used only once in each column'''
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
        
        point = [(i+1.)/n_scenario, (j+1.)/n_scenario]
        copula_val = getCopula(tgt_copula,  point)
        print "point %s copula:%s, S*copula:%s"%(point, copula_val, n_scenario * copula_val )
        return val == n_scenario * copula_val 
    
            
    model.copulaConstraint = Constraint(model.x, model.y)
    
    #objective
    def minBias_rule(model):
        '''minimize the bias between the sampling and given CDF'''
        val = 0
        for idx in model.x:
            for jdx in model.y:
                val += (model.yp[idx, jdx] + model.yn[idx, jdx])
        return val
        
    model.minBias = Objective(sense=minimize)
    
    # Create a solver
    solver= solver
    opt = SolverFactory(solver)
    
    if solver =="cplex":
        opt.options["threads"] = 4
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    display(instance)  
    
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
#     print results
    return results
    

def buildEmpiricalCopula(data):
    '''
    empirical cumulative distribution function
    @data, numpy.array, size: n_rv * n_dim (N*D)
    time complexity: O(N**2 * D )
    
    given data, build the empirical copula of the data
    '''
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
            #idx dominating jdx 
            if np.all(copula[idx, :n_dim] >= copula[jdx, :n_dim]): 
                copula[idx, n_dim] += 1
            #jdx dominating idx
            if np.all(copula[idx, :n_dim] <= copula[jdx, :n_dim]):
                copula[jdx, n_dim] += 1
                
    copula = copula.astype(np.float)/n_rv
    return copula
  

def getCopula(copula, point):
    '''
    @copula, numpy.arrary, size: n_rv * (n_dim + 1)
             the first n_dim columns are integer,
             and the last column is the copula value
    @point, numpy.array, size: n_dim
    
    check how many copula points are dominated by the point
    '''
    assert len(point) == copula.shape[1] - 1
    point = np.asarray(point, dtype=np.float)
    assert np.all( 0<=point) and np.all(point <=1)    

    n_rv, n_dim =copula.shape[0],  copula.shape[1] - 1
    dominating = sum(1. for row in xrange(n_rv) 
                     if np.all(point >= copula[row, :n_dim]))   
       
    return dominating/n_rv 


def testEmpiricalCopula():
    n_rv, n_dim = 5, 2
    data = np.random.rand(n_rv, n_dim)
    print "data:\n", data
    subplot = plt.subplot(2,1, 1)
    subplot.set_title('data')
    subplot.scatter(data[:, 0], data[:, 1])
    copula = buildEmpiricalCopula(data)
    print "copula:\n", copula
    subplot = plt.subplot(2,1, 2)
    subplot.set_title('rank')
    subplot.scatter(copula[:, 0], copula[:, 1], color="green")
    plt.show()


def testOptimal2DCopulaSampling():
    n_rv, n_dim = 4, 2
    data = np.random.randn(n_rv, n_dim)
    print "data:\n", data
    t0 = time.time()
    
    copula = buildEmpiricalCopula(data) 
    print "empirical copula:\n", copula
   
    results = optimal2DCopulaSampling(data, n_scenario = n_rv*2)
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
    print "samples:\n", samples
#     sample_copula = buildEmpiricalCopula(samples)
    sample_copula = [getCopula(copula, pt) for pt in samples]
    print "sample_copula:\n", sample_copula 
    
    print "objective:", results['yp'].sum() + results['yn'].sum() 
    print "X:\n", results['X']
    print "yp+yn:\n", results['yp'] + results['yn']
    
    print "elapsed %.3f secs"%(time.time()-t0)
    
#     plot3DCopula(copula)
    
    plt.show()
    

def plot3DCopula(copula):
    n_rv = copula.shape[0]
    step = 1./n_rv
    x = np.arange(0, 1+step, step)
    y = np.arange(0, 1+step, step)
    x, y = np.meshgrid(x, y)
    print x


def RCopula():
    import rpy2.robjects as ro
    from rpy2.robjects.numpy2ri import numpy2ri
    from rpy2.robjects.packages import importr
    copula = importr('copula')
    
    n_rv, n_dim = 6, 2
    data = np.random.rand(n_rv, n_dim)
    data2 = np.random.rand(n_rv/2, n_dim)
    print "data:\n", data
    print "data2:\n", data2

    print copula.C_n(numpy2ri(data), numpy2ri(data2))
    mycopula = buildEmpiricalCopula(data)
    print mycopula 

if __name__ == '__main__':
#     testEmpiricalCopula()
    testOptimal2DCopulaSampling()
#     RCopula()
    