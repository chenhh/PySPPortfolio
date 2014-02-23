# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
import time
import numpy as np
from coopr.pyomo import (Set, RangeSet, Param, Var, Objective, Constraint,
                         ConcreteModel, Reals, NonNegativeReals, maximize)
from coopr.opt import  SolverFactory


def optimalMultiStagePortfolio(riskyRetMtx, riskFreeRetVec, 
                               buyTransFeeMtx, sellTransFeeMtx,
                               allocatedWealthVec):
    '''
    -假設資料共T期, 投資在M個risky assets, 以及1個riskfree asset
    -求出每個risky asset每一期應該要買進以及賣出的金額
    -最後一期結算不投資
    
    @param riskyRetMtx, numpy.array, size: M * T+1
    @param riskFreeRetVec, numpy.array, size: T+1
    @param buyTransFeeMtx, numpy.array, size: M * T
    @param sellTransFeeMtx, numpy.array, size: M * T
    @param allocatedWealthVec, numpy.array, size: (M+1) (最後一個為cash)
    
    @return (buyMtx, sellMtx), numpy.array, each size: M*T
    '''
    assert buyTransFeeMtx.shape == sellTransFeeMtx.shape
    assert riskyRetMtx.shape[1] == riskFreeRetVec.size
   
    M, T =  buyTransFeeMtx.shape

    t1 = time.time()
    #create model
    model = ConcreteModel()
    
    #number of asset and number of periods
    model.M = Set(initialize=range(M))
    model.T = Set(initialize=range(T))
    model.Tp1 = Set(initialize=range(T+1))
    model.Mp1 = Set(initialize=range(M+1))
    
    #parameter
    def riskyRet_init(model, m, t):
        return riskyRetMtx[m, t]
    
    def riskFreeRet_init(model, t):
        return riskFreeRetVec[t]
    
    def buyTransFee_init(model, m, t):
        return buyTransFeeMtx[m, t]
    
    def sellTransFee_init(model, m, t):
        return sellTransFeeMtx[m, t]
    
    def allocatedWealth_init(model, m):
        return allocatedWealthVec[m]
    
    model.riskyRet = Param(model.M, model.Tp1, initialize=riskyRet_init)
    model.riskFreeRet = Param(model.Tp1,  initialize=riskFreeRet_init)
    model.buyTransFee = Param(model.M, model.T, initialize=buyTransFee_init)
    model.sellTransFee = Param(model.M, model.T, initialize=sellTransFee_init)
    model.allocatedWealth = Param(model.Mp1, initialize=allocatedWealth_init)
    
    #decision variables
    model.buys = Var(model.M, model.T, within=NonNegativeReals)
    model.sells = Var(model.M, model.T, within=NonNegativeReals)
    model.riskyWealth = Var(model.M, model.T, within=NonNegativeReals)
    model.riskFreeWealth = Var(model.T, within=NonNegativeReals)
    
    #objective
    def objective_rule(model):
        wealth =sum( (1. + model.riskyRet[m, T]) * model.riskyWealth[m, T-1] 
                     for m in xrange(M))
        wealth += (1.+ model.riskFreeRet[T]) * model.riskFreeWealth[T-1] 
        return wealth
        
    model.objective = Objective(rule=objective_rule, sense=maximize)
    
    #constraint
    def riskyWealth_constraint_rule(model, m, t):
        if t>=1:
            preWealth = model.riskyWealth[m, t-1]
        else:
            preWealth = model.allocatedWealth[m]
       
        return (model.riskyWealth[m, t] == 
                (1. + model.riskyRet[m,t])*preWealth + 
                model.buys[m,t] - model.sells[m,t])
    
    def riskyfreeWealth_constraint_rule(model, t):
        totalSell = sum((1-model.sellTransFee[mdx, t])*model.sells[mdx, t] 
                        for mdx in xrange(M))
        totalBuy = sum((1+model.buyTransFee[mdx, t])*model.buys[mdx, t] 
                       for mdx in xrange(M))
        
        if t >=1:
            preWealth = model.riskFreeWealth[t-1]  
        else:
            preWealth = model.allocatedWealth[M]
    
        return( model.riskFreeWealth[t] == 
                (1. + model.riskFreeRet[t])*preWealth + 
                totalSell - totalBuy)
        
    
    model.riskyWeathConstraint = Constraint(model.M, model.T, 
                                            rule=riskyWealth_constraint_rule)
    model.riskfreeConstraint = Constraint(model.T, 
                                          rule=riskyfreeWealth_constraint_rule)
    
    #optimizer
    opt = SolverFactory('cplex')
    opt.options["threads"] = 4
    
    instance = model.create()
    results = opt.solve(instance)
    print results
    
    instance.load(results)
    for var in instance.active_components(Var):
        varobj = getattr(instance, var)
        for idx in varobj:
            print varobj[idx], varobj[idx].value
   
    print "elapsed %.3f secs"%(time.time()-t1)
    

def testMSPortfolio():
    
    M, T = 100, 100
    Tp1 = T+1
    
    riskyRetMtx = np.random.rand(M, Tp1)/100.0
#     print "risky ret:", riskyRetMtx
    riskFreeRetMtx = np.zeros(Tp1)
#     print "riskfree ret:", riskFreeRetMtx 
    buyTransFeeMtx = np.ones((M,T))* 0.003
    sellTransFeeMtx = np.ones((M,T))* 0.004425
    allocated = np.zeros(M)
    allocated[-1] = 1e5
    optimalMultiStagePortfolio(riskyRetMtx, riskFreeRetMtx, 
                               buyTransFeeMtx, sellTransFeeMtx, allocated)
    
if __name__ == '__main__':
    testMSPortfolio()