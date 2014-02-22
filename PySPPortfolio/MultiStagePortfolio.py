# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
import time
from coopr.pyomo import (Set, RangeSet, Param, Var, Objective, Constraint,
                         ConcreteModel, Reals, NonNegativeReals)
fromt coopr.opt import  SolverFactory

def optimalMultiStagePortfolio(symbols, riskyRetMtx, riskFreeRetMtx, 
                               buyTransFeeMtx, sellTransFeeMtx):
    '''
    -假設資料共T期, 投資在M個risky assets, 以及1個riskfree asset
    -求出每個risky asset每一期應該要買進以及賣出的金額
    @param symbols, list of string, 
    @param riskyRetMtx, numpy.array, size: M * T
    @param riskFreeRetMtx, numpy.array, size: T
    @param buyTransFeeMtx, numpy.array, size: M * T
    @param sellTransFeeMtx, numpy.array, size: M * T
    
    @return (buyMtx, sellMtx), numpy.array, each size: M*T
    '''
    assert riskyRetMtx.shape == buyTransFeeMtx.shape == sellTransFeeMtx.shape
    assert riskyRetMtx.shape[1] == riskFreeRetMtx.size

    t1 = time.time()
    #create model
    model = ConcreteModel()
    
    #number of asset and number of periods
    model.M = Set(symbols)
    model.T = RangeSet(riskyRetMtx.shape[1])
    model.Tp1 = RangeSet(riskyRetMtx.shape[1]+1)
    
    #parameter
    def riskyRet_init(model, m, t):
        pass
    
    def riskfreeRet_init(model, t):
        pass
    
    def buyTransFee_init(model, m, t):
        pass
    
    def sellTransFee_init(model, m, t):
        pass
    
    model.riskyRet = Param(model.M, model.Tp1, within=Reals, initialize=riskyRet_init)
    model.riskfreeRet = Param(model.Tp1, within=Reals, initilize=riskfreeRet_init)
    model.buyTransFee = Param(model.M, model.T, within=NonNegativeReals, initialize=buyTransFee_init)
    model.sellTransFee = Param(model.M, model.T, within=NonNegativeReals, initialize=sellTransFee_init)
    
    
    #decision variables
    model.buys = Var(model.M, model.T, within=NonNegativeReals)
    model.sells = Var(model.M, model.T, within=NonNegativeReals)
    model.wealth = Var(model.T, within=NonNegativeReals)
     
    #objective
    def objective_rule(model, t):
        pass
    
    model.objective = Objective(rule=objective_rule, sense=maximize)
    
    #constraint
    def riskyWealth_constraint_rule(model, m, t):
        pass
    
    def riskyfreeWealth_constraint_rule(model, m, t):
        pass
    
    model.riskyWeathConstraint = Constraint(rule=riskyWealth_constraint_rule)
    model.riskfreeConstraint = Constraint(rule=riskyfreeWealth_constraint_rule)
    
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
    

if __name__ == '__main__':
    pass