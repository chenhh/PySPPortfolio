# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''

from coopr.pyomo import (Set, RangeSet, Param, Var, Objective, Constraint,
                         AbstractModel, Reals, NonNegativeReals)
import coopr.opt.SolverFactory as optSolverFactory

def optimalMultiStagePortfolio(riskyRetMtx, riskFreeRetMtx, 
                               buyTransFeeMtx, sellTransFeeMtx):
    '''
    -假設資料共T期, 投資在M個risky assets, 以及1個riskfree asset
    -求出每個risky asset每一期應該要買進以及賣出的金額
    @param riskyRetMtx, numpy.array, size: M * T
    @param riskFreeRetMtx, numpy.array, size: T
    @param buyTransFeeMtx, numpy.array, size: M * T
    @param sellTransFeeMtx, numpy.array, size: M * T
    
    @return (buyMtx, sellMtx), numpy.array, each size: M*T
    '''
    assert riskyRetMtx.shape == buyTransFeeMtx.shape == sellTransFeeMtx.shape
    assert riskyRetMtx.shape[1] == riskFreeRetMtx.size
    
    opt = optSolverFactory("cplex")

    model = AbstractModel()
    
    #number of asset and number of periods
    model.M = RangeSet(riskyRetMtx.shape[0])
    model.T = RangeSet(riskyRetMtx.shape[1])
    
    #parameter
    model.riskyRet = Param(model.M, model.T, within=Reals)
    model.riskfreeRet = Param(model.T, within=Reals)
    model.buyTransFeeRate = Param(model.M, model.T, within=NonNegativeReals)
    model.sellTransFeeRate = Param(model.M, model.T, within=NonNegativeReals)
    model.wealth = Param()
    
    def riskyWealth():
        pass
    
    def riskLessWealth():
        pass
    
    #decision variables
    model.buys = Var(model.M, model.T, within=NonNegativeReals)
    model.sells = Var(model.M, model.T, within=NonNegativeReals)
     
    #objective
    
    #constraint
    

if __name__ == '__main__':
    pass