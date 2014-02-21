# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''

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
     

if __name__ == '__main__':
    pass