# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
import os
import sys
import platform
import subprocess
import time
from cStringIO import StringIO
import numpy as np  
import pandas as pd
import scipy as sp
import scipy.stats as spstats
from datetime import (date, timedelta)

PklBasicFeaturesDir = os.path.join(os.getcwd(),'pkl', 'BasicFeatures')
if platform.uname()[0] == 'Linux':
    ExpResultsDir =  os.path.join('/', 'home', 'chenhh' , 'Dropbox', 
                                  'financial_experiment', 'PySPPortfolio')
elif platform.uname()[0] =='Windows':
    ExpResultsDir= os.path.join('C:\\', 'Dropbox', 'financial_experiment', 
                                'MOGEP', 'PySPPortfolio')    
    

def constructModelMtx(symbols, startDate, endDate, money, hist_day):
    '''
    -注意因為最後一期只結算不買賣
    @return riskyRetMtx, numpy.array, size: (n_rv * T+1)
    '''
    buyTransFee, sellTransFee = 0.003, 0.004425
    
    dfs = []
    for symbol in symbols:
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, 
                                    'BasicFeatures_%s_00-12.pkl'%symbol))
        dfs.append(df[startDate: endDate])
    
    n_rv, T = len(symbols), dfs[0].index.size - 1
    riskyRetMtx = np.empty((n_rv, T+1))
    for idx, df in enumerate(dfs):
        riskyRetMtx[idx, :] = df['adjROI'].values/100
    
    riskFreeRetVec = np.zeros(T+1)
    buyTransFeeMtx = np.ones((n_rv, T)) * buyTransFee
    sellTransFeeMtx = np.ones((n_rv,T))* sellTransFee
    
    #allocated [0, n_rv-1]為已配置在risky asset的金額，第n_rv為cash
    allocatedVec = np.zeros(n_rv+1)
    allocatedVec[-1] = money
    
    return {"riskyRetMtx": riskyRetMtx,
            "riskFreeRetVec": riskFreeRetVec,
            "buyTransFeeMtx": buyTransFeeMtx,
            "sellTransFeeMtx": sellTransFeeMtx,
            "allocatedVec": allocatedVec,
            "transDates": transDates
            }

def constructScenarioStructure(n_scenario):
    '''產生ScenarioStructure.dat檔案 (nodebased)
    '''
    data = StringIO()
    #declare node base
    data.write('param ScenarioBasedData := False ;\n')
    
    #stage
    data.write('set Stages := FirstStage SecondStage ;\n')
    
    #tree level
    data.write('param NodeStage := RootNode FirstStage \n')
    for scen in xrange(n_scenario):
        data.write(" " * 19 + "Node%s SecondStage\n"%(scen))
    data.write(';\n')
    
    #tree arc
    data.write('set Children[RootNode] := \n')
    for scen in  xrange(n_scenario):
        data.write('" " * 19 + Node%s\n'%(scen))
    data.write(' ;\n')

    #mapping scenario to leaf
    data.write('param ScenarioLeafNode := \n')
    for scen in  xrange(n_scenario):
        data.write('" " * 19 + Scenario%s Node%s\n'%(scen, scen))
    data.write(' ;\n')
    
    #stage variable
    data.write('set StageVariables[FirstStage] :=  buys[*]\n')
    data.write(' ' * 19 + 'sells[*];\n')
    data.write('set StageVariables[SecondStage] := riskyWealth[*]\n')
    data.write(' ' * 19 + 'riskFreeWealth;\n')
    
    #stage wealth
    data.write('StageCostVariable := FirstStage  FirstStageWealth\n')
    data.write(' '* 10 + 'SecondStage SecondStageWealth ;')
    
    fileName = os.path.join('models', 'ScenarioStructure.dat')
    with open(fileName, 'w') as fout:
        fout.write(data.getvalue())
        
    data.close()
    
    return sys.exit(0)
    
    
def constructScenarios(transDate, n_scenario, symbols, samplingRetMtx):
    '''產生transDate_Scenarios_scenario_num.dat檔案(node based)
    所以要產生rootNode.dat和scenario.dat
    '''
    assert samplingRetMtx.shape[0] == len(symbols)
    assert samplingRetMtx.shape[1] == n_scenario
    
    
    #deterministic parameters
    rootData = StringIO()
    rootData.write('set symbols := %s ;'%(" ".join(str(s) for s in xrange(len(symbols)))))
    rootData.write('param allocatedWealth : = %s ;'%())
    rootData.write('param depositWealth : = %s ;'%())
    rootData.write('param riskFreeRet : = %s ;'%())
    rootData.write('param buyTransFee : = %s ;'%())
    rootData.write('param sellTransFee : = %s ;'%())
 
    rootFileName = os.path.join('models', 'RootNode.dat')
    with open (rootFileName, 'w') as fout:
        fout.write(rootData.getvalue())
    rootData.close()
        
    for sdx in xrange(n_scenario):
        scenData = StringIO()
        scenData.write('param riskyRet : = %s ;'%())
        #檔名必須與ScenarioStrucutre.dat中一致
        scenFileName = os.path.join('models', 'Node%s.dat'%(sdx))
        with open (scenFileName, 'w') as fout:
            fout.write(scenData.getvalue())
        scenData.close()
    
    return sys.exit(0)
        
def constructTargetMomentFile(moments):
    '''file format:
    first row: 4, n_rv
    then the matrix size: 4 * n_rv
    -可在matrix之後加入任何註解
    '''
    assert moments.shape[1] == 4
    n_rv = moments.shape[0]
    data = StringIO()
    data.write('4 %s\n'%(n_rv))
    
    #write moment
    for rdx in xrange(4):
        data.write()
    
    #write comment
    
    fileName = os.path.join('.', 'tg_moms.txt')
    with open (fileName, 'w') as fout:
        fout.write(data.getvalue())
    data.close()
    
    return sys.exit(0)
    
    
def constructTargetcorrMtxFile(corrMtx):
    '''file format:
    first row: n_rv, n_rv
    then the matrix size: n_rv * n_rv
    -可在matrix之後加入任何註解
    '''
    n_rv, n_rv2 = corrMtx.shape
    assert n_rv == n_rv2
    
    data = StringIO()
    data.write('%s %s\n'%(n_rv, n_rv))
    
    for rdx in xrange(n_rv):
        data.write()
    
    fileName = os.path.join('.', 'tg_corrs.txt')
    with open (fileName, 'w') as fout:
        fout.write(data.getvalue())
    data.close()
    
    return sys.exit(0)
    

def parseSamplingMtx():
    pass
        
def fixedSymbolSPPortfolio(symbols, startDate, endDate,  money=1e6,
                           hist_day=20, n_scenario=1000):
    '''
    -固定投資標的物(symbols)，只考慮buy, sell的交易策略
    -假設symbols有n_rv個，投資期數共T期(最後一期不買賣，只結算)
    
    @param symbols, list, target assets
    @param startDate, endDate, datetime.date, 交易的起始，結束日期
    @param money, positive float, 初使資金
    @param hist_day, positive integer, 用於計算moment與corr mtx的歷史資料長度
    @param n_scenario, positive integer, 每一期產生的scenario個數
    
    @return translog 
        {wealth: 期末資金,
         wealthProcess: 總資金變動量(matrix of n+1*T),
         signalProcess: 資產買賣訊號(matrix of n*T)
         transCount: 交易次數(買進->賣出算一次),
         n_rv: 投資資產總數，
         T: 投資期數
        }
    '''
    param = constructModelMtx(symbols, startDate, endDate, money)
    n_rv, T = len(param), param['buyTransFeeMtx'].shape[1]
    
    #setup result directory
    resultDir = os.path.join(ExpResultsDir, "%s_%s"%(
                        fixedSymbolSPPortfolio.__name__, 
                        time.strftime("%y%m%d_%H%M%S")))
    
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)
    
    #因為每一期的ScenarioStructure都一樣，建一次即可
    
    for tdx in xrange(T):
        transDate = param['transDates'][tdx]
        transDateDir = os.path.join(resultDir, transDate)
        if not os.path.exists(transDateDir):
            os.mkdir(transDateDir)
        
        #算出4 moments與correlation matrix
        
        #將moments, corrMtx寫入tg_moms, tg_corrs, 
        #call scngen_HKW抽出下一期的樣本中
        subprocess.call("./scengen_HKW %s -f 1"%(n_scenario), shell=True)
    
        #讀取抽樣檔 out_scen.txt
        
        #move tg_moms.txt, tg_corrs.txt to results directory
        os.rename("tg_moms.txt", os.path.join(transDateDir, "tg_moms.txt"))
        os.rename("tg_corrs.txt", os.path.join(transDateDir, "tg_corrs.txt"))
        os.rename("out_scen.txt", os.path.join(transDateDir, "out_scen.txt"))
        
        #使用抽樣樣本建立ScenarioStructure.dat, RootNode.dat與不同scenario的檔案
        
        
        #使用抽出的樣本解SP(runef)，得到最佳的買進，賣出金額
    
    
        #更新wealthProcess與singalProcess
        pass
    
    #最後一期只結算不買賣
    
    
    

if __name__ == '__main__':
    pass