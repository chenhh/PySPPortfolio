# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

variable:
@var n_rv, positive integer: 投資標的物個數(不含deposit)
@var T, positive integer: 投資期數
@var hist_period, positive integer: 計算統計性質時，所使用的資料期數(含today)
@var startDate，endDate, datetime.date: 投資起始、結束日(非真正投資起始、結束日)
@var money, positive float: 初始資金
@var n_scenario, positive integer: 每一期產生的scenario個數
@var allocatedWealth, numpy.array, size: n_rv: 已投資在標的物的金額
@var depositWealth, positive float, 放在定存的金額


'''
import os
import shutil
import csv
import sys
import getopt
import platform
import subprocess
import time
import itertools
from cStringIO import StringIO
import numpy as np  
import pandas as pd
import scipy.stats as spstats
from datetime import date

FileDir = os.path.abspath(os.path.curdir)
PklBasicFeaturesDir = os.path.join(FileDir,'pkl', 'BasicFeatures')
if platform.uname()[0] == 'Linux':
    ExpResultsDir =  os.path.join('/', 'home', 'chenhh' , 'Dropbox', 
                                  'financial_experiment', 'PySPPortfolio')
#     ExpResultsDir =  os.path.join(FileDir, 'results')
    
elif platform.uname()[0] =='Windows':
    ExpResultsDir= os.path.join('C:\\', 'Dropbox', 'financial_experiment', 
                                'MOGEP', 'PySPPortfolio')    
    

def constructModelMtx(symbols, startDate, endDate, money, hist_period, 
                      buyTransFee=0.003, sellTransFee=0.004425,
                      alpha = 0.95, debug=False):
    '''
    -注意因為最後一期只結算不買賣
    -DataFrame以Date取資料時，有包含last day, 即df[startDate: endDate]
    -包含了endDate的資料，但是使用index取資料時，如df[2:10]，則不包含df.ix[10]的資料
    @param symbols, list
    @param startDate, endDate, datetime.date
    @param momey, positive float
    @param hist_period, positive integer
    
    dataFrame取單一row用df.ix[idx]
    df.index.get_loc(startDate)找index所在位置
    @return riskyRetMtx, numpy.array, size: (n_rv * hist_period+T)
    ''' 
    #read data
    dfs = []
    transDates = None
    for symbol in symbols:
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        tmp = df[startDate: endDate]
        startIdx = df.index.get_loc(tmp.index[0])
        endIdx =  df.index.get_loc(tmp.index[-1])
        if startIdx < (hist_period-1):
            raise ValueError('%s do not have enough data'%(symbol))
        #index from [0, hist_period-1] for estimating statistics
        data = df[startIdx-hist_period+1: endIdx+1]
       
        #check all data have the same transDate
        if transDates is None:
            transDates = data.index.values
        if not np.all(transDates == data.index.values):
            raise ValueError('symbol %s do not have the same trans. dates'%(symbol))
        dfs.append(data)
    
    #fixed transDate data
    fullTransDates = transDates             #size: hist_period + (T +1) -1
    transDates = transDates[hist_period-1:] #size: T+1
    
    #最後一期只結算不買賣, 所以要減一期 
    n_rv, T = len(symbols), len(transDates) - 1
    allRiskyRetMtx = np.empty((n_rv, hist_period+T))
    for idx, df in enumerate(dfs):
        allRiskyRetMtx[idx, :] = df['adjROI'].values/100.
    
    riskFreeRetVec = np.zeros(T+1)
    buyTransFeeMtx = np.ones((n_rv, T)) * buyTransFee
    sellTransFeeMtx = np.ones((n_rv, T))* sellTransFee
    
    #allocated 為已配置在risky asset的金額
    allocatedWealth = np.zeros(n_rv)
    depositWealth = money
    
    if debug:
        print "n_rv: %s, T: %s, hist_period: %s"%(n_rv, T, hist_period)
        print "allRiskyRetMtx size (n_rv * hist_period+T): ", allRiskyRetMtx.shape
        print "riskFreeRetVec size (T+1): ", riskFreeRetVec.shape
        print "buyTransFeeMtx size (n_rv, T): ",  buyTransFeeMtx.shape
        print "sellTransFeeMtx size (n_rv, T): ", sellTransFeeMtx.shape
        print "allocatedWealth size n_rv: ",  allocatedWealth.shape
        print "depositWealth value:", depositWealth
        print "transDates size (T+1):", transDates.shape
        print "full transDates, size(hist+T): ", fullTransDates.shape
        print "alpha value:", alpha
    return {
        "n_rv": n_rv,
        "T": T,
        "allRiskyRetMtx": allRiskyRetMtx, #size: n_rv * (hist_period+T)
        "riskFreeRetVec": riskFreeRetVec,   #size: T+1
        "buyTransFeeMtx": buyTransFeeMtx,   #size: n_rv * T
        "sellTransFeeMtx": sellTransFeeMtx, #size: n_rv * T
        "allocatedWealth": allocatedWealth, #size: n_rv
        "depositWealth": depositWealth,     #size: 1 
        "transDates": transDates,           #size: (T+1)
        "fullTransDates": fullTransDates,   #size: (hist_period+T)
        "alpha": alpha                      #size：１
        }


def constructScenarioStructureFile(n_scenario, probs):
    '''
    產生ScenarioStructure.dat檔案 (nodebased) for pysp
    @param n_scenario, positive integer, scenario個數
    @param probs, numpy.array, size: n_scenario, 每個scenario發生的機率
    '''
    assert len(probs) == n_scenario
    assert np.all(probs >= 0)
    
    data = StringIO()
    #declare node base
    data.write('param ScenarioBasedData := False ;\n')
    
    #stage
    data.write('set Stages := FirstStage SecondStage ;\n')
    
    #set nodes
    data.write('set Nodes := \n')
    data.write(' ' *4 + 'RootNode\n')
    for scen in xrange(n_scenario):
        data.write(' ' *4 + 'Node%s\n'%(scen))
    data.write(';\n\n')
    
    #tree level
    data.write('param NodeStage := \n')
    data.write(" " * 4 + 'RootNode FirstStage \n')
    for scen in xrange(n_scenario):
        data.write(" " * 4 + "Node%s SecondStage\n"%(scen))
    data.write(';\n\n')
    
    #tree arc
    data.write('set Children[RootNode] := \n')
    for scen in  xrange(n_scenario):
        data.write(" " * 4 + 'Node%s\n'%(scen))
    data.write(';\n\n')

    #probability
    data.write('param ConditionalProbability := \n')
    data.write(' ' *4 + 'RootNode 1.0\n')
    for scen in  xrange(n_scenario):
        data.write(" " * 4 + 'Node%s %s\n'%(scen, probs[scen]))
    data.write(';\n\n')

    #scenario
    data.write('set Scenarios := \n')
    for scen in xrange(n_scenario):
        data.write(" " * 4 + "Scenario%s\n"%(scen))
    data.write(';\n\n')
    
    #mapping scenario to leaf
    data.write('param ScenarioLeafNode := \n')
    for scen in  xrange(n_scenario):
        data.write(" " * 4 + 'Scenario%s Node%s\n'%(scen, scen))
    data.write(';\n\n')
    
    #stage variable (Z, Ys are for CVaR)
    data.write('set StageVariables[FirstStage] :=  buys[*] sells[*] Z;\n')
    data.write('set StageVariables[SecondStage] := riskyWealth[*] riskFreeWealth Ys;\n')
   
    #stage cost
    data.write('param StageCostVariable := FirstStage  FirstStageCost\n')
    data.write(' '* 27 + 'SecondStage SecondStageCost ;')
    
    fileName = os.path.join(FileDir, 'models', 'ScenarioStructure.dat')
    with open(fileName, 'w') as fout:
        fout.write(data.getvalue())
        
    data.close()
    

def generatingScenarios(moments, corrMtx, n_scenario, transDate, debug=False):
    '''
    --使用scengen_HKW產生scenario, 使用前tg_moms.txt與tg_corrs.txt必須存在
    '''
    if platform.uname()[0] == 'Linux':
        exe ='scengen_HKW'
    elif platform.uname()[0] == 'Windows':
        exe = 'scengen_HKW.exe'
    
    _constructTargetMomentFile(moments, transDate)    
    _constructTargetcorrMtxFile(corrMtx, transDate)
    
    moments = os.path.join(FileDir, 'tg_moms.txt')
    corrMtx = os.path.join(FileDir, 'tg_corrs.txt')
    if not os.path.exists(moments):
        raise ValueError('file %s does not exists'%(moments))
    
    if not os.path.exists(corrMtx):
        raise ValueError('file %s does not exists'%(corrMtx))
    while True:
        rc = subprocess.call('./%s %s -f 1 -l 0 -i 50 -t 100'%(
                                exe, n_scenario), shell=True)
        
        if rc != 0:
            #decrease to maxError
            rc = subprocess.call('./%s %s -f 1 -l 0 -i 50 -t 100 -m 0.01 -c 0.01'%(
                exe, n_scenario), shell=True)
        else:
            break
    
    probVec, scenarioMtx = parseSamplingMtx(fileName='out_scen.txt')
    if debug:
        os.remove('tg_moms.txt')
        os.remove('tg_corrs.txt')
        os.remove('out_scen.txt')
    
    return probVec, scenarioMtx
    

def  constructRootNodeFile(symbols, allocatedWealth, depositWealth,
                        riskyRet, riskFreeRet, buyTransFee, sellTransFee,
                        alpha):
    '''
    產生RootNode.dat for pysp
    @param symbols, list, list of symbols
    @param allocatedWealth, numpy.array, size: n_rv, 已分配到各symbol的資產
    @param depositWealth, float, 存款金額
    @param riskyRet, numpy.array, size: n_rv, 當期資產報酬率(已知)
    @param riskFreeRet, float, 存款利率
    @param buyTransFee, numpy.array, size: n_rv, 買進手續費
    @param sellTransFee, numpy.array, size: n_rv, 賣出手續費 
    '''
    #RootNode.dat, deterministic parameters
    rootData = StringIO()
    rootData.write('set symbols := %s ;\n'%(" ".join(symbols)))
    
    rootData.write('param allocatedWealth := \n')
    for symbol, alloc in itertools.izip(symbols, allocatedWealth):
        rootData.write(' ' * 4 + '%s %s\n'%(symbol, alloc))
    rootData.write(';\n\n')
    
    rootData.write('param depositWealth := %s ;\n'%(depositWealth))
    
    rootData.write('param buyTransFee := \n')
    for symbol, fee in itertools.izip(symbols, buyTransFee):
        rootData.write(' ' * 4 + '%s %s\n'%(symbol, fee))
    rootData.write(';\n\n')
                            
    rootData.write('param sellTransFee := \n')
    for symbol, fee in itertools.izip(symbols, sellTransFee):
        rootData.write(' ' * 4 + '%s %s\n'%(symbol, fee))
    rootData.write(';\n\n')
    
    rootData.write('param riskyRet := \n')
    for symbol, ret in itertools.izip(symbols,  riskyRet):
        rootData.write(' ' * 4 + '%s %s\n'%(symbol,ret))
    rootData.write(';\n\n')
    
    rootData.write('param riskFreeRet := %s ;\n'%(riskFreeRet))
    rootData.write('param alpha := %s ;\n'%(alpha))

    rootFileName = os.path.join(FileDir, 'models', 'RootNode.dat')
    with open (rootFileName, 'w') as fout:
        fout.write(rootData.getvalue())
    rootData.close()
    
    
def constructScenarioFiles(symbols, n_scenario, scenarioMtx, 
                           predictRiskFreeRet, debug=False):
    '''
    與Node[num].dat檔案(node based) for pysp
    @param n_scenario, positive integer, scenario個數
    @param symbols, list
    @param samplingRetMtx, numpy.array, size: n_rv * n_scenario
    @param predictRiskFreeRet, float, (t+1)期無風利資產報酬率
    '''
    assert scenarioMtx.shape[0] == len(symbols)
    assert scenarioMtx.shape[1] == n_scenario
    
    for sdx in xrange(n_scenario):
        scenData = StringIO()
        scen = scenarioMtx[:, sdx]
        scenData.write('param predictRiskyRet := \n')
        for symbol, ret in itertools.izip(symbols, scen):
            scenData.write(' ' * 4 + '%s %s\n'%(symbol, ret))
        scenData.write(';\n\n')
        
        scenData.write('param predictRiskFreeRet := %s ;\n'%(
                        predictRiskFreeRet))
        
        #檔名必須與ScenarioStrucutre.dat中一致
        scenFileName = os.path.join(FileDir, 'models', 'Node%s.dat'%(sdx))
        with open (scenFileName, 'w') as fout:
            fout.write(scenData.getvalue())
        scenData.close()
    
    if debug:
        for sdx in xrange(n_scenario):
            scenFileName = os.path.join(FileDir, 'models', 'Node%s.dat'%(sdx))
            os.remove(scenFileName)    
     
        
def _constructTargetMomentFile(moments, transDate):
    '''
    @param moments, numpy.array, size: n_rv * 4
    file format:
    first row: 4, n_rv
    then the matrix size: 4 * n_rv
    -可在matrix之後加入任何註解
    '''
    assert moments.shape[1] == 4
    
    n_rv = moments.shape[0]
    data = StringIO()
    data.write('4\n%s\n'%(n_rv))
    
    mom = moments.T
    #write moment
    for rdx in xrange(4):
        data.write(" ".join(str(v) for v in mom[rdx]))
        data.write('\n')
    data.write('\n')
    
    #write comments
    data.write('transDate: %s\n'%(transDate))
        
    fileName = os.path.join(FileDir, 'tg_moms.txt')
    with open (fileName, 'w') as fout:
        fout.write(data.getvalue())
    data.close()

    
def _constructTargetcorrMtxFile(corrMtx, transDate):
    '''file format:
    first row: n_rv, n_rv
    then the matrix size: n_rv * n_rv
    -可在matrix之後加入任何註解
    '''
    n_rv, n_rv2 = corrMtx.shape
    assert n_rv == n_rv2
   
    data = StringIO()
    data.write('%s\n%s\n'%(n_rv, n_rv))
    
    for rdx in xrange(n_rv):
        data.write(" ".join(str(v) for v in corrMtx[rdx, :]))
        data.write('\n')
    data.write('\n')
    
    #write comments
    data.write('transDate: %s\n'%(transDate))
        
    fileName = os.path.join(FileDir, 'tg_corrs.txt')
    with open (fileName, 'w') as fout:
        fout.write(data.getvalue())
    data.close()
    

def parseSamplingMtx(fileName='out_scen.txt'):
    '''讀取moment matching所取樣出的變數
    #開頭的為註解行
    each row is a scenario, and 
    the first element is the probability,
    the second element is the first variable of the scenario, 
    the third element is the second varible of the scenario,
    ...
    therefore each row (scenario) contains (n_rv+1) elements
    
    -可用numpy.genfromtxt方法讀取
    
    return sample matrix, numpy.array, size: n_rv * n_scenario
    '''
    with open(fileName) as fin:
        mtx = np.genfromtxt(fin)
        
    probVec = mtx[:, 0]
    scenarioMtx = mtx[:, 1:].T
    return probVec, scenarioMtx
 

def fixedSymbolSPPortfolio(symbols, startDate, endDate,  money=1e6,
                           hist_period=20, n_scenario=1000,
                           buyTransFee=0.003, sellTransFee=0.004425,
                           alpha=0.95, debug=False):
    '''
    -固定投資標的物(symbols)，只考慮buy, sell的交易策略
    -假設symbols有n_rv個，投資期數共T期(最後一期不買賣，只結算)
    
    @param symbols, list, target assets
    @param startDate, endDate, datetime.date, 交易的起始，結束日期
    @param money, positive float, 初使資金
    @param hist_period, positive integer, 用於計算moment與corr mtx的歷史資料長度
    @param n_scenario, positive integer, 每一期產生的scenario個數
    
    @return translog 
        "n_rv": n_rv,
        "T": T,
        "allRiskyRetMtx": allRiskyRetMtx,   #size: n_rv * (hist_period+T)
        #[0:hist_period]用於估計moments與corrMtx
        "riskFreeRetVec": riskFreeRetVec,   #size: T+1
        "buyTransFeeMtx": buyTransFeeMtx,   #size: n_rv * T
        "sellTransFeeMtx": sellTransFeeMtx, #size: n_rv * T
        "allocatedWealth": allocatedWealth, #size: n_rv
        "depositWealth": depositWealth,     #size: 1 
        "transDates": transDates,           #size: (T+1)
        "fullTransDates": fullTransDates,   #size: (hist_period+T)
         "alpha": alpha                      #size：１
        }
    '''
    
    t0 = time.time()
    param = constructModelMtx(symbols, startDate, endDate, money, hist_period,
                              buyTransFee, sellTransFee, alpha, debug)
    print "constructModelMtx %.3f secs"%(time.time()-t0)
    
    n_rv, T =param['n_rv'], param['T']
    allRiskyRetMtx = param['allRiskyRetMtx']
    riskFreeRetVec = param['riskFreeRetVec']
    buyTransFeeMtx = param['buyTransFeeMtx']
    sellTransFeeMtx = param['sellTransFeeMtx']
    allocatedWealth = param['allocatedWealth']
    depositWealth = param['depositWealth']
    transDates = param['transDates']
    fullTransDates = param['fullTransDates']
    
    #process from t=0 to t=(T+1)
    buyProcess = np.zeros((n_rv, T))
    sellProcess = np.zeros((n_rv, T))
    wealthProcess = np.zeros((n_rv, T+1))
    depositProcess = np.zeros(T+1)
    VaRProcess = np.zeros(T)
    
    #setup result directory
    resultDir0 = os.path.join(ExpResultsDir, "n%s_h%s_s%s_a%s"%(
                                        n_rv, hist_period, n_scenario, alpha))
    if not os.path.exists(resultDir0):
        os.mkdir(resultDir0)
        
    t1 = pd.to_datetime(transDates[0]).strftime("%Y%m%d")
    t2 = pd.to_datetime(transDates[-1]).strftime("%Y%m%d")
    rnd = time.strftime("%y%m%d%H%M%S")
    resultDir = os.path.join(resultDir0, "%s_%s-%s_%s"%(
                        fixedSymbolSPPortfolio.__name__, 
                        t1, t2, rnd))
    
    while True:
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
            break
        rnd = time.strftime("%y%m%d%H%M%S")
        resultDir = os.path.join(resultDir0, "%s_%s-%s_%s"%(
                        fixedSymbolSPPortfolio.__name__, t1, t2, rnd))
    
    
    #每一期的ScenarioStructure都一樣，建一次即可
    t = time.time()
    probs = np.ones(n_scenario, dtype=np.float)/n_scenario
    constructScenarioStructureFile(n_scenario, probs)
    print "constructScenarioStructureFile %.3f secs"%(time.time()-t)
    
    #設定subprocess的environment variable for cplex
    env = os.environ.copy()
    env['PATH'] += ':/opt/ibm/ILOG/CPLEX_Studio126/cplex/bin/x86-64_linux'
    
    for tdx in xrange(T):
        tloop = time.time()
        transDate = pd.to_datetime(transDates[tdx])
        
        #realized today return
        allocatedWealth = allocatedWealth * (1+allRiskyRetMtx[:, hist_period+tdx])
        depositWealth =  depositWealth * (1+riskFreeRetVec[tdx])
        
        #投資時已知當日的ret(即已經知道當日收盤價)
        #算出4 moments與correlation matrix
        t = time.time()
        subRiskyRetMtx = allRiskyRetMtx[:, tdx:(hist_period+tdx)]
        print "%s - %s to estimate moments"%(transDate, fullTransDates[tdx:(hist_period+tdx)])
        moments = np.empty((n_rv, 4))
        moments[:, 0] = subRiskyRetMtx.mean(axis=1)
        moments[:, 1] = subRiskyRetMtx.std(axis=1)
        moments[:, 2] = spstats.skew(subRiskyRetMtx, axis=1)
        moments[:, 3] = spstats.kurtosis(subRiskyRetMtx, axis=1)
        corrMtx = np.corrcoef(subRiskyRetMtx)
        print "%s - moments, corrMtx %.3f secs"%(transDate, time.time()-t)
        
        #call scngen_HKW抽出下一期的樣本中
        t = time.time()
        print "start generating scenarios"
        probVec, scenarioMtx = generatingScenarios(moments, corrMtx, n_scenario, transDate)
        print "%s - probVec size n_rv: %s"%(transDate, probVec.shape)
        print "%s - scenarioMtx size: (n_rv * n_scenario): %s"%(transDate, scenarioMtx.shape)
        print "%s - generation scenario, %.3f secs"%(transDate, time.time()-t)
        
        #使用抽樣樣本建立ScenarioStructure.dat, RootNode.dat與不同scenario的檔案
        t = time.time()
        constructRootNodeFile(symbols, allocatedWealth, depositWealth,
                              allRiskyRetMtx[:, hist_period+tdx],
                              riskFreeRetVec[tdx], 
                              buyTransFeeMtx[:, tdx], 
                              sellTransFeeMtx[:, tdx],
                              alpha)
        
        constructScenarioFiles(symbols, n_scenario,  scenarioMtx, riskFreeRetVec[tdx+1])
        print "%s - generation root and node files, %.3f secs"%(transDate, time.time()-t)
        
        #使用抽出的樣本解SP(runef)，得到最佳的買進，賣出金額
        t = time.time()
        modelDir = os.path.join(FileDir, "models")
        cmd = 'runef -m %s -i %s  --solution-writer=coopr.pysp.csvsolutionwriter \
             --solver=cplex --solve 1>/dev/null'%(modelDir, modelDir)
       
        rc = subprocess.call(cmd, env=env, shell=True)
        print "%s - runef, %.3f secs"%(transDate, time.time()-t)
        
        #parse ef.csv, 並且執行買賣
        #(stage, node, var, index, value)
        with open('ef.csv') as fin:  
            for row in csv.DictReader(fin, ('stage', 'node', 'var', 'symbol', 'value')): 
                for key in row.keys():
                    row[key] = row[key].strip()
                node, symbol = row['node'], row['symbol']
                 
                if node == "RootNode" and row['var'] == "Z":
                    VaRProcess[tdx] = float(row['value'])
                 
                if  node == 'RootNode' and symbol in symbols:
                    #get symbol index 
                    idx = symbols.index(row['symbol'])
                    if row['var'] == 'buys':
                        allocatedWealth[idx] += float(row['value'])
                        buy = (1 + buyTransFeeMtx[idx, tdx]) * float(row['value'])
                        buyProcess[idx, tdx] = buy
                        depositWealth -= buy
                    elif row['var'] == 'sells':
                        allocatedWealth[idx] -= float(row['value'])
                        sell = (1 - sellTransFeeMtx[idx, tdx]) * float(row['value'])
                        sellProcess[idx, tdx] = sell
                        depositWealth += sell
                    else:
                        raise ValueError('unknown variable %s'%(row))
        
        #log wealth and signal process
        wealthProcess[:, tdx] = allocatedWealth
        depositProcess[tdx] = depositWealth
            
        #remove mdl file
        mdlFile = os.path.join(os.getcwd(), 'models', 'RootNode.dat')
        
        os.remove(mdlFile)
        for sdx in xrange(n_scenario):
            mdlFile = os.path.join(os.getcwd(), 'models', 'Node%s.dat'%(sdx))
            os.remove(mdlFile)  
        
        #move temporary file
        resultFiles = ('parse_table_datacmds.py', 'ef.csv', 'efout.lp', 
                       'out_scen.txt', 'tg_corrs.txt', 'tg_moms.txt')
         
        #delete files
        for data in  resultFiles:
            os.remove(data)
         
#         transDateDir = os.path.join(resultDir, transDate.strftime("%Y%m%d"))
#         if not os.path.exists(transDateDir):
#             os.mkdir(transDateDir)
#         for res in resultFiles:
#             try:
#                 tgt = os.path.join(transDateDir, res)
#                 shutil.move(res, tgt)
#             except Exception as e:
#                 print e
                
        
        print '*'*75
        print "transDate %s PySP OK, current wealth %s"%(
                transDate,  allocatedWealth.sum() + depositWealth)
        print "%.3f secs"%(time.time()-tloop)
        print '*'*75
    
    #最後一期只結算不買賣
    wealthProcess[:, -1] = allocatedWealth * (1+allRiskyRetMtx[:, -1])
    depositProcess[-1] =  depositWealth * (1+riskFreeRetVec[-1])
   
    finalWealth = (np.dot(allocatedWealth, (1+allRiskyRetMtx[:, -1])) + 
                   depositWealth * (1+riskFreeRetVec[-1]))
    print "final wealth %s"%(finalWealth)
    
    #store data in pkl
    pd_buyProc = pd.DataFrame(buyProcess.T, index=transDates[:-1], columns=symbols)
    pd_sellProc = pd.DataFrame(sellProcess.T, index=transDates[:-1], columns=symbols) 
    pd_wealthProc = pd.DataFrame(wealthProcess.T, index=transDates, columns=symbols)
    pd_depositProc = pd.Series(depositProcess.T, index=transDates)
    pd_VaRProc = pd.Series(VaRProcess.T, index=transDates[:-1])
    
    buyProcFile = os.path.join(resultDir, 'buyProcess.pkl')
    sellProcFile = os.path.join(resultDir, 'sellProcess.pkl')
    wealthProcFile = os.path.join(resultDir, 'wealthProcess.pkl')
    depositProcFile = os.path.join(resultDir, 'depositProcess.pkl')
    VaRProcFile = os.path.join(resultDir, 'VaRProcess.pkl')
    
    pd_buyProc.save(buyProcFile)
    pd_sellProc.save(sellProcFile)
    pd_wealthProc.save(wealthProcFile)
    pd_depositProc.save(depositProcFile)
    pd_VaRProc.save(VaRProcFile)
    
    #store data in csv 
    buyProcCSV = os.path.join(resultDir, 'buyProcess.csv')
    sellProcCSV = os.path.join(resultDir, 'sellProcess.csv')
    wealthProcCSV = os.path.join(resultDir, 'wealthProcess.csv')
    depositProcCSV = os.path.join(resultDir, 'depositProcess.csv')
    VaRProcCSV = os.path.join(resultDir, 'VaRProcess.csv')
    
    pd_buyProc.to_csv(buyProcCSV)
    pd_sellProc.to_csv(sellProcCSV)
    pd_wealthProc.to_csv(wealthProcCSV)
    pd_depositProc.to_csv(depositProcCSV)
    pd_VaRProc.to_csv(VaRProcCSV)
    
    #print results
    print "buyProcess:\n",pd_buyProc
    print "sellProcess:\n",pd_sellProc
    print "wealthProcess:\n",pd_wealthProc
    print "depositProcess:\n",pd_depositProc
    print "VaRProcess:\n", pd_VaRProc
    
    #generating summary files
    summary = StringIO()
    summary.write('n_rv: %s\n'%(n_rv))
    summary.write('T: %s\n'%(T))
    summary.write('alpha: %s\n'%(alpha))
    summary.write('symbols: %s \n'%(",".join(symbols)))
    summary.write('transDates (T+1): %s \n'%(
                    ",".join([pd.to_datetime(t).strftime("%Y%m%d") 
                              for t in transDates])))
    summary.write('hist_period: %s\n'%(hist_period))
    summary.write('final wealth:%s \n'%(finalWealth))
    
    fileName = os.path.join(resultDir, 'summary.txt')
    with open (fileName, 'w') as fout:
        fout.write(summary.getvalue())
    summary.close()
    
    print "simulation ok, %.3f secs"%(time.time()-t0)



def testScenarios():
    n_rv, T = 3, 100
    data = np.random.randn(n_rv, T)
    moments = np.empty((n_rv, 4))
    moments[:, 0] = data.mean(axis=1)
    moments[:, 1] = data.std(axis=1)
    moments[:, 2] = spstats.skew(data, axis=1)
    moments[:, 3] = spstats.kurtosis(data, axis=1)
    print "moments:\n",moments
    corrMtx = np.corrcoef(data)
    print "corrMtx:\n", corrMtx
    transDate = date(2000,1,1)
    n_scenario = 1000
    probVec, scenarioMtx = generatingScenarios(moments, corrMtx, 
                                    n_scenario, transDate, debug=False)
    print scenarioMtx.shape
    s_moments = np.empty((n_rv, 4))
    s_moments[:, 0] = scenarioMtx.mean(axis=1)
    s_moments[:, 1] = scenarioMtx.std(axis=1)
    s_moments[:, 2] = spstats.skew(scenarioMtx, axis=1)
    s_moments[:, 3] = spstats.kurtosis(scenarioMtx, axis=1)
    s_corrMtx = np.corrcoef(scenarioMtx)
    print "s_moments:\n", s_moments
    print "s_corrMtx:\n", s_corrMtx
   
    

if __name__ == '__main__':
   
    money = 1e6
    debug = True
    
    #market value top 20 (2013/12/31)
    symbols = ['2330', '2317', '6505', '2412', '2454',
                '2882', '1303', '1301', '1326', '2881',
                '2002', '2308', '3045', '2886', '2891',
                '1216', '2382', '2105', '2311', '2912'
               ]
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:h:s:a:y:", 
                            ["symbols", "histperiod", "scenario", 
                             "alpha", "year"])
        
        for opt, arg in opts:
            if opt in ('-n', '--symbols'):
                n_symbol = int(arg)
                symbolIDs = symbols[:n_symbol]
                
            if opt in ('-h', '--histperiod'):
                hist_period = int(arg)
            
            if opt in ('-s', '--scenario'):
                n_scenario = int(arg)
                
            if opt in ('-a', '--alpha'):
                alpha = float(arg)
            
            if opt in ('-y', '--year'):
                year = int(arg)
                startDate = date(year, 1, 1)
                endDate = date(year, 1, 31)
            
        fixedSymbolSPPortfolio(symbolIDs, startDate, endDate,  money=money,
                           hist_period=hist_period , n_scenario=n_scenario,
                           alpha=alpha, debug=debug)
                
    except getopt.GetoptError as e:  
    # print help information and exit:
        print e 
        sys.exit(-1)    
    