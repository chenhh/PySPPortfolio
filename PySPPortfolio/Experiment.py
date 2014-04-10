# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
start from 2005/1/1 to 2013/12/31
'''

from __future__ import division
import os
import platform
import time
import numpy as np 
import scipy.stats as spstats
from HKWWrapper import HKW_wrapper

FileDir = os.path.abspath(os.path.curdir)
PklBasicFeaturesDir = os.path.join(FileDir,'pkl', 'BasicFeatures')
if platform.uname()[0] == 'Linux':
    ExpResultsDir =  os.path.join('/', 'home', 'chenhh' , 'Dropbox', 
                                  'financial_experiment', 'PySPPortfolio')
    
elif platform.uname()[0] =='Windows':
    ExpResultsDir= os.path.join('C:\\', 'Dropbox', 'financial_experiment', 
                                'PySPPortfolio')    
    
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


def fixedSymbolSPPortfolio(symbols, startDate, endDate,  money=1e6,
                           hist_period=20, n_scenario=1000,
                           buyTransFee=0.001425, sellTransFee=0.004425,
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
    
    #每一期的ScenarioStructure都一樣，建一次即可
    t = time.time()
    probs = np.ones(n_scenario, dtype=np.float)/n_scenario
    constructScenarioStructureFile(n_scenario, probs)
    print "constructScenarioStructureFile %.3f secs"%(time.time()-t)
    
    #設定subprocess的environment variable for cplex
    env = os.environ.copy()
    env['PATH'] += ':/opt/ibm/ILOG/CPLEX_Studio126/cplex/bin/x86-64_linux'
    
    genScenErrDates = []
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
#         print "%s - %s to estimate moments"%(transDate, fullTransDates[tdx:(hist_period+tdx)])
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
        if probVec is None and scenarioMtx is None:
            paramtxt = 'err_%s_n%s_h%s_s%s_a%s.txt'%(
                        transDate, n_rv, hist_period, n_scenario, alpha)
            errorFile = os.path.join(ExpResultsDir, paramtxt)
            with open(errorFile, 'a') as fout:
                fout.write('startDate:%s, endDate:%s\n'%(startDate, endDate))
                fout.write('transDate:%s\n'%(transDate))
                fout.write('n%s_h%s_s%s_a%s\n'%(n_rv, hist_period, n_scenario, alpha))
                fout.write('moment:\n%s\n'%(moments))
                fout.write('corrMtx:\n%s\n'%(corrMtx))
                fout.close()
            
            genScenErrDates.append(transDate)
            #log and goto next period
            wealthProcess[:, tdx] = allocatedWealth
            depositProcess[tdx] = depositWealth
            continue
        
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
            try:
                os.remove(data)
            except OSError:
                pass
                        
        print '*'*75
        print "%s-%s n%s-h%s-s%s-a%s, genscenErr:[%s]\ntransDate %s PySP OK, current wealth %s"%(
                startDate, endDate, n_rv, hist_period, n_scenario, alpha, len(genScenErrDates),    
                transDate,  allocatedWealth.sum() + depositWealth)
        print "%.3f secs"%(time.time()-tloop)
        print '*'*75
    
    #最後一期只結算不買賣
    wealthProcess[:, -1] = allocatedWealth * (1+allRiskyRetMtx[:, -1])
    depositProcess[-1] =  depositWealth * (1+riskFreeRetVec[-1])
   
    finalWealth = (np.dot(allocatedWealth, (1+allRiskyRetMtx[:, -1])) + 
                   depositWealth * (1+riskFreeRetVec[-1]))
    print "final wealth %s"%(finalWealth)
    
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
    summary.write('scenario: %s\n'%(n_scenario))
    summary.write('alpha: %s\n'%(alpha))
    summary.write('symbols: %s \n'%(",".join(symbols)))
    summary.write('transDates (T+1): %s \n'%(
                    ",".join([pd.to_datetime(t).strftime("%Y%m%d") 
                              for t in transDates])))
    summary.write('hist_period: %s\n'%(hist_period))
    summary.write('final wealth:%s \n'%(finalWealth))
    summary.write('generate scenario error count:%s\n'%(len(genScenErrDates)))
    summary.write('generate scenario error dates:\n%s\n'%(
                 ",".join([t.strftime("%Y%m%d") 
                for t in genScenErrDates])))
    summary.write('machine: %s, simulation time:%.3f secs \n'%(
                platform.node(), time.time()-t0))
    
    fileName = os.path.join(resultDir, 'summary.txt')
    with open (fileName, 'w') as fout:
        fout.write(summary.getvalue())
    summary.close()
    
    print "%s-%s n%s-h%s-s%s-a%s\nsimulation ok, %.3f secs"%(
             startDate, endDate, n_rv, hist_period, n_scenario, alpha,    
            time.time()-t0)


if __name__ == '__main__':
    pass