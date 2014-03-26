# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

PySP class for portfolio problem
'''
import os
import time
import subprocess
import numpy as np
import shutil
from cStringIO import StringIO

class SPPortfolio(object):
    '''PySP class for modeling portfolio problem using NodeBasedData'''
    
    def __init__(self, modelDir, nodeDir, n_scenario):
        self.modelDir = modelDir
        self.nodeDir = nodeDir
        self.n_scenario = n_scenario
        #equalprobs
        self.probs = np.ones(n_scenario, np.float)/n_scenario
        
        #add env to utilizing mysolutionwriter
        self.env =  os.environ.copy()
        self.env['PYTHONPATH'] = os.getcwd()
        
    def genReferenceModel(self):
        refModel = os.path.join(os.path.curdir, 'models', "minCVaRReferenceModel.py")
        shutil.copy(refModel, self.modelDir)
        shutil.move(os.path.join(self.modelDir, "minCVaRReferenceModel.py"),
                    os.path.join(self.modelDir, "ReferenceModel.py"))
    
    def genScenarioStructureFile(self):
        '''
        產生ScenarioStructure.dat檔案 (nodebased) for pysp
        @param n_scenario, positive integer, scenario個數
        @param probs, numpy.array, size: n_scenario, 每個scenario發生的機率
        '''
        data = StringIO()
        #declare nodebase and stages
        data.write('param ScenarioBasedData := False ;\n')
        data.write('set Stages := FirstStage SecondStage ;\n')
        
        #set nodes
        data.write('set Nodes := \n')
        data.write(' ' *4 + 'RootNode\n')
        for scen in xrange(self.n_scenario):
            data.write(' ' * 4 + 'Node%s\n'%(scen))
        data.write(';\n\n')
        
        #tree level
        data.write('param NodeStage := \n')
        data.write(" " * 4 + 'RootNode FirstStage \n')
        for scen in xrange(self.n_scenario):
            data.write(" " * 4 + "Node%s SecondStage\n"%(scen))
        data.write(';\n\n')
        
        #tree arc
        data.write('set Children[RootNode] := \n')
        for scen in  xrange(self.n_scenario):
            data.write(" " * 4 + 'Node%s\n'%(scen))
        data.write(';\n\n')
    
        #probability
        data.write('param ConditionalProbability := \n')
        data.write(' ' *4 + 'RootNode 1.0\n')
        for scen in  xrange(self.n_scenario):
            data.write(" " * 4 + 'Node%s %s\n'%(scen, self.probs[scen]))
        data.write(';\n\n')
    
        #scenario
        data.write('set Scenarios := \n')
        for scen in xrange(self.n_scenario):
            data.write(" " * 4 + "Scenario%s\n"%(scen))
        data.write(';\n\n')
        
        #mapping scenario to leaf
        data.write('param ScenarioLeafNode := \n')
        for scen in  xrange(self.n_scenario):
            data.write(" " * 4 + 'Scenario%s Node%s\n'%(scen, scen))
        data.write(';\n\n')
        
        #stage variable (Z, Ys are for CVaR)
        data.write('set StageVariables[FirstStage] :=  buys[*] sells[*] Z;\n')
        data.write('set StageVariables[SecondStage] := riskyWealth[*] riskFreeWealth Ys;\n')
       
        #stage cost
        data.write('param StageCostVariable := FirstStage  FirstStageCost\n')
        data.write(' '* 27 + 'SecondStage SecondStageCost ;')
        
        fileName = os.path.join(self.modelDir, 'ScenarioStructure.dat')
        with open(fileName, 'w') as fout:
            fout.write(data.getvalue())
        data.close()
    
    
    def  genRootNodeFile(self, symbols, allocatedWealth, depositWealth,
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
        @param alpha, float, confidence level of CVaR 
        '''
        #RootNode.dat, deterministic parameters
        rootData = StringIO()
        rootData.write('set symbols := %s ;\n'%(" ".join(symbols)))
        
        #已投資於風險資產金額
        rootData.write('param allocatedWealth := \n')
        for symbol, alloc in zip(symbols, allocatedWealth):
            rootData.write(' ' * 4 + '%s %s\n'%(symbol, alloc))
        rootData.write(';\n\n')
        
        #定存金額
        rootData.write('param depositWealth := %s ;\n'%(depositWealth))
        
        #各風險資產買進手續費?
        rootData.write('param buyTransFee := \n')
        for symbol, fee in zip(symbols, buyTransFee):
            rootData.write(' ' * 4 + '%s %s\n'%(symbol, fee))
        rootData.write(';\n\n')
        
        #各風險資產賣出手續費                   
        rootData.write('param sellTransFee := \n')
        for symbol, fee in zip(symbols, sellTransFee):
            rootData.write(' ' * 4 + '%s %s\n'%(symbol, fee))
        rootData.write(';\n\n')
        
        #風險資產報酬率
        rootData.write('param riskyRet := \n')
        for symbol, ret in zip(symbols,  riskyRet):
            rootData.write(' ' * 4 + '%s %s\n'%(symbol,ret))
        rootData.write(';\n\n')
        
        #無風險資產報酬率
        rootData.write('param riskFreeRet := %s ;\n'%(riskFreeRet))
        rootData.write('param alpha := %s ;\n'%(alpha))
    
        rootFileName = os.path.join(self.nodeDir, 'RootNode.dat')
        with open (rootFileName, 'w') as fout:
            fout.write(rootData.getvalue())
        rootData.close()
    
    
    def genScenarioFiles(self, symbols, scenarioMtx, predictRiskFreeRet):
        '''
        與Node[num].dat檔案(node based) for pysp
        @param symbols, list
        @param samplingRetMtx, numpy.array, 
                size: n_rv * n_scenario, (t+1)期風資產報酬率
        @param predictRiskFreeRet, float, (t+1)期無風利資產報酬率
        '''
        assert scenarioMtx.shape[0] == len(symbols)
        assert scenarioMtx.shape[1] == self.n_scenario
        
        for sdx in xrange(self.n_scenario):
            scenData = StringIO()
            scen = scenarioMtx[:, sdx]
            scenData.write('param predictRiskyRet := \n')
            for symbol, ret in zip(symbols, scen):
                scenData.write(' ' * 4 + '%s %s\n'%(symbol, ret))
            scenData.write(';\n\n')
            
            scenData.write('param predictRiskFreeRet := %s ;\n'%(
                            predictRiskFreeRet))
            
            #檔名必須與ScenarioStrucutre.dat中一致
            scenFileName = os.path.join(self.nodeDir, 'Node%s.dat'%(sdx))
            with open (scenFileName, 'w') as fout:
                fout.write(scenData.getvalue())
            scenData.close()
        
    def solve(self):
        t = time.time()
        output ="data_%s.lp"%(os.getpid())
        cmd = 'runef -m %s -i %s --output-file=%s --solution-writer=jsonsolutionwriter\
             --solver=glpk  --solve '%(self.modelDir, self.nodeDir, output)
        
        rc = subprocess.call(cmd, env=self.env, shell=True)
        
        shutil.move(output, os.path.join('model%s'%(os.getpid()), output))
        print "runef, %.3f secs"%(time.time()-t)
        
def test():
    t = time.time()
    modelDir =os.path.join(os.path.curdir, "model%s"%os.getpid())
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)
        
    n_scenario = 100
    obj = SPPortfolio(modelDir, modelDir, n_scenario)
    obj.genReferenceModel()
    obj.genScenarioStructureFile()
    
    symbols = ('2330', '2317')
    allocatedWealth = np.zeros(len(symbols))
    depositWealth = 0.
    riskyRet = np.random.randn(len(symbols))
    riskFreeRet = 0.2
    buyTransFee = np.ones(len(symbols))*0.001425
    sellTransFee = np.ones(len(symbols))*0.004425
    alpha = 0.95
    scenarioMtx = np.random.randn(len(symbols), n_scenario)
    predictRiskFreeRet = 0.2
    obj.genRootNodeFile(symbols, allocatedWealth, depositWealth,
                        riskyRet, riskFreeRet, buyTransFee, sellTransFee,
                        alpha)
    obj.genScenarioFiles(symbols, scenarioMtx, predictRiskFreeRet)
    obj.solve()
    print "all, %.3f secs"%(time.time()-t)
    
if __name__ == '__main__':
#     import sys
#     sys.path.append(os.path.join(os.getcwd()))
    test()