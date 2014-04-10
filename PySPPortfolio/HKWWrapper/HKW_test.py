# -*- coding: utf-8 -*-
import HKW_wrapper
import platform
import os
import numpy as np
import scipy.stats as spstats
from cStringIO import StringIO
from datetime import date
import subprocess
FileDir = os.path.abspath(os.path.curdir)
 
n_rv = 10
n_scenarios = 10
data = np.random.rand(n_rv, 200)
tgtMoms = np.empty((n_rv, 4))
tgtMoms[:, 0] = data.mean(axis=1)
tgtMoms[:, 1] = data.std(axis=1)
tgtMoms[:, 2] = spstats.skew(data, axis=1)
tgtMoms[:, 3] = spstats.kurtosis(data, axis=1)

tgtCorrs = np.corrcoef(data)


# def _constructTargetMomentFile(moments, transDate):
#     '''
#     @param moments, numpy.array, size: n_rv * 4
#     file format:
#     first row: 4, n_rv
#     then the matrix size: 4 * n_rv
#     -可在matrix之後加入任何註解
#     '''
#     assert moments.shape[1] == 4
#     
#     n_rv = moments.shape[0]
#     data = StringIO()
#     data.write('4\n%s\n'%(n_rv))
#     
#     mom = moments.T
#     #write moment
#     for rdx in xrange(4):
#         data.write(" ".join(str(v) for v in mom[rdx]))
#         data.write('\n')
#     data.write('\n')
#     
#     #write comments
#     data.write('transDate: %s\n'%(transDate))
#         
#     fileName = os.path.join(FileDir, 'tg_moms.txt')
#     with open (fileName, 'w') as fout:
#         fout.write(data.getvalue())
#     data.close()
# 
#     
# def _constructTargetcorrMtxFile(corrMtx, transDate):
#     '''file format:
#     first row: n_rv, n_rv
#     then the matrix size: n_rv * n_rv
#     -可在matrix之後加入任何註解
#     '''
#     n_rv, n_rv2 = corrMtx.shape
#     assert n_rv == n_rv2
#    
#     data = StringIO()
#     data.write('%s\n%s\n'%(n_rv, n_rv))
#     
#     for rdx in xrange(n_rv):
#         data.write(" ".join(str(v) for v in corrMtx[rdx, :]))
#         data.write('\n')
#     data.write('\n')
#     
#     #write comments
#     data.write('transDate: %s\n'%(transDate))
#         
#     fileName = os.path.join(FileDir, 'tg_corrs.txt')
#     with open (fileName, 'w') as fout:
#         fout.write(data.getvalue())
#     data.close()
# 
# 
# def generatingScenarios(moments, corrMtx, n_scenario, transDate, debug=False):
#     '''
#     --使用scengen_HKW產生scenario, 使用前tg_moms.txt與tg_corrs.txt必須存在
#     '''
#     if platform.uname()[0] == 'Linux':
#         exe ='scengen_HKW'
#     elif platform.uname()[0] == 'Windows':
#         exe = 'scengen_HKW.exe'
#     
#     _constructTargetMomentFile(moments, transDate)    
#     _constructTargetcorrMtxFile(corrMtx, transDate)
#     
#     momentFile = os.path.join(FileDir, 'tg_moms.txt')
#     corrMtxFile = os.path.join(FileDir, 'tg_corrs.txt')
#     if not os.path.exists(momentFile):
#         raise ValueError('file %s does not exists'%(momentFile))
#     
#     if not os.path.exists(corrMtxFile):
#         raise ValueError('file %s does not exists'%(corrMtxFile))
#     
#     #deal with convergence problem
#     for kdx in xrange(3):
#         momErr, corrErr = 1e-3 * (10**kdx), 1e-3 * (10**kdx)
#         print "kdx: %s, momErr: %s, corrErr:%s"%(kdx, momErr, corrErr)
#         rc = subprocess.call('./%s %s -f 1 -l 0 -i 50 -t 50 -m %s -c %s'%(
#                         exe, n_scenario, momErr, corrErr), shell=True)
#         if rc == 0:
#             break
#         elif kdx == 2:
#             print "transDate:", transDate
#             print "moment:", moments
#             print "corrMtx:", corrMtx
#           
#             return None, None
#             
#     #Problem with the target correlation matrix - Cholesky failed!
#     
#     probVec, scenarioMtx = parseSamplingMtx(fileName='out_scen.txt')
#     if debug:
#         os.remove('tg_moms.txt')
#         os.remove('tg_corrs.txt')
#         os.remove('out_scen.txt')
#     
#     return probVec, scenarioMtx
#     
# 
# # print tgtMoms
# # print tgtCorrs
# print generatingScenarios(tgtMoms, tgtCorrs, n_scenarios, date(2000,1,1))
HKW_wrapper.HeuristicMomentMatching(tgtMoms, tgtCorrs, n_scenarios)
