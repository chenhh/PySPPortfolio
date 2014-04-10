# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
from __future__ import division
cimport numpy as np
import numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t



# @cython.boundscheck(False)
cpdef HeuristicCopula(np.ndarray data, double alpha, int n_scenario):
    '''
    @data, numpy.array, size: N*D, N is num. of data, D is dimensions
    @alpha, float, confidence level of CVaR
    @n_scenario, integer
    '''
                    
    pass



if __name__ == '__main__':
    pass