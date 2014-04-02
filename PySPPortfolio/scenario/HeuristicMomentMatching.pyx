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

cpdef HeuristicMomentMatching (np.ndarray tgtMoms, np.ndarray tgtCorrs, int n_scenario):
    pass


cdef np.ndarray CubicTransform (np.ndarray tgtMoms):
    pass


cdef np.ndarray Cholesky (np.ndarray tgtMoms, np.ndarray tgtCorrs):
    pass

cdef np.ndarray MomentCorrStats (np.ndarray outMoms, np.ndarray outCorrs,
                                 np.ndarray tgtMoms, np.ndarray tgtCorrs):
    pass

if __name__ == '__main__':
    pass