# -*- coding: utf-8 -*-
'''

@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''

import numpy as np
import Performance
from time import time

def testPerformance(period=100):
    data = np.random.rand(period) - 0.5
    
    t = time()
    print "performance:"
    print  "Sharpe:", Performance.Sharpe(data)
    print  "SortinoFull:", Performance.SortinoFull(data)
    print  "SortinoPartial:", Performance.SortinoPartial(data)
    print "%.3f secs"%(time()-t)
    
    t = time()
    


if __name__ == '__main__':
    testPerformance(1000)