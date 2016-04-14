# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

import numpy as np
import pandas as pd

def fun1(v1=1, *args, **kwargs):
    print "fun 1 v1:", v1
    print "fun 1:", kwargs
    fun2(**kwargs)


def fun2(*args, **kwargs):
    print "fun2:", kwargs['alpha']

def product():

    res = np.zeros(100)
    for v1 in range(10):
        for v2 in range(10):
            ret1 = (1+v1/10.)
            ret2 = (1+v2/10.)
            val = 10 * ret1 * ret2
            res[v1*10+v2] = val
            # print val
    res.sort()
    print res
    print "90 VaR",res[9]
    print "90 CVaR", res[:10].mean()


    print "80 VaR",res[19]
    print "80 CVaR", res[:20].mean()


if __name__ == '__main__':
    # fun1(2, alpha=0.5)
    product()