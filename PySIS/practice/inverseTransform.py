# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
已知目標dist.之inverse cdf F_1,

'''
import numpy as np
import matplotlib.pyplot as plt

def inverseTransformSampling(invCDF, n_sample=1000):
    #sampling from uniform dist. [0,1)
    ys = np.random.rand(n_sample)
    
    vfunc = np.vectorize(invCDF, otypes=[np.float])
    xs = vfunc(ys)
    
    return xs

def test_inverseTransformSampling():
    invCDF = lambda x: (0<x<1)*3*x**2
    
    
    print inverseTransformSampling(invCDF)

if __name__ == '__main__':
    test_inverseTransformSampling()