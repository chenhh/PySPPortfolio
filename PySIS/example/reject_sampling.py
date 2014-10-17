# -*- coding: utf-8 -*-
'''
.. codeauthor:: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
'''
__author__ = 'Hung-Hsin Chen'

import numpy as np
import matplotlib.pyplot as plt

def reject_sampling(n_sample=100000):
    '''
    使用uniform dist抽出f(x)=6*x*(1-x), 0<x<1
    '''
    pts = np.random.rand(n_sample)
    accepts = []




    #plot
    plt.hist(pts, bins=1000)
    # plt.plot(pts)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.title("reject sampling")
    plt.show()

if __name__ == '__main__':
    reject_sampling()