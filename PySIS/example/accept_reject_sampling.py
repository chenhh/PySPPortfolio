# -*- coding: utf-8 -*-
'''
.. codeauthor:: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>

target pdf X~f(t)
user-specified pdf Y~g(t)
f(t)/g(t) <=c, forall t, s.t. f(t) > 0

1. 找一個可簡單生成的隨機變數Y, 其pdf g滿足f(t)/g(t) <=c, forall t, s.t. f(t) > 0
2. 從g中生成一個隨機數y
3. 從U(0, 1)生成一個隨機數u
4. 若 u < f(y)/(cg(y)), accept x=y, else reject y, and goto step 2

使用此方法必須知道
1. f(t)的support
2. c最少要取多少才能使得c*g(t) >= f(t)
3. f(t)的analytic form
'''
__author__ = 'Hung-Hsin Chen'

import numpy as np
import matplotlib.pyplot as plt

def AR_sampling(n_sample=10000):
    '''
    target function f(x) = 6*x*(1-x), 0<x<1
    '''
    #samples
    count = 0
    accepts = []

    while count <= n_sample:
        u = np.random.rand()
        y = np.random.rand()
        if y*(1-y) > u:
            accepts.append(y)
            count+=1

    #functions
    x = np.linspace(0, 1, n_sample)
    y = 6*x*(1-x)

    #plots
    plt.title('f(x)=6*x*(1-x)')
    n, bins, patches = plt.hist(accepts, bins=100, normed=True)
    plt.setp(patches, 'facecolor', 'g')
    plt.plot(x, y, 'r',  linewidth=1.5)
    plt.show()

if __name__ == '__main__':
    AR_sampling()