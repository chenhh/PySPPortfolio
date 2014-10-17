# -*- coding: utf-8 -*-
'''
.. codeauthor:: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
importance fucntion (user-specified) pdf X~f(x),
target pdf g(x), which satisfied {x: g(x) >0} and f(x) >0



'''
__author__ = 'Hung-Hsin Chen'

import numpy as np

def importance_sampling(n_sample=10000):
    '''
    \int_0^1 e^{-x}/(1+x^2) dx

    importance function:
    f0(x) = 1,                       0<x<1
    f1(x) = e^{-x},                  0 < x < \infty
    f2(x) = 1/\pi(1+x^2),            -\infty < x < \infty
    f3(x) = e^{-x} (1-e^{-1})^{-1},  0 < x <1
    f4(x) = 4/\pi(1+x^2),            0 < x < 1
    '''
    values = np.zeros(5)
    stds = np.zeros(5)
    #target function
    g = lambda x: np.exp(-x - np.log(1+x**2))*(x>0)*(x<1)

    #f0
    x = np.random.rand(n_sample)
    fg = g(x)
    values[0] = fg.mean()
    stds[0] = fg.std()

    #f1
    x = np.random.exponential(size=n_sample)
    fg = g(x)/np.exp(-x)
    values[1] = fg.mean()
    stds[1] = fg.std()

    #f2
    x=np.random.standard_cauchy(size=n_sample)
    for idx,val in enumerate(x):
        if val > 1 or val < 0:
            x[idx] = 2
    fg =  g(x)/(1/(np.pi*(1+x*x)))
    values[2] = fg.mean()
    stds[2] = fg.std()


    print values
    print stds



if __name__ == '__main__':
    importance_sampling()