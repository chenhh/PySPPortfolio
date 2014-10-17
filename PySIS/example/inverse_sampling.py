# -*- coding: utf-8 -*-
'''
.. codeauthor:: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
如果target pdf的inverse cdf analytic form已知，可用inverse cdf抽出樣本,
抽出的樣本分佈即為target pdf
'''
__author__ = 'Hung-Hsin Chen'

import numpy as np
import matplotlib.pyplot as plt

def inverse_sampling(n_sample=10000):
    '''
    target pdf: f(x)=3*x**2, 0<x<1
    its cdf: F(x)=x**3
    inverse cdf G(u) = u**(1/3)
    '''
    #sample
    pts = np.random.rand(n_sample)
    samples = pts**(1./3)

    #function
    x = np.linspace(0, 1, n_sample)
    y = 3*x**2

    n, bins, patches = plt.hist(samples, bins=100, normed=True)
    # print n, bins, patches
    plt.setp(patches, 'facecolor', 'g')
    plt.title('f(x)=3*x**2')
    plt.plot(x, y, 'r',  linewidth=1.5)
    plt.show()


def inverse_sampling2(exp_lambda, n_sample=10000):
    '''
    target pdf: X~Exp(lambda) = lambda* e**(−lambda*x)
    its cdf F(x) = 1- e**(-lambda*x)
    inverse cdf G(u = -1/lambda* log(1-u)
    '''
    #sample
    pts = np.random.rand(n_sample)
    samples = -np.log(1-pts)/exp_lambda

    #function
    x = np.linspace(0, 10, n_sample)
    y = exp_lambda * np.exp(-exp_lambda*x)

    n, bins, patches = plt.hist(samples, bins=100, normed=True)
    # print n, bins, patches
    plt.setp(patches, 'facecolor', 'g')
    plt.title('X ~ Exp(%s)'%(exp_lambda))
    plt.plot(x, y, 'r',  linewidth=1.5)
    plt.show()

if __name__ == '__main__':
    # inverse_sampling()
    inverse_sampling2(2)