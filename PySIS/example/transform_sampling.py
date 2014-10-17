# -*- coding: utf-8 -*-
'''
.. codeauthor:: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
有些distrbution與其它distrubtion有函數的關系，也可以用此方法抽樣
e.g.
1 Z~U(0,1), then Z**2~chisquare(1)
2.U~chisquare(m), and V~chisquare(n), then F=(U/m)/(V/n)~F(m,n)
3.Z~U(0,1) and V~chisquare(n) indep. then T=Z/sqrt(V/n) ~t(n)
'''
__author__ = 'Hung-Hsin Chen'

import numpy as np
import matplotlib.pyplot as plt

def transform_sampling(n_sample=50000, theta=0.5):
    '''
    U, V ~ U(0,1) indep.
    then X=floor(1+ logV/log(1-(1-theta)**U)~Logarithmic(theta)

    log-dist pmf f(k,theta) = -1/ln(1-theta)* theta**k/k
    '''
    #sample
    U = np.random.rand(n_sample)
    V = np.random.rand(n_sample)
    X = np.floor(1.+ np.log(V)/np.log(1.-(1.-theta)**U))

    #function
    x = np.arange(10)+1
    y = -1/np.log(1-theta)*theta**x/x

    n, bins, patches = plt.hist(X, bins=100, normed=True)
    plt.title('X ~ Logarithmic(%s)'%(theta))
    plt.plot(x, y, 'r',  linewidth=1.5)
    plt.show()


if __name__ == '__main__':
    transform_sampling()