# -*- coding: utf-8 -*-
'''
.. codeauthor:: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
Monte Carlo技術主要是用來求函數ｇ(.)的期望值
'''
__author__ = 'Hung-Hsin Chen'

import numpy as np

def MC_example(n_sample=10000):
    '''
    求\int_0^1 e**(-x) dx之值
    '''
    x = np.random.rand(n_sample)
    value = (np.exp(-x)).mean()

    print "MC:", value
    print "target:", (1-np.exp(-1))


def MC_example2(n_sample=10000):
    '''
    求\int_2^4 e**(-x) dx
    '''
    x = np.random.rand(n_sample)*2+2
    value = (np.exp(-x)).mean() *2

    print "MC:", value
    print "target:", (np.exp(-2)-np.exp(-4))

def MC_example3(n_sample=10000):
    '''
    f(x) = \int_{-\infty}^{x} 1/sqrt(2\pi) e^{-t^2/2} dt
    因為積分無下界，所以要分成x>0與x<=0處理.

    x>0時，只要處理 \int_0^x e^{-t^2/2} dt即可。
    因為樣本從U(0,x）取出，但x會變動。
    所以用變數變換y=t/x，改求\int_0^1 xe^{-(xy)^2} dy
    '''
    #x>0


if __name__ == '__main__':
    # MC_example()
    MC_example2()