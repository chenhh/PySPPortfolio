# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

'''
def say_hello_to(name):
    print("Hello %s!" % name)


def f(double x):
    '''static type in argument,
    但是傳回值仍為python float type
    '''
    return x**2-x


cpdef double f2(double x) except? -2:
    '''
    如果開頭以cdef定義時，此函數只能被cython呼叫，
    如果要讓python也能呼叫此函數，需以cpdef開頭。
    The except? -2 means that an error will be checked for if -2 is returned 
    '''

    return x**2-x


def integrate_f(double a, double b, int N):
    '''
    因為i為c type integer, 所以for loop會compile成C lang的loop
    
    '''
    cdef int i
    cdef double s, dx
    s = 0
    dx = (b-a)/N
    for i in range(N):
        s += f(a+i*dx)
    return s * dx

