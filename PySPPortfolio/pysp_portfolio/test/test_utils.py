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
    print "fun2:", kwargs

if __name__ == '__main__':
    fun1(2, alpha=0.5)