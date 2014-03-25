# -*- coding: utf-8 -*-
'''
Created on 2014/3/11

@author: Hung-Hsin Chen
python setup.py build_ext --inplace

'''
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
print np.get_include()
setup(
  name = 'HKW',
  ext_modules = cythonize("HKW.pyx")#                         include_path=[np.get_include()]),
) 
