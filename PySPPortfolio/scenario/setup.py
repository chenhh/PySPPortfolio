# -*- coding: utf-8 -*-
'''
Created on 2014/3/11

@author: chenhh
python setup.py build_ext --inplace

'''
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
              "Moment", 
              ["Moment.pyx"],
              libraries=["m"],
              include_dirs = [np.get_include()],
    ),
    Extension(
              "Copula", 
              ["Copula.pyx"],
              libraries=["m"],
              ibrary_dirs=[np.get_include()],
    ),
]

setup(
  name = 'Scenario generation',
  ext_modules = cythonize(extensions),
) 
