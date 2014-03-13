# -*- coding: utf-8 -*-
'''
Created on 2014/3/11

@author: chenhh
python setup.py build_ext --inplace

'''
from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Hello world app',
  ext_modules = cythonize("hello.pyx"),
)