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
    Extension("Hello", ["hello.pyx"],
                        include_dirs = [np.get_include()]
             ),
    Extension("fibo", ["fibo.pyx"],
                    libraries=["calcul"],
                    library_dirs=["."],
                    ),
    ]
setup(
  name = 'Hello world app',
  ext_modules = cythonize(extensions),
) 
