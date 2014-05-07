# -*- coding: utf-8 -*-
'''

@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

python setup.py build_ext --inplace

'''
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
              "CPerformance", 
              ["cPerformance.pyx"],
              libraries=["m"],
              include_dirs = [np.get_include()],
    ),

]

setup(
      name = 'statistics',
      author = 'Hung-Hsin Chen',
      author_email = 'chenhh@par.cse.nsysu.edu.tw',
      ext_modules = cythonize(extensions),
) 
