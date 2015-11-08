# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
              "moment_matching",
              ["moment_matching.pyx"],
              libraries=["m"],
              include_dirs = [np.get_include()],
    ),
#     Extension(
#               "Copula", 
#               ["Copula.pyx"],
#               libraries=["m"],
#               include_dirs = [np.get_include()],
# #               library_dirs=[np.get_include()],
#     ),
]

setup(
      name = 'Scenario generation',
      author = 'Hung-Hsin Chen',
      author_email = 'chenhh@par.cse.nsysu.edu.tw',
      ext_modules = cythonize(extensions),
) 
