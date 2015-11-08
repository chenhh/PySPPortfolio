# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

import os
import platform

PROJECT_DIR = os.path.abspath(os.path.curdir)
PKL_BASIC_FEATURES_DIR = os.path.join(PROJECT_DIR,'pkl', 'BasicFeatures')
os_type = platform.uname()[0]
if  os_type == 'Linux':
    EXP_RESULT_DIR =  os.path.join('/', 'home', 'chenhh' , 'Dropbox',
                                  'financial_experiment', 'pysp_portfolio')

elif os_type =='Windows':
     EXP_RESULT_DIR = os.path.join('C:\\', 'Dropbox', 'financial_experiment',
                                'pysp_portfolio')
else:
    raise ValueError('unknown os platform:{}'.format(os_type))

__all__ = ['PROJECT_DIR', 'PKL_BASIC_FEATURES_DIR', 'EXP_RESULT_DIR']
