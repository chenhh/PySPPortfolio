import os
import platform

FileDir = os.path.abspath(os.path.curdir)
PklBasicFeaturesDir = os.path.join(FileDir,'pkl', 'BasicFeatures')
osType = platform.uname()[0]
if  osType == 'Linux':
    ExpResultsDir =  os.path.join('/', 'home', 'chenhh' , 'Dropbox', 
                                  'financial_experiment', 'PySPPortfolio')
    
elif osType =='Windows':
    ExpResultsDir= os.path.join('C:\\', 'Dropbox', 'financial_experiment', 
                                'PySPPortfolio')  