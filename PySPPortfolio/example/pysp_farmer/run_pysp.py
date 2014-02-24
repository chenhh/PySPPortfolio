'''
Created on 2014/2/24

@author: chenhh
'''
import subprocess
import os

def runFarmer():
   
#     subprocess.call(['runef', '-m models', '-i nodedata', '--solver=cplex', '--solve'])
    os.system("runef -m models -i nodedata --solver=glpk --solve")

if __name__ == '__main__':
    runFarmer()