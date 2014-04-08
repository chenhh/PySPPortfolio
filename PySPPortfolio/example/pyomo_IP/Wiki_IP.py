# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
integer programming exmaple from Wikipedia
http://en.wikipedia.org/wiki/Integer_programming


max y
-x +y <=1
3x+2y <=12
2x+3y <=12
x, y>=0
x, y is integer
'''

from coopr.pyomo import *
from coopr.opt import  SolverFactory
import time

def IP():
    
    t1 = time.time()
    model = ConcreteModel()

    #decision variable
    model.x = Var(within=NonNegativeIntegers)
    model.y = Var(within=NonNegativeIntegers)
    
    #constraint
    def rule1(model):
        return (-model.x + model.y) <=1
    
    model.c1 = Constraint(rule=rule1)

    def rule2(model):
        return (3 * model.x +2* model.y) <=12
    
    model.c2 = Constraint(rule=rule2)
    
    def rule3(model):
        return (2*model.x + 3*model.y) <=12
    
    model.c3 = Constraint(rule=rule3)
    
    #objective
    def obj_rule(model):
        return model.y
    
    model.obj = Objective(sense=maximize)
    
    #optimizer
    opt = SolverFactory('glpk')
    
    instance = model.create()
    results = opt.solve(instance)
    print type(results)
    print results
    
    instance.load(results)
    
    display(instance)
#     maxy = results.Solution.Objective.__default_objective__['value']

#     print "maxy:", maxy
   
    print "elapsed %.3f secs"%(time.time()-t1)


if __name__ == '__main__':
    IP()