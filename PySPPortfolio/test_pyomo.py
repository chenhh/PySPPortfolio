'''
Created on 2014/2/21

@author: chenhh
'''

from __future__ import division
from coopr.pyomo import *

def Pharmacy():
    model = AbstractModel()
    
    #set
    model.drug = Set()
    model.material = Set()
    
    #parameter
    model.sellPrice = Param(model.drug)
    model.grams = Param(model.drug)
    model.HRhours = Param(model.drug)
    model.EQhours = Param(model.drug)
    model.EQCost = Param(model.drug)
    
    model.materialCost = Param(model.material)
    model.materialContent = Param(model.material)
    
    #decision variables
    model.D = Var(model.drug, domain=NonNegativeReals)
    model.R = Var(model.material, domain=NonNegativeReals)
     
    
    #objective
    def objective_rule(model):
        summation(model.sellPrice, model.D) -summation(model.EQCost, model.D) - summation(model.materialCost, model.R) 
    
    model.OBJ = Objective(rule=objective_rule)
    
    #constraint
    def balance_constraint_rule(model):
        summation(model.materialContent, model.R) - (model.grams, model.D) >=0
        
    model.balanceConstraint = Constraint(rule=balance_constraint_rule)
    
    
    

if __name__ == '__main__':
    pass