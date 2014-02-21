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
    model.budget = Set()
    
    #parameter
    model.sellPrice = Param(model.drug)
    model.grams = Param(model.drug)
    model.HRhours = Param(model.drug)
    model.EQhours = Param(model.drug)
    model.EQCost = Param(model.drug)
    
    model.materialCost = Param(model.material)
    model.materialContent = Param(model.material)
    model.budgeValue = Param(model.budget)
    
    #decision variables
    model.D = Var(model.drug, domain=NonNegativeReals)
    model.R = Var(model.material, domain=NonNegativeReals)
     
    
    #objective
    def objective_rule(model):
        return summation(model.sellPrice, model.D) -summation(model.EQCost, model.D) - summation(model.materialCost, model.R) 
    
    model.OBJ = Objective(rule=objective_rule, sense=maximize)
    
    #constraint
    def balance_constraint_rule(model):
        return summation(model.materialContent, model.R) - (model.grams, model.D) >=0
    
    def storage_constraint_rule(model):
        return summation(model.R) <= model.budgeValue['storage']
    
    def HR_constraint_rule(model):
        return summation(model.HRhour, model.D) <= model.budgeValue['HR']
        
    def EQ_constraint_rule(model):
        return summation(model.EQhour, model.D) <= model.budgeValue['hour']
    
    def money_constraint_rule(model):
        return summation(model.EQCost, model.D) + summation(model.materialCost, model.R) <=model.budgeValue['money']
    
        
    model.balanceConstraint = Constraint(rule=balance_constraint_rule)
    model.storageConstraint = Constraint(rule=storage_constraint_rule)
    model.HRConstraint = Constraint(rule=HR_constraint_rule)
    model.EQConstraint = Constraint(rule=EQ_constraint_rule)
    model.moneyConstraint = Constraint(rule=money_constraint_rule)

if __name__ == '__main__':
    pass