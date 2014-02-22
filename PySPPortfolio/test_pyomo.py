'''
Created on 2014/2/21

@author: chenhh
'''

from __future__ import division
from coopr.pyomo import *
from coopr.opt import SolverFactory

def abstractPharmacy():
    # Create a solver
    
    opt = SolverFactory('cplex')
    
    #build model
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
        return summation(model.sellPrice, model.D) - \
               summation(model.EQCost, model.D) - \
               summation(model.materialCost, model.R) 
    
    model.Objective = Objective(rule=objective_rule, sense=maximize)
    
    #constraint
    def balance_constraint_rule(model):
        return summation(model.materialContent, model.R) - \
            summation(model.grams, model.D) >= 0
    
    def storage_constraint_rule(model):
        return summation(model.R) <= model.budgeValue['storage']
    
    def HR_constraint_rule(model):
        return summation(model.HRhours, model.D) <= model.budgeValue['HR']
        
    def EQ_constraint_rule(model):
        return summation(model.EQhours, model.D) <= model.budgeValue['hour']
    
    def money_constraint_rule(model):
        return (summation(model.EQCost, model.D) + 
                summation(model.materialCost, model.R)) <=model.budgeValue['money']
    
    model.balanceConstraint = Constraint(rule=balance_constraint_rule)
    model.storageConstraint = Constraint(rule=storage_constraint_rule)
    model.HRConstraint = Constraint(rule=HR_constraint_rule)
    model.EQConstraint = Constraint(rule=EQ_constraint_rule)
    model.moneyConstraint = Constraint(rule=money_constraint_rule)

    # Create a model instance and optimize
    data = DataPortal()
    data.load(filename='pharmacy.dat')
    instance = model.create(data)
#     model.pprint()
    results = opt.solve(instance)
    print results

def concretePharmacy():
    model = ConcreteModel()
    

if __name__ == '__main__':
    abstractPharmacy()