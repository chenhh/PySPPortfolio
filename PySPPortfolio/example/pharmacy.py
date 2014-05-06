'''
Created on 2014/2/21

@author: chenhh
'''

from __future__ import division
from coopr.pyomo import *
from coopr.opt import SolverFactory
import time

def abstractPharmacy():
    t1 = time.time()

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
    
    model.materialCost = Param(model.material, mutable=True)
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
    # Create a solver
    opt = SolverFactory('cplex')
    
    data = DataPortal()
    data.load(filename='pharmacy.dat')
    instance = model.create(data)
    instance.pprint()
    results = opt.solve(instance)
    print results
    print "original elapsed %.3f secs"%(time.time()-t1)
    
    t2 = time.time()
    #change parameter and resolve
    getattr(instance, "materialCost")[2] = 199
    instance.preprocess()
    results = opt.solve(instance)
    print results
 
 
    print "resolve, elapsed %.3f secs"%(time.time()-t2)
    
    
    
def concretePharmacy():
    t1 = time.time()
    
    model = ConcreteModel()
    
    model.drug=[1,2]
    model.material = [1,2]
    model.budget = ["money", "HR", "hour", "storage"]
    #set
#     model.drug = Set(initialize=[1,2])
#     model.material = Set(initialize=[1,2])
#     model.budget = Set(initialize=["money", "HR", "hour", "storage"])
    
    #parameter
#     model.sellPrice = Param(model.drug, initialize={1:6200, 2:6900})
    model.sellPrice = Param(model.drug, mutable=True)
    model.sellPrice[1] = 6200
    model.sellPrice[2] = 6900
    model.grams = Param(model.drug, initialize={1:0.5, 2:0.6})
    model.HRhours = Param(model.drug, initialize={1:90, 2:100})
    model.EQhours = Param(model.drug, initialize={1:40, 2:50})
    model.EQCost = Param(model.drug, initialize={1:700, 2:800})
    
    model.materialCost = Param(model.material, initialize={1:100, 2:199.9},
                               mutable=True)
    model.materialContent = Param(model.material, initialize={1:0.01, 2:0.02})
    model.budgeValue = Param(model.budget, initialize={"money":1e5, "HR":2000, 
                                            "hour":800, "storage":1000})
    
    #decision variables
    model.D = Var(model.drug, domain=NonNegativeReals)
    model.R = Var(model.material, domain=NonNegativeReals)
     
    #objective
    def objective_rule(model):
        return summation(model.sellPrice, model.D) - \
               summation(model.EQCost, model.D) - \
               summation(model.materialCost, model.R) 
    
    model.Obj = Objective(rule=objective_rule, sense=maximize)
    
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
    
    # Create a solver
    opt = SolverFactory('cplex')
    print "opt options:", opt.options
#     opt.options["threads"] = 4
    instance = model.create()
    results = opt.solve(instance)
    print type(instance)
    print dir(instance)
    print results
    print 
#     print results.Solution.Objective.x1.Value
#     print results.Solver.Status
#     print results.Solution.Status
#     print type(results)
#     print dir(results)
    
    instance.load(results)
    
    #print variable method 1
    print instance.D[1].value
    print instance.D[2].value
    print instance.R[1].value
    print instance.R[2].value
    print "obj:", results.Solution.Objective.__default_objective__['value']
    
    #print variable method 2
#     for var in instance.active_components(Var):
#         varobj = getattr(instance, var)
#         for idx in varobj:
#             print varobj[idx], varobj[idx].value
    
    display(instance)
    #output file to yaml format
#     results.write()
    print "original elapsed %.3f secs"%(time.time()-t1)
    
#     t2 = time.time()
#     #change parameter and resolve
#     getattr(instance, "materialCost")[2] = 199
#     instance.preprocess()
#     results = opt.solve(instance, tee=True)
#     print results
#     print "solver status:", str(results.Solution.Status)
#   
#  
#     print "resolve, elapsed %.3f secs"%(time.time()-t2)
    
if __name__ == '__main__':
#     abstractPharmacy()
    concretePharmacy()
