# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen

'''

from FuncDesigner import *
from openopt import LP
from time import time

t = time() 
# Define some oovars
drugs = oovar(2)
material = oovar(2)
budgets = oovar(3)

 
# Let's define some linear functions
f1 = 4*x+5*y + 3*z + 5
f2 = f1.sum() + 2*x + 4*y + 15
f3 = 5*f1 + 4*f2 + 20
 
# Define objective; sum(a) and a.sum() are same as well as for numpy arrays
obj = x.sum() + y - 50*z + sum(f3) + 2*f2.sum() + 4064.6
 
# Define some constraints via Python list or tuple or set of any length, probably via while/for cycles
constraints = [x+5*y<15, x[0]<4, f1<[25, 35], f1>-100, 2*f1+4*z<[80, 800], 5*f2+4*z<100, -5<x,  x<1, -20<y,  y<20, -4000<z, z<4]
 
# Start point - currently matters only size of variables, glpk, lpSolve and cvxopt_lp use start val = all-zeros
startPoint = {x:[8, 15], y:25, z:80} # however, using numpy.arrays is more recommended than Python lists
 
# Create prob
p = LP(obj, startPoint, constraints = constraints)
 
# Solve
r = p.solve('glpk') # glpk is name of solver involved, see OOF doc for more arguments
 
# Decode solution
print('Solution: x = %s   y = %f  z = %f' % (str(x(r)), y(r), z(r)))
# Solution: x = [-4.25 -4.25]   y = -20.000000  z = 4.000000
print "elapsed %.3f secs"%(time() - t)