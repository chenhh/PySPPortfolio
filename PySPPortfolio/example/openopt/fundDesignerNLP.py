# -*- coding: utf-8 -*-
'''
Created on 2014/3/26
min (x-1)^2 + (y-2)^2 + (z-3)^4 
s.t. 
    y > 5
    4x-5z < -1
    (x-10)^2 + (y+1)^2 < 50

'''

from FuncDesigner import *
from openopt import NLP
x,y,z = oovars('x', 'y', 'z')
f = (x-1)**2 + (y-2)**2 + (z-3)**4
startPoint = {x:0, y:0, z:0}
constraints = [y>5, 4*x-5*z<-1, (x-10)**2 + (y+1)**2 < 50] 
p = NLP(f, startPoint, constraints = constraints)
r = p.solve('ralg')
x_opt, y_opt, z_opt = r(x,y,z)
print(x_opt, y_opt, z_opt) # x=6.25834212, y=4.99999936, z=5.2066737