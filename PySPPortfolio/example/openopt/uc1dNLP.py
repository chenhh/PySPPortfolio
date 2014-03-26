# -*- coding: utf-8 -*-
'''
Unconstrained one-dimensional nonlinear problem
'''

from openopt import NLP

#solve f(x)=(x-1)**2, start point at x=4
p = NLP(lambda x: (x-1)**2, 4)

#ralg solver
r = p.solve('ralg')
print('optim point: coordinate=%f  objective function value=%f' % (r.xf, r.ff))