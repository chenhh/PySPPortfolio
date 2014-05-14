# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

'''

from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free

cdef Py_ssize_t idx, i, n = 100
cdef int * local_buf
cdef size_t size = 10

with nogil, parallel():
    local_buf = <int *> malloc(sizeof(int) * size)
    if local_buf == NULL:
        abort()

    # populate our local buffer in a sequential loop
    for i in xrange(size):
        local_buf[i] = i * 2

    # share the work using the thread-local buffer(s)
    for i in prange(n, schedule='guided'):
        func(local_buf)

    free(local_buf)

if __name__ == '__main__':
    pass