# -*- coding: utf-8 -*-

import time
import numpy as np
import pyopencl as cl

n_sample = 50000
a_np = np.random.rand(n_sample).astype(np.float32)
b_np = np.random.rand(n_sample).astype(np.float32)

t1 = time.time()
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
__kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()
print "build time %f secs"%(time.time()-t1)

t2 = time.time()
mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)
print "compute time %f secs"%(time.time()-t2)

# Check on CPU with Numpy:
t3 = time.time()
print(res_np - (a_np + b_np))
print "cpu time %f secs"%(time.time()-t3)
print(np.linalg.norm(res_np - (a_np + b_np)))