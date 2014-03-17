# -*- coding: utf-8 -*-
import pyopencl as cl
from pyopencl import array
import numpy

if __name__ == "__main__":
    vector = numpy.zeros((1, 1), cl.array.vec.float4)
    matrix = numpy.zeros((1, 4), cl.array.vec.float4)
    matrix[0, 0] = (1, 2, 4, 8)
    matrix[0, 1] = (16, 32, 64, 128)
    matrix[0, 2] = (3, 6, 9, 12)
    matrix[0, 3] = (5, 10, 15, 25)
    vector[0, 0] = (1, 2, 4, 8)
    
    ## Step #1. Obtain an OpenCL platform.
    platform = cl.get_platforms()[0]
    
    ## It would be necessary to add some code to check the check the support for
    ## the necessary platform extensions with platform.extensions
    
    ## Step #2. Obtain a device id for at least one device (accelerator).
    device = platform.get_devices()[0]
    
    ## It would be necessary to add some code to check the check the support for
    ## the necessary device extensions with device.extensions
    
    ## Step #3. Create a context for the selected device.
    context = cl.Context([device])
    
    ## Step #4. Create the accelerator program from source code.
    ## Step #5. Build the program.
    ## Step #6. Create one or more kernels from the program functions.
    program = cl.Program(context, """
        __kernel void matrix_dot_vector(__global const float4 *matrix,
        __global const float4 *vector, __global float *result)
        {
          int gid = get_global_id(0);
          result[gid] = dot(matrix[gid], vector[0]);
        }
        """).build()
    
    ## Step #7. Create a command queue for the target device.
    queue = cl.CommandQueue(context)
    
    ## Step #8. Allocate device memory and move input data from the host to the device memory.
    mem_flags = cl.mem_flags
    matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
    vector_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=vector)
    matrix_dot_vector = numpy.zeros(4, numpy.float32)
    destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, matrix_dot_vector.nbytes)
    
    ## Step #9. Associate the arguments to the kernel with kernel object.
    ## Step #10. Deploy the kernel for device execution.
    program.matrix_dot_vector(queue, matrix_dot_vector.shape, None, matrix_buf, vector_buf, destination_buf)
    
    ## Step #11. Move the kernel’s output data to host memory.
    cl.enqueue_copy(queue, matrix_dot_vector, destination_buf)
    
    ## Step #12. Release context, program, kernels and memory.
    ## PyOpenCL performs this step for you, and therefore,
    ## you don't need to worry about cleanup code
    
    print(matrix_dot_vector)