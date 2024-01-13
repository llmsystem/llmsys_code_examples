import ctypes
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import os
import numpy as np

# Load the shared library
cur_path = os.getcwd()
lib = ctypes.CDLL(os.path.join(cur_path, "vector_add.so"))

size = 10
a = np.random.randint(1, 10, size, dtype=np.int32)
b = np.random.randint(1, 10, size, dtype=np.int32)
ccpu = np.zeros(size, dtype=np.int32)
cgpu = np.zeros(size, dtype=np.int32)

print(f"Input a: {a}\nInput b: {b}")

print(f"Numpy add: {a + b}, {type(a+b)}")

# Define argtypes and returntypes of the C function
lib.VecAddCPU.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]

lib.VecAddCPU.restype = None

lib.VecAddCUDA.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

lib.VecAddCUDA.restype = None

# Call the C function
lib.VecAddCPU(a, b, ccpu, size)

print(f"CPU add: {ccpu}, {type(ccpu)}")

# Load the arrays to CUDA device
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.to_gpu(cgpu)

# Call the C wrapper function with CUDA kernel
lib.VecAddCUDA(
    ctypes.cast(a_gpu.ptr, ctypes.POINTER(ctypes.c_int)),
    ctypes.cast(b_gpu.ptr, ctypes.POINTER(ctypes.c_int)),
    ctypes.cast(c_gpu.ptr, ctypes.POINTER(ctypes.c_int)),
    ctypes.c_int(size)
)

print(f"GPU add: {c_gpu}, {type(c_gpu)}")
# Load the gpuarray back to array in the host device
cgpu = c_gpu.get()
print(f"After offload: {cgpu}, {type(cgpu)}")



lib.VecAddCUDA2.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]

lib.VecAddCUDA2.restype = None
cgpu2 = np.zeros(size, dtype=np.int32)
lib.VecAddCUDA2(a, b, cgpu2, size)
print(f"GPU add2: {cgpu2}, {type(cgpu2)}")


