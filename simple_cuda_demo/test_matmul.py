import ctypes
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import os
import numpy as np

# Load the shared library
cur_path = os.getcwd()
lib = ctypes.CDLL(os.path.join(cur_path, "matmul.so"))

m, n, p = 4, 4, 2
np.random.seed(0)
a = np.random.randint(1, 3, [m, n]).astype(np.float32)
b = np.random.randint(1, 3, [n, p]).astype(np.float32)
cgpu = np.zeros([m, p], dtype=np.float32)
cgputile = np.zeros([m, p], dtype=np.float32)

print(f"Input a: {a}\nInput b: {b}")

print(f"Numpy matmul: {a @ b}, {type(a@b)}")

# Define argtypes and returntypes of the C function
lib.Matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

lib.Matmul.restype = None

# Load the arrays to CUDA device
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.to_gpu(cgpu)

# Call the C wrapper function with CUDA kernel
lib.Matmul(
    ctypes.cast(a_gpu.ptr, ctypes.POINTER(ctypes.c_float)),
    ctypes.cast(b_gpu.ptr, ctypes.POINTER(ctypes.c_float)),
    ctypes.cast(c_gpu.ptr, ctypes.POINTER(ctypes.c_float)),
    ctypes.c_int(m),
    ctypes.c_int(n),
    ctypes.c_int(p)
)

print(f"GPU matmul: {c_gpu}, {type(c_gpu)}")
# Load the gpuarray back to array in the host device
cgpu = c_gpu.get()
print(f"After offload: {cgpu}, {type(cgpu)}")

# Compare result
ccpu = a @ b
print(f"Compare result: {np.linalg.norm(ccpu - cgpu)}")