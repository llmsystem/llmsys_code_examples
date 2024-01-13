import ctypes
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Load the shared library
cur_path = os.getcwd()
lib = ctypes.CDLL(os.path.join(cur_path, "window_sum.so"))

in_array = np.array([i+1 for i in range(12)], dtype=np.float32)
simple_array = np.zeros(8, dtype=np.float32)
shared_array = np.zeros(8, dtype=np.float32)

print(f"Input: {in_array}")
win_temp = sliding_window_view(in_array, 5)
np_res = np.sum(win_temp, axis=1)
print(f"Numpy window sum: {np_res}")

# Define argtypes and returntypes of the C function
lib.WindowSumSimple.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

lib.WindowSumSimple.restype = None

lib.WindowSumShared.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
]

lib.WindowSumShared.restype = None

# Load the arrays to CUDA device
in_gpu = gpuarray.to_gpu(in_array)
simple_out_gpu = gpuarray.to_gpu(simple_array)
shared_out_gpu = gpuarray.to_gpu(shared_array)

# Call the C wrapper function with CUDA kernel
lib.WindowSumSimple(
    ctypes.cast(in_gpu.ptr, ctypes.POINTER(ctypes.c_float)),
    ctypes.cast(simple_out_gpu.ptr, ctypes.POINTER(ctypes.c_float)),
    ctypes.c_int(len(simple_array)),
)

# Load the gpuarray back to array in the host device
simple_array = simple_out_gpu.get()
print(f"GPU simple window sum: {simple_array}")

# Call the C wrapper function with CUDA kernel
lib.WindowSumShared(
    ctypes.cast(in_gpu.ptr, ctypes.POINTER(ctypes.c_float)),
    ctypes.cast(shared_out_gpu.ptr, ctypes.POINTER(ctypes.c_float)),
    ctypes.c_int(len(in_gpu)),
    ctypes.c_int(len(shared_array)),
)

# Load the gpuarray back to array in the host device
shared_array = shared_out_gpu.get()
print(f"GPU shared window sum: {shared_array}")