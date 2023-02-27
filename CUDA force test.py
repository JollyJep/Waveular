import CUDA_Calculation_Centre as ccc
import numpy as np
from numba import cuda
import numba
print(cuda.gpus)


Points = np.array([np.array([1,1.5]), np.array([1, 3])])
Shifted_points = np.array([np.array([1, 2.1]), np.array([1.1, 3.7])])
l0 = 0.5
output = np.zeros(np.shape(Points))
k = float(10)
blockdim = (32)
griddim = (Points.shape[0] // blockdim)
vector_difference = np.zeros(np.shape(Points[0]))
modulus = np.array([0.0])
Force_store = np.zeros(np.shape(Points[0]))
cuda_test = ccc.CUDA_Calculations()
cuda_test.runner(output, Points, Shifted_points, k, l0, blockdim, griddim, vector_difference, modulus, Force_store)
