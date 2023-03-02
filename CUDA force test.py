import CUDA_Calculation_Centre as ccc
import CPU_Calculation_Centre as cpu
import numpy as np
from numba import cuda
import numba
print(cuda.gpus)
cuda.detect()


Points = np.zeros((3, 3, 3))
Points[0][0] = np.array([1, 1, 0])
Points[0][1] = np.array([1, 2, 0])
Points[0][2] = np.array([1, 3, 0])
Points[1][0] = np.array([2, 1, 0])
Points[1][1] = np.array([2.1, 2, 1])
Points[1][2] = np.array([2, 3, 0])
Points[2][0] = np.array([3, 1, 0])
Points[2][1] = np.array([3, 2, 0])
Points[2][2] = np.array([3, 3, 0])
ref_grid = np.full((3, 3), True)
velocity = np.zeros((3, 3, 3))
velocity[1][1] = np.array(([1, 0, 0]))
l0 = 0.5
output = np.zeros(np.shape(Points))
k = float(10)
blockdim = (3, 3)
griddim = (len(Points) // blockdim[0], len(Points[0]) // blockdim[1])
print(griddim)
vector_difference = np.zeros(3)
modulus = np.array([0.0])
Force_store = np.zeros(3)
cuda_test = cpu.CPU_Calculations()
width = 3
height = 3
x_scale = 3
y_scale = 3
c = 2
divisor_w = 1/width * x_scale
divisor_h = 1 / height * y_scale
divisor = np.array([divisor_w, divisor_h])
coord_change = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]), np.array([1, 1]), np.array([-1, 1]), np.array([-1, -1]), np.array([1, -1])])
cuda_test.runner(output, Points, k, l0, blockdim, griddim, vector_difference, modulus, Force_store, ref_grid, width, height, coord_change, divisor, velocity, c)
