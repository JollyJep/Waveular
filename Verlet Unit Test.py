import numpy as np
import CUDA_Calculation_Centre as ccc
Points = np.zeros((3, 3, 3))
Points[0][0] = np.array([10, 10, 20])
Points[0][1] = np.array([10, 20, 10])
Points[0][2] = np.array([10, 30, 10])
Points[1][0] = np.array([20, 10, 30])
Points[1][1] = np.array([21, 20, 10])
Points[1][2] = np.array([20, 30, 10])
Points[2][0] = np.array([30, 10, 20])
Points[2][1] = np.array([30, 20, 40])
Points[2][2] = np.array([30, 30, 10])
ref_grid = np.full((3, 3), True)
velocity = np.zeros(np.shape(Points), dtype=np.float64)
velocity[0][0] = np.array([10, 0, 10])
velocity[0][1] = np.array([0, 10, 10])
velocity[0][2] = np.array([5, 5, 10])
velocity[1][0] = np.array([5, 1, 0])
velocity[1][1] = np.array([2, 2, -10])
velocity[1][2] = np.array([7, 0, 0])
velocity[2][0] = np.array([0, 7, 0])
velocity[2][1] = np.array([-1, 0, 0])
velocity[2][2] = np.array([0, -5, 0])
acceleration = np.zeros(np.shape(Points), dtype=np.float64)
l0 = np.full(8, 0.5)
output = np.zeros(np.shape(Points), dtype=np.float64)
k = float(10)
sigma = 0
blockdim = (3, 3)
griddim = (len(Points) // blockdim[0], len(Points[0]) // blockdim[1])
vector_difference = np.zeros(np.shape(output))
modulus = np.zeros((len(output), len(output[0])))
Force_store = np.zeros(np.shape(output))
width = 3
height = 3
x_scale = 3
y_scale = 3
c = 2
deltaT = 0.1
divisor_w = 1/width * x_scale
divisor_h = 1 / height * y_scale
divisor = np.array([divisor_w, divisor_h, 1])
coord_change = np.array([np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]), np.array([1, 1, 0]), np.array([-1, 1, 0]), np.array([-1, -1, 0]), np.array([1, -1, 0])])
calc = ccc.CUDA_Calculations(Points, velocity, acceleration, k, sigma, l0, c, coord_change, ref_grid, divisor, 2.5 * 2.5, np.array([0, 0, -9.81], dtype=np.float64), True, deltaT, debug=True)
for x in range(50):
    calc.verlet(coord_change, True)

