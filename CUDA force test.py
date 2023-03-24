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
velocity = np.zeros(np.shape(Points), dtype=np.float64)
acceleration = np.zeros(np.shape(Points), dtype=np.float64)
velocity[1][1] = np.array(([1, 0, 0]))
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
divisor = np.array([divisor_w, divisor_h])
coord_change = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]), np.array([1, 1]), np.array([-1, 1]), np.array([-1, -1]), np.array([1, -1])])
calc = ccc.CUDA_Calculations(Points, velocity, acceleration, k, sigma, l0, c, coord_change, ref_grid, divisor, 2.5 * 2.5, np.array([0, 0, 0], dtype=np.float64), True, deltaT, debug=True)
position, energies = calc.runner(coord_change)
expected_values_resultant_force = np.array([-6.167, 0, -54.318]) #Resultant force on the central particle as found manually
outputs = []
for n, direction in enumerate(expected_values_resultant_force):
    if np.isclose(direction, calc.resultant_force_gpu[1, 1, n], rtol=0.01):
        outputs.append("within 1%")
    else:
        outputs.append("not within 1%")
print(f"\n----------------------------------------\nFor the x direction, the simulated forces are {outputs[0]}, with a value of {calc.resultant_force_gpu[1, 1, 0]} as compared to the predicted value of {expected_values_resultant_force[0]}. This gives a percentage difference of {(calc.resultant_force_gpu[1, 1, 0] -expected_values_resultant_force[0])/ expected_values_resultant_force[0] * 100}%.\n\n"
      f"For the y direction, the simulated forces are {outputs[1]}, with a value of {calc.resultant_force_gpu[1, 1, 1]} as compared to the predicted value of {expected_values_resultant_force[1]}. This gives a percentage difference of {(calc.resultant_force_gpu[1, 1, 1] -expected_values_resultant_force[1])/ expected_values_resultant_force[1] * 100}%.\n\n"
      f"For the z direction, the simulated forces are {outputs[2]}, with a value of {calc.resultant_force_gpu[1, 1, 2]} as compared to the predicted value of {expected_values_resultant_force[2]}. This gives a percentage difference of {(calc.resultant_force_gpu[1, 1, 2] -expected_values_resultant_force[2])/ expected_values_resultant_force[2] * 100}%.\n\n----------------------------------------\n")

