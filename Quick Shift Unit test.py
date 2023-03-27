import numpy as np
import CUDA_Calculation_Centre as ccc
# Define point grid
Points = np.zeros((3, 3, 3))
Points[0][0] = np.array([1, 1, 0])
Points[0][1] = np.array([1, 2, 0])
Points[0][2] = np.array([1, 3, 0])
Points[1][0] = np.array([2, 1, 0])
Points[1][1] = np.array([2, 2, 0])
Points[1][2] = np.array([2, 3, 0])
Points[2][0] = np.array([3, 1, 0])
Points[2][1] = np.array([3, 2, 0])
Points[2][2] = np.array([3, 3, 0])
# Define initialisation parameters
ref_grid = np.full((3, 3), True)
velocity = np.zeros(np.shape(Points), dtype=np.float64)
acceleration = np.array([0, 0, -9.81], dtype=np.float64)
l0 = np.full(8, 0.5)
output = np.zeros(np.shape(Points), dtype=np.float64)
k = float(10)
sigma = 0
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
calc = ccc.CUDA_Calculations(Points, velocity, acceleration, k, sigma, l0, c, coord_change, ref_grid, divisor, 2.5 * 2.5, np.array([0, 0, -9.81], dtype=np.float64), True, deltaT, debug=True)  # Initialise CUDA class
expected_array = np.zeros((5, 5, 3))
expected_array[0, 0] = np.array([0, 0, 0])
expected_array[1, 0] = np.array([1, 0, 0])
expected_array[2, 0] = np.array([2, 0, 0])
expected_array[3, 0] = np.array([3, 0, 0])
expected_array[4, 0] = np.array([4, 0, 0])
expected_array[0, 1] = np.array([0, 1, 0])
expected_array[0, 2] = np.array([0, 2, 0])
expected_array[0, 3] = np.array([0, 3, 0])
expected_array[0, 4] = np.array([0, 4, 0])
expected_array[4, 1] = np.array([4, 1, 0])
expected_array[4, 2] = np.array([4, 2, 0])
expected_array[4, 3] = np.array([4, 3, 0])
expected_array[4, 4] = np.array([4, 4, 0])
expected_array[1, 4] = np.array([1, 4, 0])
expected_array[2, 4] = np.array([2, 4, 0])
expected_array[3, 4] = np.array([3, 4, 0])
test_array = calc.quick_shift(Points, coord_change, ref_grid, divisor, velocity)
if (test_array == expected_array).all():
    print("Quick shift working nominally")
else:
    print("Error in quick shift")
    print(test_array)

