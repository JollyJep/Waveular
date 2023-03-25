import numpy as np
import CUDA_Calculation_Centre as ccc
# Define initial positions
Points = np.zeros((3, 3, 3))
Points[0][0] = np.array([10, 10, 20])
Points[0][1] = np.array([10, 20, 10])
Points[0][2] = np.array([10, 30, 10])
# Define ref_grid and initial velocities
ref_grid = np.full((3, 3), True)
velocity = np.zeros(np.shape(Points), dtype=np.float64)
velocity[0][0] = np.array([10, 0, 10])
velocity[0][1] = np.array([0, 10, -10])
velocity[0][2] = np.array([5, 5, 0])
# Define constant acceleration
acceleration = np.array([0, 0, -9.81], dtype=np.float64)
# Define basic spring attributes just to allow CUDA_Calculation_Centre to initialise
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
calc = ccc.CUDA_Calculations(Points, velocity, acceleration, k, sigma, l0, c, coord_change, ref_grid, divisor, 2.5 * 2.5, np.array([0, 0, -9.81], dtype=np.float64), True, deltaT, debug=True)
for x in range(50): # Run velocity Verlet for 5 seconds in 0.1 second chunks
    calc.verlet(coord_change, True)
pos_expected = np.array([np.array([60, 10, -52.62]), np.array([10, 70, -162.6]), np.array([35, 55, -112.6])])   # Manually calculated final positions and velocities, using SUVAT
vel_expected = np.array([np.array([10, 0, -39.05]), np.array([0, 10, -59.05]), np.array([5, 5, -49.05])])
outputs = []
for n, test in enumerate(pos_expected): #Test against expected
    if np.isclose(test, calc.pos_grid[0, n], rtol=0.01).all():
        outputs.append("within 1%")
    else:
        outputs.append("not within 1%")
    if np.isclose(vel_expected[n], calc.velocity[0, n], rtol=0.01).all():
        outputs.append("within 1%")
    else:
        outputs.append("not within 1%")
print(f"\n----------------------------------------\nFor the first test, the position was {outputs[0]} of the predicted value, and the velocity was {outputs[1]} of the predicted value.\n\n"
      f"For the second test, the position was {outputs[2]} of the predicted value, and the velocity was {outputs[3]} of the predicted value.\n\n"
      f"For the third test, the position was {outputs[4]} of the predicted value, and the velocity was {outputs[5]} of the predicted value.\n\n----------------------------------------\n")