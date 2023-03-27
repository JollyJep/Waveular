import numpy as np
import CUDA_Calculation_Centre as ccc
import CPU_Calculation_Centre as cpu
# Define initialisation parameters
Points = np.zeros((3, 3, 3))
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
calc = ccc.CUDA_Calculations(Points, velocity, acceleration, k, sigma, l0, c, coord_change, ref_grid, divisor, 2.5 * 2.5, np.array([0, 0, -9.81], dtype=np.float64), True, deltaT, debug=True)  # Initialise GPU
calc_cpu = cpu.CPU_Calculations(Points, velocity, acceleration, k, sigma, l0, c, coord_change, ref_grid, divisor, 2.5 * 2.5, np.array([0, 0, -9.81], dtype=np.float64), True, deltaT, debug=True)   # Initialise CPU
pool_mass = 15   # Mass in Kilos
particle_number = 1 #Find weight for 1 particle
g = np.array([0, 0, -9.81]) # Standard earth gravitational acceleration
test_weight, _ = calc.weight(pool_mass, particle_number, g)
test_weight_cpu, _ = calc_cpu.weight(pool_mass, particle_number, g)
known_weight = np.array([0, 0, -147.15])    # Manually calculated weight
if np.isclose(test_weight, known_weight, rtol=0.01).all():
    print("GPU weight calculations nominal")
else:
    print("Error in GPU weight calculations")
    print(test_weight)
if np.isclose(test_weight_cpu, known_weight, rtol=0.01).all():
    print("CPU weight calculations nominal")
else:
    print("Error in CPU weight calculations")
    print(test_weight_cpu)