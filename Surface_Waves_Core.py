import numpy as np
import scipy
import numba
from numba import njit
from numba import cuda
import pandas as pd
from matplotlib import pyplot as plt
import Pool_Simulation as Pool_sim
import Grid_Creation_System as gcs
import CUDA_Calculation_Centre as ccc
import CPU_Calculation_Centre as cpu

def Pool_Simulation_Setup(shape="circular", x_dim=10, y_dim=10, z_dim=1, viscosity=1, density=1):
    Pool_boundaries = Pool_sim.Pool_Simulation(shape=shape, x_size=x_dim, y_size=y_dim, depth=z_dim, viscosity=viscosity, density=density)
    Pool_boundaries.pool_boundary_creator()
    return Pool_boundaries


def grid_setup(Pool_boundaries):
    grid = gcs.grid_creation(Pool_boundaries.x_size, Pool_boundaries.y_size)
    grid.grid_for_shape(Pool_boundaries)
    plot_x = []
    plot_y = []
    for x in range(grid.width):
        for y in range(grid.height):
            if grid.ref_grid[x][y] == True:
                plot_x.append(grid.grid[x][y][0])
                plot_y.append(grid.grid[x][y][1])
    plt.scatter(plot_x, plot_y, s=2)
    plt.show()
    calc = cpu.CPU_Calculations()
    coord_change = np.array(
        [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]), np.array([1, 1]), np.array([-1, 1]),
         np.array([-1, -1]), np.array([1, -1])])
    output = np.zeros(np.shape(grid.grid))
    blockdim = (1000, 1000)
    griddim = (len(grid.grid) // blockdim[0], len(grid.grid[0]) // blockdim[1])
    l0 = 0.5 * (Pool_boundaries.x_size/len(grid.grid[0]))
    velocity = np.zeros(np.shape(grid.grid))
    c = 2
    divisor_w = 1 / grid.width * Pool_boundaries.x_size
    divisor_h = 1 / grid.height * Pool_boundaries.y_size
    divisor = np.array([divisor_w, divisor_h])
    k = 10
    vector_difference = np.array([0.0, 0.0, 0.0])
    modulus = np.array([0.0])
    Force_store = np.array([0.0,0.0,0.0])
    calc.runner(output, grid.grid, k, l0, blockdim, griddim, vector_difference, modulus, Force_store, grid.ref_grid, grid.width, grid.height, coord_change, divisor, velocity, c)



Pool_boundaries = Pool_Simulation_Setup()
grid_setup(Pool_boundaries)
