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
import time

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
    calculation_system(grid, Pool_boundaries, True, 10, 2)



def calculation_system(grid, pool, run_cuda, k, c):
    coord_change = np.array(
        [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]), np.array([1, 1, 0]),
         np.array([-1, 1, 0]), np.array([-1, -1, 0]), np.array([1, -1, 0])])
    c = 2
    divisor_w = 1 / grid.width * Pool_boundaries.x_size
    divisor_h = 1 / grid.height * Pool_boundaries.y_size
    divisor = np.array([divisor_w, divisor_h, 0])
    k = 10
    l0 = np.array([(pool.x_size / len(grid.grid)), (pool.x_size / len(grid.grid)), (pool.y_size / len(grid.grid[0])), (pool.y_size / len(grid.grid[0])), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2)])
    velocity = np.zeros(np.shape(grid.grid), dtype=np.float64)
    ref_grid = grid.ref_grid
    if run_cuda:
        calc = ccc.CUDA_Calculations()
    else:
        calc = cpu.CPU_Calculations()
    start = time.time()
    for x in range(50):
        calc.runner(grid.grid, k, l0, ref_grid, coord_change, divisor, velocity, c)
    print(time.time() - start)

if __name__ == "__main__":
    Pool_boundaries = Pool_Simulation_Setup()
    grid_setup(Pool_boundaries)