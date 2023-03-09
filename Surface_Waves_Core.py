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
    print("here")
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
        [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]), np.array([1, 1]), np.array([-1, 1]),
         np.array([-1, -1]), np.array([1, -1])])
    output = np.zeros(np.shape(grid.grid))
    blockdim = (224, 224)
    griddim = (len(grid.grid) // blockdim[0], len(grid.grid[0]) // blockdim[1])
    c = 2
    divisor_w = 1 / grid.width * Pool_boundaries.x_size
    divisor_h = 1 / grid.height * Pool_boundaries.y_size
    divisor = np.array([divisor_w, divisor_h])
    k = 10
    vector_difference = np.zeros(np.shape(output))
    modulus = np.zeros((len(output), len(output[0])))
    Force_store = np.zeros(np.shape(output))
    l0 = np.array([(pool.x_size / len(grid.grid)), (pool.x_size / len(grid.grid)), (pool.y_size / len(grid.grid[0])), (pool.y_size / len(grid.grid[0])), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2)])
    velocity = np.zeros(np.shape(grid.grid), dtype=np.float32)
    ref_grid = grid.ref_grid
    if run_cuda:
        calc = ccc.CUDA_Calculations()
        vector_difference = cuda.to_device(vector_difference)
        Force_store = cuda.to_device(Force_store)
        output = cuda.to_device(output)
        divisor = cuda.to_device(divisor)
        ref_grid = cuda.to_device(grid.ref_grid)
        grid_gpu = cuda.to_device(grid.grid)
        velocity = cuda.to_device(velocity)
        coord_change = cuda.to_device(coord_change)


    else:
        calc = cpu.CPU_Calculations()
    start = time.time()
    for x in range(500):
        calc.runner(output, grid_gpu, k, l0, blockdim, griddim, vector_difference, modulus, Force_store, ref_grid, coord_change, divisor, velocity, c)
    print(time.time() - start)


Pool_boundaries = Pool_Simulation_Setup()
grid_setup(Pool_boundaries)