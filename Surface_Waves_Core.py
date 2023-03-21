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


def Pool_Simulation_Setup(shape="circular", x_dim=100, y_dim=100, z_dim=1, viscosity=1, density=1):
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
    calculation_system(grid, Pool_boundaries, True, 10, 2, True)



def calculation_system(grid, pool, run_cuda, k, c, mega_arrays):
    coord_change = np.array(
        [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]), np.array([1, 1, 0]),
         np.array([-1, 1, 0]), np.array([-1, -1, 0]), np.array([1, -1, 0])])
    c = 0.005
    divisor_w = 1 / grid.width * Pool_boundaries.x_size
    divisor_h = 1 / grid.height * Pool_boundaries.y_size
    divisor = np.array([divisor_w, divisor_h, 0], dtype=np.float64)
    k = 5000
    sigma = 10
    l0 = np.array([(pool.x_size / len(grid.grid)), (pool.x_size / len(grid.grid)), (pool.y_size / len(grid.grid[0])), (pool.y_size / len(grid.grid[0])), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2), np.sqrt(((pool.x_size / len(grid.grid)))**2 + ((pool.y_size / len(grid.grid[0])))**2)], dtype=np.float64)
    velocity = np.zeros(np.shape(grid.grid), dtype=np.float64)
    acceleration = np.zeros(np.shape(grid.grid), dtype=np.float64)
    ref_grid = grid.ref_grid
    deltaT=0.0001
    if run_cuda:
        calc = ccc.CUDA_Calculations(grid.grid, velocity, acceleration, k, sigma, l0, c, coord_change, ref_grid, divisor, 2.5*2.5, np.array([0, 0, -9.81], dtype=np.float64), mega_arrays, deltaT)
    else:
        calc = cpu.CPU_Calculations()
    start = time.time()
    index_offset = 0
    repeats = 30
    for x in range(repeats):
        position, energies = calc.runner(coord_change)
        pool_attributes = np.array([deltaT, ref_grid])
        number = ""
        while len(number) + len(str(x)) < len(str(repeats)):
            number += "0"
        number += str(x)
        if mega_arrays:
            np.savez_compressed("./Output/mega_array_pos_" + number, position)
            np.savez_compressed("./Output/mega_array_eng_" + number, energies)
            np.savez_compressed("./Output/mega_array_atr_" + number, pool_attributes)
        else:
            if x == 0:
                data_output = np.full((25, len(grid.grid[0]), len(grid.grid[0][0]), 3), np.nan)
                data_output[x] = position
            elif x + 1 % 25 == 0:
                data_output[x - index_offset] = position
                np.savez_compressed("./Output/mini_dat_" + str(index_offset/25), position)
                data_output = np.full((25, len(grid.grid[0]), len(grid.grid[0][0]), 3), np.nan)
                index_offset += 25
            else:
                data_output[x - index_offset] = position
    if not mega_arrays:
        if data_output[0][0][0] != np.nan:
            np.savez_compressed("./Output/mini_dat_" + str(index_offset/25), position)
    print(time.time() - start)


if __name__ == "__main__":
    Pool_boundaries = Pool_Simulation_Setup()
    grid_setup(Pool_boundaries)