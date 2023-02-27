import numpy as np
import scipy
import numba
from numba import njit
from numba import cuda
import pandas as pd
from matplotlib import pyplot as plt
import Pool_Simulation as Pool_sim
import Grid_Creation_System as gcs


def Pool_Simulation_Setup(shape="circular", x_dim=1, y_dim=1, z_dim=1, viscosity=1, density=1):
    Pool_boundaries = Pool_sim.Pool_Simulation(shape=shape, x_size=x_dim, y_size=y_dim, depth=z_dim, viscosity=viscosity, density=density)
    Pool_boundaries.pool_boundary_creator()
    return Pool_boundaries


def grid_setup(Pool_boundaries):
    grid = gcs.grid_creation(Pool_boundaries.x_size, Pool_boundaries.y_size)
    grid.grid_for_shape(Pool_boundaries)
    print(np.shape(grid.grid))
    plt.scatter(grid.grid[0], grid.grid[1], s=2)
    plt.show()
    data_matrix_creator(grid)


def data_matrix_creator(grid):
    data_matrix = np.array([np.array([grid.grid[0], grid.grid[1], np.zeros(len(grid.grid[0]))]), np.array([np.zeros(len(grid.grid[0])), np.zeros(len(grid.grid[0])), np.zeros(len(grid.grid[0]))]), np.array([np.zeros(len(grid.grid[0])), np.zeros(len(grid.grid[0])), np.zeros(len(grid.grid[0]))]), np.array([np.zeros(len(grid.grid[0])), np.zeros(len(grid.grid[0])), np.zeros(len(grid.grid[0]))])], dtype=np.float64)
    print(np.shape(data_matrix))
    print(data_matrix[0].nbytes)

Pool_boundaries = Pool_Simulation_Setup()
grid_setup(Pool_boundaries)
