import numpy as np
import scipy
import numba
from numba import njit
from numba import cuda
import pandas as pd
from matplotlib import pyplot as plt
import Pool_Simulation as Pool_sim
import Grid_Creation_System as gcs


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


Pool_boundaries = Pool_Simulation_Setup()
grid_setup(Pool_boundaries)
