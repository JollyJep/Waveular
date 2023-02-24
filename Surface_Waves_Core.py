import numpy as np
import scipy
import numba
from numba import njit
from numba import cuda

import pandas as pd
from matplotlib import pyplot as plt
import Pool_Simulation as Pool_sim
import CUDA_Calculation_Centre as ccc

def Pool_Simulation_Setup(shape="circular", x_dim=1, y_dim=1, z_dim=1, viscosity=1, density=1):
    Pool_boundaries = Pool_sim.Pool_Simulation(shape=shape, x_size=x_dim, y_size=y_dim, depth=z_dim, viscosity=viscosity, density=density)
    Pool_boundaries.pool_boundary_creator()
    #plt.plot(Pool_boundaries.boundary[0], Pool_boundaries.boundary[1])
    return Pool_boundaries


def CUDA_setup(Pool_boundaries, x_grid=10,y_grid=10, ):
    cuda_grid = ccc.CUDA_Calculations(x_grid, y_grid)
    cuda_grid.grid_for_shape(Pool_boundaries)
    plt.scatter(cuda_grid.grid[0], cuda_grid.grid[1])
    plt.show()


Pool_boundaries = Pool_Simulation_Setup()
CUDA_setup(Pool_boundaries)
