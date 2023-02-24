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
    #plt.plot(Pool_boundaries.boundary[0], Pool_boundaries.boundary[1])
    return Pool_boundaries


def grid_setup(Pool_boundaries, x_grid=10, y_grid=10):
    grid = gcs.grid_creation(x_grid, y_grid)
    grid.grid_for_shape(Pool_boundaries)


Pool_boundaries = Pool_Simulation_Setup()
grid_setup(Pool_boundaries)
