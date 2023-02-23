import numpy as np
import scipy
import numba
from numba import njit
from numba import cuda

import pandas as pd
from matplotlib import pyplot as plt
import Pool_Simulation as Pool_sim

def Pool_Simulation_Setup(shape="rectangle", x_dim=1, y_dim=1, z_dim=1, viscosity=1, density=1):
    Pool_boundaries = Pool_sim.Pool_Simulation(shape=shape, x_size=x_dim, y_size=y_dim, depth=z_dim, viscosity=viscosity, density=density)
    Pool_boundaries.pool_boundary_creator()
    Pool_boundaries.boundary

Pool_Simulation_Setup()
