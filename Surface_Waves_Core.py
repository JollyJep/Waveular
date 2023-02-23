import numpy as np
import numba
from numba import njit
from numba import cuda
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import Pool_Simulation as Pool_sim

def Pool_Simulation_Setup(shape="circular", x_dim=1, y_dim=1, z_dim=1, viscosity=1, density=1):
    Pool_sim.Pool_Simulation(shape=shape, x_size=x_dim, y_size=y_dim, depth=z_dim, viscosity=viscosity, density=density)


