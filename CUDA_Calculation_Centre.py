import numpy as np
import numba
from numba import njit
from numba import cuda
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import PIL



class CUDA_Calculations:

    def __init__(self, max_grid_size_x=1000, max_grid_size_y=1000):
        self.grid_size_x = max_grid_size_x
        self.grid_size_y = max_grid_size_y

