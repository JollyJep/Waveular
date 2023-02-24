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

    def grid_for_shape(self, pool):
        shape = pool.boundary
        img = shape.convert("RGBA")
        width, height = img.size
        grid_x = np.array([])
        grid_y = np.array([])
        reflect = np.array([])
        for x in range(width):
            for y in range(height):
                if img.getpixel((x, y))[0] < 5 and img.getpixel((x, y))[3] == 255:
                    grid_x = np.append(grid_x, x)
                    grid_y = np.append(grid_y, y)
                    reflect = np.append(reflect, True)
                elif img.getpixel((x, y))[0] >= 5 and img.getpixel((x, y))[3] == 255:
                    grid_x = np.append(grid_x, x)
                    grid_y = np.append(grid_y, y)
                    reflect = np.append(reflect, False)
        self.grid = np.array([grid_x, grid_y])
        print(self.grid)

