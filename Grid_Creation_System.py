import numpy as np
import numba
from numba import njit
from numba import cuda
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import PIL



class grid_creation:

    def __init__(self, max_grid_size_x=1000, max_grid_size_y=1000):
        self.grid_size_x = max_grid_size_x
        self.grid_size_y = max_grid_size_y

    def grid_for_shape(self, pool):
        shape = pool.boundary
        img = shape.convert("RGBA")
        width, height = img.size
        pixels = np.asarray(img)
        grid_x = [1]
        grid_y = [1]
        reflect = [True]
        self.grid, self.reflect = self.quick_pixel(width, height, pixels, grid_x, grid_y, reflect)
        print(self.grid)


    @staticmethod
    @njit(parallel=True)
    def quick_pixel(width, height, pixels, grid_x, grid_y, reflect):
        for x in range(width):
            for y in range(height):
                if pixels[x, y][0] < 5 and pixels[x, y][3] == 255:
                    grid_x.append(x)
                    grid_y.append(y)
                    reflect.append(True)
                elif pixels[x, y][0] >= 5 and pixels[x, y][3] == 255:
                    grid_x.append(x)
                    grid_y.append(y)
                    reflect.append(False)
        return np.array([grid_x[1:], grid_y[1:]]), np.array(reflect)