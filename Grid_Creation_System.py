import numpy as np
import numba
from numba import jit
from numba import njit
from numba import cuda
from numba.typed import List
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import PIL



class grid_creation:

    def __init__(self, x_size, y_size):
        self.x_scale = x_size
        self.y_scale = y_size

    def grid_for_shape(self, pool):
        shape = pool.boundary
        img = shape.convert("RGBA")
        width, height = img.size
        pixels = np.asarray(img)
        grid_x = List()
        grid_x.append(1)
        grid_y = List()
        grid_y.append(1)
        reflect = List()
        reflect.append(1)
        grid_x, grid_y, reflect = self.quick_pixel(width, height, pixels, grid_x, grid_y, reflect)
        grid_x = np.array(grid_x)
        grid_y = np.array(grid_y)
        grid = np.array([grid_x,grid_y])
        divisor = np.array([np.full(np.shape(grid)[1], 1 / width), np.full(np.shape(grid)[1], 1 / height)])
        self.grid = grid * divisor
        self.grid_base = grid
        self.reflect = np.array(reflect)


    @staticmethod
    @jit()
    def quick_pixel(width, height, pixels, grid_x, grid_y, reflect):
        for x in range(width):
            for y in range(height):
                if pixels[x, y][0] < 5 and pixels[x, y][3] == 255:
                    grid_x.append(x)
                    grid_y.append(y)
                    reflect.append(1)
                elif pixels[x, y][0] >= 5 and pixels[x, y][3] == 255 and pixels[x, y][1] == 0:
                    grid_x.append(x)
                    grid_y.append(y)
                    reflect.append(0)
        return grid_x[1:], grid_y[1:], reflect[1:]