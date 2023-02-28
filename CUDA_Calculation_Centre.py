import numpy as np
import numba
from numba import jit
from numba import cuda
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import PIL
import math



class CUDA_Calculations:

    def __init__(self):
        return


    def runner(self, output, pos_grid, shifted_pos, k, l0, block, grid, vector_difference, modulus, Force_local):
        self.Hookes_law[2,1](output, pos_grid, k, l0, vector_difference, modulus, Force_local)
        print(output)

    @staticmethod
    @cuda.jit(debug=True)
    def Hookes_law(output, pos_grid, k, l0, vector_difference, modulus, Force_local, x, y):
        i = cuda.grid(1)
        for j in range(8):
            if i > len(pos_grid[0]) :
                return
            vector_difference[0] = pos_grid[i][0] - pos_grid[i][0]   #CUDA is not allowed to create new variables, hence separating vectors to copy x,y,z values instead
            vector_difference[1] = pos_grid[i][1] - pos_grid[i][1]
            vector_difference[2] = pos_grid[i][2] - pos_grid[i][2]
            modulus[0] = (vector_difference[0] **2 + vector_difference[1] ** 2)**(1/2)
            Force_local[0] = -k * (modulus[0] - l0) * 1/modulus[0] * vector_difference[0]
            Force_local[1] = -k * (modulus[0] - l0) * 1 / modulus[0] * vector_difference[1]
            Force_local[2] = -k * (modulus[0] - l0) * 1 / modulus[0] * vector_difference[2]
            output[i][0] = output[i][0] + Force_local[0]
            output[i][1] = output[i][1] + Force_local[1]
            output[i][2] = output[i][2] + Force_local[2]
