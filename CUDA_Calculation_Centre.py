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


    def runner(self, output, pos_grid, shifted_pos, k, l0, block, grid, vector_difference, modulus):
        self.Hookes_law[2,1](output, pos_grid, shifted_pos, k, l0, vector_difference, modulus, Force_local)
        print(output)

    @staticmethod
    @cuda.jit(debug=True)
    def Hookes_law(output, pos_grid, shifted_pos, k, l0, vector_difference, modulus, Force_local):
        i = cuda.grid(1)
        if i > len(pos_grid[0]) :
            return
        vector_difference[0] = pos_grid[i][0] - shifted_pos[i][0]   #CUDA is not allowed to create new variables, hence separating vectors to copy x,y,z values instead
        vector_difference[1] = pos_grid[i][1] - shifted_pos[i][1]
        modulus[0] = (vector_difference[0] **2 + vector_difference[1] ** 2)**(1/2)
        #print(vector_difference)
        output[i][0] = -k * (modulus[0] - l0) * 1/modulus[0] * vector_difference[0]
        output[i][1] = -k * (modulus[0] - l0) * 1 / modulus[0] * vector_difference[1]
