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


    def runner(self, output, pos_grid, k, l0, block, grid, vector_difference, modulus, Force_local, ref_grid, coord_change, divisor, velocity, c):
        self.Hookes_law[block, grid](output, pos_grid, k, l0, vector_difference, modulus, Force_local, ref_grid, coord_change, divisor, velocity, c)
        #self.Hookes_law(output, pos_grid, k, l0, vector_difference, modulus, Force_local, ref_grid, width,
        #                             height, coord_change, divisor)


    @staticmethod
    @cuda.jit()
    def Hookes_law(output, pos_grid, k, l0, vector_difference, modulus, Force_local, ref_grid, coord_change, divisor, velocity, c):
        i, j = cuda.grid(2)
        for repeat in range(8):
            if i > len(pos_grid[0]) or j > len(pos_grid[1]):
                return
            if not ref_grid[i][j]:
                return
            if i+coord_change[repeat][0] < 0 or j + coord_change[repeat][1] < 0 or i+coord_change[repeat][0] > len(pos_grid[0]) - 1 or j + coord_change[repeat][1] > len(pos_grid[1]) - 1 or not ref_grid[i+coord_change[repeat][0]][j+coord_change[repeat][1]]:
                vector_difference[i][j][0] = coord_change[repeat][0] * divisor[0]
                vector_difference[i][j][1] = coord_change[repeat][1] * divisor[1]
                vector_difference[i][j][2] = pos_grid[i][j][2]
            else:
                vector_difference[i][j][0] = pos_grid[i+coord_change[repeat][0]][j + coord_change[repeat][1]][0] - pos_grid[i][j][0] #CUDA is not allowed to create new variables, hence separating vectors to copy x,y,z values instead
                vector_difference[i][j][1] = pos_grid[i+coord_change[repeat][0]][j + coord_change[repeat][1]][1] - pos_grid[i][j][1]
                vector_difference[i][j][2] = pos_grid[i+coord_change[repeat][0]][j + coord_change[repeat][1]][2] - pos_grid[i][j][2]
            modulus[i][j] = (vector_difference[i][j][0] ** 2 + vector_difference[i][j][1] ** 2 + vector_difference[i][j][2] ** 2) ** (1/2)
            Force_local[i][j][0] = k * (modulus[i][j] - l0[repeat]) * 1/modulus[i][j] * vector_difference[i][j][0]
            Force_local[i][j][1] = k * (modulus[i][j] - l0[repeat]) * 1 / modulus[i][j] * vector_difference[i][j][1]
            Force_local[i][j][2] = k * (modulus[i][j] - l0[repeat]) * 1 / modulus[i][j] * vector_difference[i][j][2]
            output[i][j][0] = output[i][j][0] + Force_local[i][j][0]
            output[i][j][1] = output[i][j][1] + Force_local[i][j][1]
            output[i][j][2] = output[i][j][2] + Force_local[i][j][2]

            if i > len(pos_grid[0]) or j > len(pos_grid[1]):
                return
            if not ref_grid[i][j]:
                return
            if i+coord_change[repeat][0] < 0 or j + coord_change[repeat][1] < 0 or i+coord_change[repeat][0] > len(velocity[0]) - 1 or j + coord_change[repeat][1] > len(velocity[1]) - 1 or not ref_grid[i+coord_change[repeat][0]][j+coord_change[repeat][1]]:
                vector_difference[i][j][0] = velocity[i][j][0]
                vector_difference[i][j][1] = velocity[i][j][1]
                vector_difference[i][j][2] = velocity[i][j][2]
            else:
                vector_difference[i][j][0] = velocity[i][j][0] - velocity[i+coord_change[repeat][0]][j + coord_change[repeat][1]][0] #CUDA is not allowed to create new variables, hence separating vectors to copy x,y,z values instead
                vector_difference[i][j][1] = velocity[i][j][1] - velocity[i+coord_change[repeat][0]][j + coord_change[repeat][1]][1]
                vector_difference[i][j][2] = velocity[i][j][2] - velocity[i+coord_change[repeat][0]][j + coord_change[repeat][1]][2]
            Force_local[i][j][0] = c * vector_difference[i][j][0]
            Force_local[i][j][1] = c * vector_difference[i][j][0]
            Force_local[i][j][2] = c * vector_difference[i][j][0]
            output[i][j][0] = output[i][j][0] + Force_local[i][j][0]
            output[i][j][1] = output[i][j][1] + Force_local[i][j][1]
            output[i][j][2] = output[i][j][2] + Force_local[i][j][2]

