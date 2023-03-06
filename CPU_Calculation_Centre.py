import time
import timeit
import numpy as np
import numba
from numba import njit
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import PIL
import math
from numba import prange


class CPU_Calculations:

    def __init__(self):
        return


    def runner(self, output, pos_grid, k, l0, block, grid, vector_difference,  Force_local, ref_grid, coord_change, divisor, velocity, c):
        self.Hookes_law(output, pos_grid, k, l0, vector_difference,  Force_local, ref_grid, coord_change, divisor, velocity, c)
        print(output)

    @staticmethod
    #@njit()
    def Hookes_law(output, pos_grid, k, l0, vector_difference,  Force_local, ref_grid, coord_change, divisor, velocity, c):
        for i in prange(3):
            for j in range(3):
                for repeat in range(8):
                    if i > len(pos_grid[0]) or j > len(pos_grid[1]):
                        return
                    if not ref_grid[i][j]:
                        return
                    if i+coord_change[repeat][0] < 0 or j + coord_change[repeat][1] < 0 or i+coord_change[repeat][0] > len(pos_grid[0]) - 1 or j + coord_change[repeat][1] > len(pos_grid[1]) - 1 or not ref_grid[i+coord_change[repeat][0]][j+coord_change[repeat][1]]:
                        vector_difference[0] = coord_change[repeat][0] * divisor[0]
                        vector_difference[1] = coord_change[repeat][1] * divisor[1]
                        vector_difference[2] = pos_grid[i][j][2]
                    else:
                        vector_difference[0] = pos_grid[i+coord_change[repeat][0]][j + coord_change[repeat][1]][0] - pos_grid[i][j][0] #CUDA is not allowed to create new variables, hence separating vectors to copy x,y,z values instead
                        vector_difference[1] = pos_grid[i+coord_change[repeat][0]][j + coord_change[repeat][1]][1] - pos_grid[i][j][1]
                        vector_difference[2] = pos_grid[i+coord_change[repeat][0]][j + coord_change[repeat][1]][2] - pos_grid[i][j][2]
                    vector_difference[3] = (vector_difference[0] ** 2 + vector_difference[1] ** 2 + vector_difference[2] ** 2) ** (1/2)
                    Force_local[0] = k * (vector_difference[3] - l0[repeat]) * 1/vector_difference[3] * vector_difference[0]
                    Force_local[1] = k * (vector_difference[3] - l0[repeat]) * 1 / vector_difference[3] * vector_difference[1]
                    Force_local[2] = k * (vector_difference[3] - l0[repeat]) * 1 / vector_difference[3] * vector_difference[2]
                    output[i][j][0] = output[i][j][0] + Force_local[0]
                    output[i][j][1] = output[i][j][1] + Force_local[1]
                    output[i][j][2] = output[i][j][2] + Force_local[2]
                    if i > len(pos_grid[0]) or j > len(pos_grid[1]):
                        return
                    if not ref_grid[i][j]:
                        return
                    if i + coord_change[repeat][0] < 0 or j + coord_change[repeat][1] < 0 or i + coord_change[repeat][
                        0] > len(velocity[0]) - 1 or j + coord_change[repeat][1] > len(velocity[1]) - 1 or not \
                    ref_grid[i + coord_change[repeat][0]][j + coord_change[repeat][1]]:
                        vector_difference[0] = velocity[i][j][0]
                        vector_difference[1] = velocity[i][j][1]
                        vector_difference[2] = velocity[i][j][2]
                    else:
                        vector_difference[0] = velocity[i][j][0] - velocity[i + coord_change[repeat][0]][j + coord_change[repeat][1]][0]  # CUDA is not allowed to create new variables, hence separating vectors to copy x,y,z values instead
                        vector_difference[1] = velocity[i][j][1] - velocity[i + coord_change[repeat][0]][j + coord_change[repeat][1]][1]
                        vector_difference[2] = velocity[i][j][2] - velocity[i + coord_change[repeat][0]][j + coord_change[repeat][1]][2]
                    Force_local[0] = c * vector_difference[0]
                    Force_local[1] = c * vector_difference[1]
                    Force_local[2] = c * vector_difference[2]
                    output[i][j][0] = output[i][j][0] + Force_local[0]
                    output[i][j][1] = output[i][j][1] + Force_local[1]
                    output[i][j][2] = output[i][j][2] + Force_local[2]


        def weight_calc(mass, g):
            return mass * g