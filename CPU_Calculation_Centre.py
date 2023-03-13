import numpy as np
from numba import njit, jit
from numba import prange
import concurrent.futures
from multiprocessing import Lock


class CPU_Calculations:

    def __init__(self):
        return

    def runner(self, pos_grid, k, l0, ref_grid, coord_change, divisor, velocity, c):
        self.hookes_law(pos_grid, k, l0, ref_grid, coord_change, divisor, velocity, c)

    def hookes_law(self, pos_grid, k, l0, ref_grid, coord_change, divisor, velocity, c):
        # Define shifting array
        shift_pos = self.quick_shift(pos_grid, coord_change, ref_grid, divisor)
        # Define arrays
        k = np.full(np.shape(pos_grid), k)
        l0 = np.array(l0)
        resultant_force = np.zeros(np.shape(pos_grid), dtype=np.float64)

        shifted_pos = np.zeros((8, len(shift_pos[0]), len(shift_pos[1]), 3))
        for repeat in range(8):
            shifted_pos[repeat] = np.roll(shift_pos, (coord_change[repeat][0], coord_change[repeat][1]), (0, 1))
        resultant_force = self.cpu_math(shifted_pos, pos_grid, k, l0, resultant_force)





    @staticmethod
    @jit()
    def cpu_math(shifted_pos, pos_grid, k, l0, resultant_force):
        for repeat in prange(8):
            vector_difference = shifted_pos[repeat][1: len(shifted_pos[0][0]) - 1, 1:len(shifted_pos[0][1]) - 1] - pos_grid
            modulus = np.sqrt(vector_difference[:, :, 0] ** 2 + vector_difference[:, :, 1] ** 2 + vector_difference[:, :, 2] ** 2)
            modulus = np.expand_dims(modulus, 2)
            force = k * (modulus - l0[repeat]) * 1 / modulus * vector_difference
            resultant_force +=force
        return resultant_force


    @staticmethod
    @njit(parallel=True)
    def quick_shift(pos_grid, coord_change, ref_grid, divisor):
        shift_pos = np.zeros((len(pos_grid[0]) + 2, len(pos_grid[1]) + 2, 3))
        for repeat in prange(8):
            for i in range(len(pos_grid[0])):
                for j in range(len(pos_grid[1])):
                    if i + coord_change[repeat][0] < 0 or j + coord_change[repeat][1] < 0 or i + coord_change[repeat][0] > len(pos_grid[0]) - 1 or j + coord_change[repeat][1] > len(pos_grid[1]) - 1 or not ref_grid[i + coord_change[repeat][0]][j + coord_change[repeat][1]]:
                        shift_pos[i + coord_change[repeat][0] + 1][j + coord_change[repeat][1] + 1] = coord_change[repeat] * divisor + pos_grid[i][j]

        shift_pos[1: len(shift_pos[0]) - 1, 1:len(shift_pos[1]) - 1] = pos_grid
        return shift_pos
