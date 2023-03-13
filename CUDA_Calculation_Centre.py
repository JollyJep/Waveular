import numpy as np
from numba import njit, prange
import cupy as cp
from cupyx.profiler import benchmark


class CUDA_Calculations:

    def __init__(self):
        return


    def runner(self , pos_grid, k, l0, ref_grid, coord_change, divisor, velocity, c):
        print(benchmark((self.hookes_law), (pos_grid, k, l0, ref_grid, coord_change, divisor, velocity, c), n_repeat=2))
        #self.hookes_law(pos_grid, k, l0, ref_grid, coord_change, divisor, velocity, c)


    def hookes_law(self, pos_grid, k, l0, ref_grid, coord_change, divisor, velocity, c):
        #Define shifting array
        shift_pos = self.quick_shift(pos_grid, coord_change, ref_grid, divisor)



        #Define arrays on the gpu
        pos_grid_gpu = cp.array(pos_grid)
        shift_pos_gpu = cp.array(shift_pos)
        k_gpu = cp.full(cp.shape(pos_grid_gpu), k)
        l0_gpu = cp.array(l0)
        resultant_force_gpu = cp.zeros(cp.shape(pos_grid_gpu), dtype=np.float64)


        #Gpu math
        for repeat in range(8):
            shifted_pos_gpu = cp.roll(shift_pos_gpu, (coord_change[repeat][0], coord_change[repeat][1]), (0, 1))
            vector_difference_gpu = shifted_pos_gpu[1: len(shift_pos_gpu[0]) - 1, 1:len(shift_pos_gpu[1]) - 1] - pos_grid_gpu
            modulus_gpu = cp.sqrt(vector_difference_gpu[:, :, 0] ** 2 + vector_difference_gpu[:, :, 1] ** 2 + vector_difference_gpu[:, :, 2] ** 2)
            modulus_gpu = cp.expand_dims(modulus_gpu, 2)
            force_gpu = k_gpu * (modulus_gpu - l0_gpu[repeat]) * 1/modulus_gpu * vector_difference_gpu
            resultant_force_gpu += force_gpu



    @staticmethod
    @njit(parallel=True)
    def quick_shift(pos_grid, coord_change, ref_grid, divisor):
        shift_pos = np.zeros((len(pos_grid[0]) + 2, len(pos_grid[1]) + 2, 3))
        for repeat in prange(8):
            for i in range(len(pos_grid[0])):
                for j in range(len(pos_grid[1])):
                    if i + coord_change[repeat][0] < 0 or j + coord_change[repeat][1] < 0 or i + \
                            coord_change[repeat][0] > len(pos_grid[0]) - 1 or j + coord_change[repeat][1] > len(pos_grid[1]) - 1 or not ref_grid[i + coord_change[repeat][0]][j + coord_change[repeat][1]]:
                           shift_pos[i + coord_change[repeat][0] + 1][j + coord_change[repeat][1] + 1] = coord_change[repeat] * divisor + pos_grid[i][j]

        shift_pos[1: len(shift_pos[0]) - 1, 1:len(shift_pos[1]) - 1] = pos_grid
        return shift_pos