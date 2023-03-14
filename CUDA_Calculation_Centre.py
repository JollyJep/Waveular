import numpy as np
from numba import njit, prange
import cupy as cp
from cupyx.profiler import benchmark


class CUDA_Calculations:

    def __init__(self, pos_grid, velocity, acceleration, k, l0, c, pool_mass=10, g=np.array([0, 0, -9.81]), mega_array=True):
        self.pool_mass = pool_mass
        g_arr = np.zeros(np.shape(pos_grid))
        g_arr[:, :] = g
        self.pos_grid = pos_grid
        self.pos_grid_gpu = cp.array(pos_grid)
        self.velocity_gpu = cp.array(velocity)
        self.acceleration_gpu = cp.array(acceleration)
        self.k_gpu = cp.full(cp.shape(self.pos_grid_gpu), k)
        self.l0_gpu = cp.array(l0)
        self.c_gpu = cp.full(cp.shape(self.pos_grid_gpu), c)
        self.weight_arry, self.mass_arry = self.weight(self.pool_mass, len(pos_grid[0]) * len(pos_grid[1]), g_arr)
        self.weight_arry_gpu = cp.array(self.weight_arry)
        self.mass_arry_gpu = cp.array(self.mass_arry)
        if mega_array == True:
            mega_pos_grid = np.zeros((int((3000000000 * 0.75)//pos_grid.nbytes), len(pos_grid[0]), len(pos_grid[1]), 3), dtype=np.float64)
            self.mega_pos_grid_gpu = cp.array(mega_pos_grid, dtype=np.float64)
        self.mega_arrays = mega_array
        self.resultant_force_gpu = cp.zeros(np.shape(pos_grid))

    def runner(self, k, l0, ref_grid, coord_change, divisor, c, deltaT):
        #print(benchmark((self.hookes_law), (pos_grid, k, l0, ref_grid, coord_change, divisor, velocity, c), n_repeat=50))
        #self.hookes_law(pos_grid, k, l0, ref_grid, coord_change, divisor, velocity, c)
        if self.mega_arrays:
            for index, array in enumerate(self.mega_pos_grid_gpu):
                self.verlet(self.pos_grid, k, l0, ref_grid, coord_change, divisor, c, deltaT)
                self.mega_pos_grid_gpu[index] = self.pos_grid_gpu
            return cp.asnumpy(self.mega_pos_grid_gpu)
        else:
            self.verlet(self.pos_grid, k, l0, ref_grid, coord_change, divisor, c, deltaT)
            return


    def hookes_law(self, k, l0, ref_grid, coord_change, divisor, velocity, c):
        #Define shifting array
        shift_pos, shift_velocity = self.quick_shift(cp.asnumpy(self.pos_grid_gpu), coord_change, ref_grid, divisor, cp.asnumpy(self.velocity_gpu))

        #Define arrays on the gpu
        shift_pos_gpu = cp.array(shift_pos)
        shift_velocity_gpu = cp.array(shift_velocity)
        resultant_force_gpu = cp.zeros(cp.shape(self.pos_grid_gpu))

        #Gpu math
        for repeat in range(8):
            shifted_pos_gpu = cp.roll(shift_pos_gpu, (coord_change[repeat][0], coord_change[repeat][1]), (0, 1))
            shifted_velocity_gpu = cp.roll(shift_velocity_gpu, (coord_change[repeat][0], coord_change[repeat][1]), (0, 1))
            vector_difference_gpu = shifted_pos_gpu[1: len(shift_pos_gpu[0]) - 1, 1: len(shift_pos_gpu[1]) - 1] - self.pos_grid_gpu
            velocity_difference_gpu = shifted_velocity_gpu[1: len(shift_velocity_gpu[0]) - 1, 1: len(shift_velocity_gpu) - 1] - self.velocity_gpu
            modulus_gpu = cp.sqrt(vector_difference_gpu[:, :, 0] ** 2 + vector_difference_gpu[:, :, 1] ** 2 + vector_difference_gpu[:, :, 2] ** 2)
            modulus_gpu = cp.expand_dims(modulus_gpu, 2)
            print(modulus_gpu[500][500])
            resultant_force_gpu += self.k_gpu * (modulus_gpu - self.l0_gpu[repeat]) * 1/modulus_gpu * vector_difference_gpu - self.c_gpu * velocity_difference_gpu
        self.resultant_force_gpu = resultant_force_gpu + self.weight_arry_gpu



    @staticmethod
    @njit(parallel=True)
    def quick_shift(pos_grid, coord_change, ref_grid, divisor, velocity):
        shift_pos = np.zeros((len(pos_grid[0]) + 2, len(pos_grid[1]) + 2, 3))
        velocity_pos = np.zeros((len(pos_grid[0]) + 2, len(pos_grid[1]) + 2, 3))
        for repeat in prange(8):
            for i in range(len(pos_grid[0])):
                for j in range(len(pos_grid[1])):
                    if i + coord_change[repeat][0] < 0 or j + coord_change[repeat][1] < 0 or i + coord_change[repeat][0] > len(pos_grid[0]) - 1 or j + coord_change[repeat][1] > len(pos_grid[1]) - 1 or not ref_grid[i + coord_change[repeat][0]][j + coord_change[repeat][1]]:
                            shift_pos[i + coord_change[repeat][0] + 1][j + coord_change[repeat][1] + 1] = coord_change[repeat] * divisor + pos_grid[i][j]
        shift_pos[1: len(shift_pos[0]) - 1, 1:len(shift_pos[1]) - 1] = pos_grid
        velocity_pos[1: len(velocity_pos[0]) - 1, 1:len(velocity_pos[1]) - 1] = velocity
        return shift_pos, velocity_pos


    @staticmethod
    @njit()
    def weight(pool_mass, particle_number, g):
        return pool_mass/particle_number * g, pool_mass/particle_number



    def verlet(self, pos_grid, k, l0, ref_grid, coord_change, divisor, c, deltaT):
        mid_velocity_gpu = self.velocity_gpu + 0.5 * self.acceleration_gpu * deltaT
        self.pos_grid_gpu = self.pos_grid_gpu + mid_velocity_gpu * deltaT
        self.hookes_law(k, l0, ref_grid, coord_change, divisor, mid_velocity_gpu, c)
        self.acceleration_gpu = self.resultant_force_gpu/self.mass_arry_gpu
        print(self.pos_grid_gpu[500][500])
        self.velocity_gpu = mid_velocity_gpu + 0.5 * self.acceleration_gpu * deltaT