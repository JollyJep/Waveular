import numpy as np
from numba import njit, prange
import cupy as cp
from cupyx.profiler import benchmark
import time
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


class CUDA_Calculations:

    def __init__(self, pos_grid, velocity, acceleration, k, sigma, l0, c, coord_change, ref_grid, divisor, pool_mass=10, g=np.array([0, 0, -9.81], dtype=np.float64), mega_array=True, timestep=0.1):
        self.pool_mass = pool_mass
        g_arr = np.zeros(np.shape(pos_grid), dtype=np.float64)
        g_arr[:, :] = g
        self.g_gpu = g[2]
        self.pos_grid = pos_grid
        self.pos_grid_gpu = cp.array(pos_grid, dtype=cp.float64)
        self.velocity_gpu = cp.array(velocity, dtype=cp.float64)
        self.acceleration_gpu = cp.array(acceleration, dtype=np.float64)
        self.k_gpu = k
        self.l0_gpu = cp.array(l0, np.float64)
        self.c_gpu = cp.full(cp.shape(self.pos_grid_gpu), c, dtype=np.float64)
        self.weight_arry, self.mass_arry = self.weight(self.pool_mass, len(pos_grid[0]) * len(pos_grid[1]), g_arr)
        self.weight_arry_gpu = cp.array(self.weight_arry, dtype=np.float64)
        self.mass_arry_gpu = cp.full(cp.shape(self.pos_grid_gpu), self.mass_arry, dtype=np.float64)
        if mega_array == True:
            mega_pos_grid = np.zeros((int((750000000/10)//pos_grid.nbytes), len(pos_grid[0]), len(pos_grid[1]), 3), dtype=np.float64)
            self.mega_pos_grid_gpu = cp.array(mega_pos_grid, dtype=np.float64)
        self.mega_arrays = mega_array
        self.resultant_force_gpu = cp.zeros(np.shape(pos_grid), dtype=np.float64)
        self.shift_pos, self.shift_velocity = self.quick_shift(cp.asnumpy(self.pos_grid_gpu), coord_change, ref_grid, divisor,cp.asnumpy(self.velocity_gpu))
        self.shift_pos_gpu = cp.array(self.shift_pos)
        self.shift_velocity_gpu = cp.array(self.shift_velocity)
        self.ref_grid_gpu = cp.array(ref_grid)
        self.ref_grid_gpu = cp.expand_dims(self.ref_grid_gpu, 2)
        self.deltaT = timestep
        self.one_by_deltaT = cp.full(cp.shape(self.pos_grid_gpu), 1/timestep, dtype=np.float64)
        self.sigma = sigma



    def runner(self, coord_change):
        self.kinetics_gpu = cp.zeros((len(self.mega_pos_grid_gpu), len(self.mega_pos_grid_gpu[0]), len(self.mega_pos_grid_gpu[0, 0])))
        self.gpe_gpu = cp.zeros((len(self.mega_pos_grid_gpu), len(self.mega_pos_grid_gpu[0]), len(self.mega_pos_grid_gpu[0, 0])))
        self.epe_gpu = cp.zeros((len(self.mega_pos_grid_gpu), len(self.mega_pos_grid_gpu[0]), len(self.mega_pos_grid_gpu[0, 0])))
        self.frame = 0
        if self.mega_arrays:
            for index, array in enumerate(self.mega_pos_grid_gpu):
                self.verlet(coord_change)
                self.mega_pos_grid_gpu[index] = self.pos_grid_gpu
            self.energy_gpu = cp.array([self.kinetics_gpu, self.gpe_gpu, self.epe_gpu])
            return cp.asnumpy(self.mega_pos_grid_gpu), cp.asnumpy(self.energy_gpu)
        else:
            self.verlet(coord_change)
            return


    def hookes_law(self, coord_change):
        #Define arrays on the gpu
        shift_pos_gpu = self.shift_pos_gpu
        shift_velocity_gpu = self.shift_velocity_gpu
        shift_pos_gpu[1: len(shift_pos_gpu[0]) - 1, 1:len(shift_pos_gpu[1]) - 1] = self.pos_grid_gpu
        shift_velocity_gpu[1: len(shift_velocity_gpu[0]) - 1, 1:len(shift_velocity_gpu[1]) - 1] = self.velocity_gpu
        resultant_force_gpu = cp.zeros(cp.shape(self.pos_grid_gpu))

        #Gpu math
        for repeat in range(8):
            shifted_pos_gpu = cp.roll(shift_pos_gpu, (coord_change[repeat][0], coord_change[repeat][1]), (0, 1))
            shifted_velocity_gpu = cp.roll(shift_velocity_gpu, (coord_change[repeat][0], coord_change[repeat][1]), (0, 1))
            vector_difference_gpu = shifted_pos_gpu[1: len(shift_pos_gpu[0]) - 1, 1: len(shift_pos_gpu[1]) - 1] - self.pos_grid_gpu
            velocity_difference_gpu = shifted_velocity_gpu[1: len(shift_velocity_gpu[0]) - 1, 1: len(shift_velocity_gpu) - 1] - self.velocity_gpu
            modulus_gpu = cp.sqrt(vector_difference_gpu[:, :, 0] ** 2 + vector_difference_gpu[:, :, 1] ** 2 + vector_difference_gpu[:, :, 2] ** 2)
            self.epe_gpu[self.frame] += 1/2 * self.k_gpu * (modulus_gpu - self.l0_gpu[repeat]) ** 2
            modulus_gpu = cp.expand_dims(modulus_gpu, 2)
            resultant_force_gpu += self.k_gpu * (modulus_gpu - self.l0_gpu[repeat]) * 1 / modulus_gpu * vector_difference_gpu + self.c_gpu * velocity_difference_gpu
        self.resultant_force_gpu = (resultant_force_gpu + self.weight_arry_gpu) * self.ref_grid_gpu


    def surface_tension(self):
        grad_x_gpu, grad_y_gpu = cp.gradient(self.pos_grid[:,:,2])
        normal_vec_gpu = cp.zeros_like(self.pos_grid_gpu)
        normal_vec_gpu[:, :, 0] = -grad_x_gpu
        normal_vec_gpu[:, :, 1] = -grad_y_gpu
        normal_vec_gpu[:, :, 2] = 1
        mod_grad_z_gpu = cp.sqrt(grad_x_gpu ** 2 + grad_y_gpu ** 2 + 1)
        div_normal_vec_gpu = cp.gradient(normal_vec_gpu[:, :, 0], axis=0) +cp.gradient(normal_vec_gpu[:, :, 1], axis=1)
        curvature_gpu = div_normal_vec_gpu/ mod_grad_z_gpu
        self.resultant_force_gpu += -self.sigma * curvature_gpu[:, :, cp.newaxis] * normal_vec_gpu


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
        return shift_pos, velocity_pos


    @staticmethod
    @njit()
    def weight(pool_mass, particle_number, g):
        return pool_mass/particle_number * g, pool_mass/particle_number



    def verlet(self, coord_change):
        mid_velocity_gpu = self.velocity_gpu + 0.5 * self.acceleration_gpu * self.deltaT
        self.pos_grid_gpu = self.pos_grid_gpu + mid_velocity_gpu * self.deltaT
        self.hookes_law(coord_change)
        self.surface_tension()
        self.acceleration_gpu = self.resultant_force_gpu/self.mass_arry_gpu
        self.velocity_gpu = mid_velocity_gpu + 0.5 * self.acceleration_gpu * self.deltaT
        self.kinetics_gpu[self.frame] = 0.5 * self.mass_arry * (self.velocity_gpu[:, :, 0] ** 2 + self.velocity_gpu[:, :, 1] ** 2 + self.velocity_gpu[:, :, 2] ** 2)
        self.gpe_gpu[self.frame] = -self.mass_arry * self.g_gpu * self.pos_grid_gpu[:, :, 2]
        self.frame += 1

