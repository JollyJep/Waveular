import numpy as np
from numba import njit, prange, jit
import cupy as cp
from cupyx.profiler import benchmark
import time
import sys


class CUDA_Calculations:
    """
    ----------------------------------------------------------------
    Description:
    CUDA_Calculations is the class used to calculate all attributes of all particles in the simulation, across time.
    This is the class where the vast majority of calculations are handled, be that forces or Verlet integration.
    All subordinate calculation functions are handled with self.runner.
    ----------------------------------------------------------------
    Inputs: pos_grid/self.pos_grid/self.pos_grid_gpu (Numpy (input)/Numpy/Cupy float64 array) - Maintains current positions in cartesian SI coordinates of all particles within the simulation with a 2d array of 3 vectors.
            velocity/self.velocity_gpu (Numpy (input)/Cupy float64 array) - Maintains current velocities in cartesian SI coordinates of all particles within the simulation with a 2d array of 3 vectors.
            acceleration/self.acceleration_gpu (Numpy (input)/Cupy float64 array) - Maintains current accelerations in cartesian SI coordinates of all particles within the simulation with a 2d array of 3 vectors.
            k/self.k_gpu (float/float) - Spring constant between all particle neighbours.
            sigma/self.sigma (float/float) - Surface tension constant.
            l0/self.l0_gpu (Numpy (input)/Cupy float64 array) - Equilibrium lengths of springs in all 8 directions from any particle.
            c/self.c_gpu (float (input)/Cupy float64 array) - Damping coefficient of all springs.
            coord_change (Numpy int array) - Defines grid coordinate transformations to all 8 nearest neighbours from any particle.
            ref_grid/self.ref_grid_gpu (Numpy (input)/Cupy bool array) - Defines where the simulation should occur on the grid. Used as Numpy arrays prefer to be rectangular than oddly shaped.
            divisor (Numpy float64 array) - Array used to convert from grid coordinates to SI simulation coordinates.
            pool_mass (float) - Mass of all particles in the pool.
            g/g_arr/self.g_gpu (Numpy (input)/Numpy float64 array/float) - Gravitational acceleration.
            mega_array (bool) - Used to decide if multi-timestep output arrays should be used. Theoretically reduces performance loss due to cpu-gpu copying overhead.
            self.mega_pos_grid_gpu (Cupy float64 array) - Multi-timestep array to reduce copy overhead.
            timestep/self.deltaT (float, float) - Timestep of simulation.
            debug (bool) - For unit testing.
            integrator (string) - Choose numerical method
    ----------------------------------------------------------------
    Outputs: self.mega_pos_grid_gpu (Cupy float64 array outputted as Numpy float64) - Contains all the positions of all particles in SI cartesian coordinates for the number of timesteps that fit into the allocated gpu memory.
             self.energy_gpu (Cupy float64 array outputted as Numpy float64) - Contains all the different energies of all particles in SI units for the number of timesteps that fit as done in mega_pos_grid.
    ----------------------------------------------------------------
    """
    def __init__(self, pos_grid, velocity, acceleration, k, sigma, l0, c, coord_change, ref_grid, divisor, pool_mass=10, g=np.array([0, 0, -9.81], dtype=np.float64), mega_array=True, timestep=0.1, debug=False, VRAM=100000, integrator="ER"):
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
        self.weight_arry, self.mass_arry = self.weight(self.pool_mass, len(pos_grid[0]) * len(pos_grid[1]), g_arr)  #Weight is a constant value and hence can be worked out before time
        self.weight_arry_gpu = cp.array(self.weight_arry, dtype=np.float64)
        self.mass_arry_gpu = cp.full(cp.shape(self.pos_grid_gpu), self.mass_arry, dtype=np.float64)
        if not debug:
            mega_pos_grid = np.zeros((int((VRAM)//pos_grid.nbytes), len(pos_grid[0]), len(pos_grid[1]), 3), dtype=np.float64)    #Defines size of mega_pos_grid, based on memory allocation
            self.mega_pos_grid_gpu = cp.array(mega_pos_grid, dtype=np.float64)
        elif mega_array and debug:
            mega_pos_grid = np.zeros((1, len(pos_grid[0]), len(pos_grid[1]), 3),dtype=np.float64)
            self.mega_pos_grid_gpu = cp.array(mega_pos_grid, dtype=np.float64)
        self.mega_arrays = mega_array
        self.resultant_force_gpu = cp.zeros(np.shape(pos_grid), dtype=np.float64)   # Defines empty resultant force array
        self.shift_pos = self.quick_shift(cp.asnumpy(self.pos_grid_gpu), coord_change, ref_grid, divisor, cp.asnumpy(self.velocity_gpu))    # Creates shiftable array using Numba accelerated algorithm
        self.shift_velocity = np.zeros((len(pos_grid[0]) + 2, len(pos_grid[1]) + 2, 3))  # Creates oversized velocity array, to allow for shifting for particles on the edge
        # Copy arrays to gpu (any cp.array is stored in gpu memory)
        self.shift_pos_gpu = cp.array(self.shift_pos)
        self.shift_velocity_gpu = cp.array(self.shift_velocity)
        self.ref_grid_gpu = cp.array(ref_grid)
        self.ref_grid_gpu = cp.expand_dims(self.ref_grid_gpu, 2)
        self.deltaT = timestep
        self.one_by_deltaT = cp.full(cp.shape(self.pos_grid_gpu), 1/timestep, dtype=np.float64)
        self.sigma = sigma
        self.integrator = integrator



    def runner(self, coord_change): # Host function, manages filling of the mega arrays
        self.kinetics_gpu = cp.zeros((len(self.mega_pos_grid_gpu), len(self.mega_pos_grid_gpu[0]), len(self.mega_pos_grid_gpu[0, 0])))
        self.gpe_gpu = cp.zeros((len(self.mega_pos_grid_gpu), len(self.mega_pos_grid_gpu[0]), len(self.mega_pos_grid_gpu[0, 0])))
        self.epe_gpu = cp.zeros((len(self.mega_pos_grid_gpu), len(self.mega_pos_grid_gpu[0]), len(self.mega_pos_grid_gpu[0, 0])))
        self.frame = 0
        for index, array in enumerate(self.mega_pos_grid_gpu):  # Minor arrays now work
            if self.integrator == "V":
                self.verlet(coord_change)
            elif self.integrator == "ER":
                self.euler_richardson(coord_change)
            self.mega_pos_grid_gpu[index] = self.pos_grid_gpu
        self.energy_gpu = cp.array([self.kinetics_gpu, self.gpe_gpu, self.epe_gpu])
        return cp.asnumpy(self.mega_pos_grid_gpu), cp.asnumpy(self.energy_gpu)



    def hookes_law(self, coord_change):
        #Define arrays on the gpu
        shift_pos_gpu = self.shift_pos_gpu
        shift_velocity_gpu = self.shift_velocity_gpu
        shift_pos_gpu[1: len(shift_pos_gpu[0]) - 1, 1:len(shift_pos_gpu[1]) - 1] = self.pos_grid_gpu
        shift_velocity_gpu[1: len(shift_velocity_gpu[0]) - 1, 1:len(shift_velocity_gpu[1]) - 1] = self.mid_velocity_gpu
        resultant_force_gpu = cp.zeros(cp.shape(self.pos_grid_gpu))

        #Gpu math
        for repeat in range(8): #Repeat for every neighbour direction. With a bit of work could be cut in half by mirroring forces, however this will increase VRAM requirements by around 10-20%.
            shifted_pos_gpu = cp.roll(shift_pos_gpu, (coord_change[repeat][0], coord_change[repeat][1]), (0, 1))    # Allows particles to psuedo-interact with neighbours, by simply opening neighbouring positional data to particle for Hooke's Law.
            shifted_velocity_gpu = cp.roll(shift_velocity_gpu, (coord_change[repeat][0], coord_change[repeat][1]), (0, 1))  # Allows particles to psuedo-interact with neighbours, by simply opening neighbouring velocity data to particle damping.
            vector_difference_gpu = shifted_pos_gpu[1: len(shift_pos_gpu[0]) - 1, 1: len(shift_pos_gpu[1]) - 1] - self.pos_grid_gpu # Position vector from particle to neighbour
            velocity_difference_gpu = shifted_velocity_gpu[1: len(shift_velocity_gpu[0]) - 1, 1: len(shift_velocity_gpu) - 1] - self.mid_velocity_gpu   # Velocity vector from particle to neighbour
            modulus_gpu = cp.sqrt(vector_difference_gpu[:, :, 0] ** 2 + vector_difference_gpu[:, :, 1] ** 2 + vector_difference_gpu[:, :, 2] ** 2)  # Length in metres of the distance between the particle and its neighbour
            self.epe_gpu[self.frame] += 1/2 * self.k_gpu * (modulus_gpu - self.l0_gpu[repeat]) ** 2 # Add current elastic potential energy to this frame. (Frame only changes after 8 repeats)
            modulus_gpu = cp.expand_dims(modulus_gpu, 2)    # Float modulus to Cupy array of shape of pos_grid_gpu
            resultant_force_gpu += self.k_gpu * (modulus_gpu - self.l0_gpu[repeat]) * 1 / modulus_gpu * vector_difference_gpu + self.c_gpu * velocity_difference_gpu    # Combination of Hooke's law (F = k(x-l0)) and damping (F=cv). Note no minus due to directions of difference vectors
        self.resultant_force_gpu = (resultant_force_gpu + self.weight_arry_gpu) * self.ref_grid_gpu # Increase this timestep's resultant force, with Hooke's law, damping and weight

    def surface_tension(self):  # Finds force due to cohesive nature of fluids
        grad_x_gpu, grad_y_gpu = cp.gradient(self.pos_grid_gpu[:,:,2])  # Vector gradients, to give vectors parallel to surface
        normal_vec_gpu = cp.zeros_like(self.pos_grid_gpu)
        normal_vec_gpu[:, :, 0] = -grad_x_gpu   # Find local surface normals
        normal_vec_gpu[:, :, 1] = -grad_y_gpu
        normal_vec_gpu[:, :, 2] = 1
        mod_grad_z_gpu = cp.sqrt(grad_x_gpu ** 2 + grad_y_gpu ** 2 + 1)
        div_normal_vec_gpu = cp.gradient(normal_vec_gpu[:, :, 0], axis=0) +cp.gradient(normal_vec_gpu[:, :, 1], axis=1) # Divergence of normal vectors
        curvature_gpu = div_normal_vec_gpu / mod_grad_z_gpu # Find the curvature of the surface in 2 directions
        self.resultant_force_gpu -= self.sigma * curvature_gpu[:, :, cp.newaxis] * normal_vec_gpu # Calculate forces pointing in the direction of the local normals to the surface trying to make the surface as flat as possible

    @staticmethod # Numba is not friends with classes
    @njit(parallel=True) # Used to accelerate shift grid logic
    def quick_shift(pos_grid, coord_change, ref_grid, divisor, velocity):
        shift_pos = np.zeros((len(pos_grid[0]) + 2, len(pos_grid[1]) + 2, 3))   # Add a border around the known live and dead particles of fake particles, used for position and velocities, but not simulated
        for repeat in prange(8):    # All 8 neighbour directions
            for i in range(len(pos_grid[0])):   # Loop through x and y coordinates
                for j in range(len(pos_grid[1])):
                    if i + coord_change[repeat][0] < 0 or j + coord_change[repeat][1] < 0 or i + coord_change[repeat][0] > len(pos_grid[0]) - 1 or j + coord_change[repeat][1] > len(pos_grid[1]) - 1 or not ref_grid[i + coord_change[repeat][0]][j + coord_change[repeat][1]]: # If particle when it was a pixel is next to alpha or white or the wall of the png, then
                            shift_pos[i + coord_change[repeat][0] + 1][j + coord_change[repeat][1] + 1] = coord_change[repeat] * divisor + pos_grid[i][j]   # Create the positions of all non simulating pixels, so that the water surface can experience force due to walls or islands.
        return shift_pos


    @staticmethod # Numba and classes do not mix well
    @njit() # Make the weight calculation quick
    def weight(pool_mass, particle_number, g):
        return pool_mass/particle_number * g, pool_mass/particle_number # Returns weight in N and mass per particle in kg



    def verlet(self, coord_change, debug_verlet=False): # Simulation intergrator, chosen due to high energy conservation in kinematic situations, really Verlocity Verlet, based on https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/5/54/Skript_sim_methods_I.pdf
        self.mid_velocity_gpu = self.velocity_gpu + 0.5 * self.acceleration_gpu * self.deltaT    # Midpoint Velocity
        self.pos_grid_gpu = self.pos_grid_gpu + self.mid_velocity_gpu * self.deltaT  # Position at time +deltaT
        if not debug_verlet:    # Used to specifically unit test Verlet with basic kinematics when debug_verlet = True
            self.hookes_law(coord_change)   # Force due to Hooke's law, weight and damping, done in this order as to make verlocity verlety work with velocity dependant forces, see https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/5/54/Skript_sim_methods_I.pdf
            self.surface_tension()  # Force due to cohesive forces
            self.acceleration_gpu = self.resultant_force_gpu/self.mass_arry_gpu # Acceleration at  time + deltaT
        self.velocity_gpu = self.mid_velocity_gpu + 0.5 * self.acceleration_gpu * self.deltaT # Velocity at time + deltaT
        if not debug_verlet:
            self.kinetics_gpu[self.frame] = 0.5 * self.mass_arry * (self.velocity_gpu[:, :, 0] ** 2 + self.velocity_gpu[:, :, 1] ** 2 + self.velocity_gpu[:, :, 2] ** 2)    # Update energies for this frame
            self.gpe_gpu[self.frame] = -self.mass_arry * self.g_gpu * self.pos_grid_gpu[:, :, 2]
            self.frame += 1
        if debug_verlet:
            self.pos_grid = cp.asnumpy(self.pos_grid_gpu)   # Allow unit test numpy copies of arrays for easier access
            self.velocity = cp.asnumpy(self.velocity_gpu)


    def euler_richardson(self, coord_change, debug_er=False): # Simulation intergrator, chosen due to high energy conservation in kinematic situations, really Verlocity Verlet, based on https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/5/54/Skript_sim_methods_I.pdf
        self.mid_velocity_gpu = self.velocity_gpu
        if not debug_er:  # Used to specifically unit test Verlet with basic kinematics when debug_verlet = True
            self.hookes_law(
                coord_change)  # Force due to Hooke's law, weight and damping, done in this order as to make verlocity verlety work with velocity dependant forces, see https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/5/54/Skript_sim_methods_I.pdf
            self.surface_tension()  # Force due to cohesive forces
            self.acceleration_gpu = self.resultant_force_gpu / self.mass_arry_gpu  # Acceleration at  time + deltaT
        self.mid_velocity_gpu = self.velocity_gpu + 0.5 * self.acceleration_gpu * self.deltaT    # Midpoint Velocity
        self.pos_grid_gpu_store = self.pos_grid_gpu
        self.pos_grid_gpu = self.pos_grid_gpu + 0.5 * self.velocity_gpu * self.deltaT   # Mid position
        if not debug_er:  # Used to specifically unit test Verlet with basic kinematics when debug_verlet = True
            self.hookes_law(
                coord_change)  # Force due to Hooke's law, weight and damping, done in this order as to make verlocity verlety work with velocity dependant forces, see https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/5/54/Skript_sim_methods_I.pdf
            self.surface_tension()  # Force due to cohesive forces
            self.acceleration_gpu = self.resultant_force_gpu / self.mass_arry_gpu  # Acceleration at  time + 0.5 deltaT
        self.pos_grid_gpu = self.pos_grid_gpu_store + self.mid_velocity_gpu * self.deltaT  # Position at time + deltaT
        self.velocity_gpu = self.velocity_gpu + self.acceleration_gpu * self.deltaT  # Velocity at time + deltaT
        if not debug_er:
            self.kinetics_gpu[self.frame] = 0.5 * self.mass_arry * (self.velocity_gpu[:, :, 0] ** 2 + self.velocity_gpu[:, :, 1] ** 2 + self.velocity_gpu[:, :, 2] ** 2)    # Update energies for this frame
            self.gpe_gpu[self.frame] = -self.mass_arry * self.g_gpu * self.pos_grid_gpu[:, :, 2]
            self.frame += 1
        if debug_er:
            self.pos_grid = cp.asnumpy(self.pos_grid_gpu)   # Allow unit test numpy copies of arrays for easier access
            self.velocity = cp.asnumpy(self.velocity_gpu)