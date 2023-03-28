import numpy as np
from matplotlib import pyplot as plt
import Pool_Simulation as Pool_sim
import Grid_Creation_System as gcs
import CUDA_Calculation_Centre as ccc
import CPU_Calculation_Centre as cpu
import Settings_hub
import time


def Pool_Simulation_Setup(shape="circular", x_dim=100, y_dim=100):  # Launches class that finds pool image and defines basic attributes
    Pool_boundaries = Pool_sim.Pool_Simulation(shape=shape, x_size=x_dim, y_size=y_dim)
    Pool_boundaries.pool_boundary_creator()
    return Pool_boundaries  # Returns object containing pool image in RAM


def grid_setup(
        Pool_boundaries, settings):  # Handles conversion from image to usable grid points in a numpy array. Also creates reference grid that indicates where the simulation will have walls
    grid = gcs.grid_creation()
    grid.grid_for_shape(Pool_boundaries)
    plot_x = []
    plot_y = []
    for x in range(
            grid.width):  # Creates format specifically used for plotting (ie np.size = [x_dim, y_dim, xyz vector] to 1d lists of x and y
        for y in range(grid.height):
            if grid.ref_grid[x][y] == True:
                plot_x.append(grid.grid[x][y][0])
                plot_y.append(grid.grid[x][y][1])
    plt.scatter(plot_x, plot_y, s=2)
    plt.show()
    calculation_system(grid, Pool_boundaries, run_cuda=True,
                       mega_arrays=True, settings=settings)  # Due to lack of time run_cuda and mega_arrays must be True as CPU and lightweight alternatives were not developed further


def calculation_system(grid, pool, run_cuda, mega_arrays, settings):  # Main hub function that runs the calculation classes
    coord_change = np.array(
        [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]), np.array([1, 1, 0]),
         np.array([-1, 1, 0]), np.array([-1, -1, 0]),
         np.array([1, -1, 0])])  # Defined as grid coordinates to nearest neighbours to particles in all 8 directions
    divisor_w = 1 / grid.width * Pool_boundaries.x_size
    divisor_h = 1 / grid.height * Pool_boundaries.y_size
    divisor = np.array([divisor_w, divisor_h, 0],
                       dtype=np.float64)  # Definition of grid coordinates to physics coordinates
    l0 = np.array([(pool.x_size / len(grid.grid)), (pool.x_size / len(grid.grid)), (pool.y_size / len(grid.grid[0])),
                   (pool.y_size / len(grid.grid[0])),
                   np.sqrt(((pool.x_size / len(grid.grid))) ** 2 + ((pool.y_size / len(grid.grid[0]))) ** 2),
                   np.sqrt(((pool.x_size / len(grid.grid))) ** 2 + ((pool.y_size / len(grid.grid[0]))) ** 2),
                   np.sqrt(((pool.x_size / len(grid.grid))) ** 2 + ((pool.y_size / len(grid.grid[0]))) ** 2),
                   np.sqrt(((pool.x_size / len(grid.grid))) ** 2 + ((pool.y_size / len(grid.grid[0]))) ** 2)],
                  dtype=np.float64) # Defines original length for all 8 directions of springs for each particle
    velocity = np.zeros(np.shape(grid.grid), dtype=np.float64)  # Create empty velocity and acceleration arrays
    acceleration = np.zeros(np.shape(grid.grid), dtype=np.float64)
    ref_grid = grid.ref_grid
    if settings.mega_arrays == False:
        settings.VRAM = 2 * int(grid.grid.nbytes)
    if settings.CUDA:    # Only use cuda, cpu implementation needs time that I do not have
        calc = ccc.CUDA_Calculations(grid.grid, velocity, acceleration, settings.k, settings.sigma, l0, settings.c, coord_change, ref_grid,
                                     divisor, settings.pool_mass, settings.g, settings.mega_arrays,
                                     settings.deltaT, VRAM=settings.VRAM, integrator=settings.integrater)
    else:
        calc = cpu.CPU_Calculations(grid.grid, velocity, acceleration, settings.k, settings.sigma, l0, settings.c,
                                     coord_change, ref_grid,
                                     divisor, settings.pool_mass, settings.g, settings.mega_arrays,
                                     settings.deltaT, RAM=settings.VRAM, integrator=settings.integrater)  # Code not optimised in the slightest, just uses a direct cpu implementation of gpu code
    start = time.time()
    np.savez_compressed("./Output/0ref", ref_grid)  # Save extra simulation information for plotting, 0 and 1 before file name to make sure files are opened first
    np.savez("./Output/1_time_step", settings.deltaT)
    for x in range(settings.repeats):    # Main loop
        position, energies = calc.runner(coord_change)  #Runs calculations each repeat
        number = "" # For alphabetical I/O handling
        while len(number) + len(str(x)) < len(str(settings.repeats)):    # Makes sure that for a single run, all files are alphabetically in order by fixing number length. (Ie for 1000 repeats, the file for repeat 34 is 0034)
            number += "0"
        number += str(x)
        if mega_arrays:     # Saves all data to disk to use later. Saves in repeat chunks, which is good for both RAM management and power failure/crash data loss mitigation
            np.savez_compressed("./Output/mega_array_pos_" + number, position)
            np.savez_compressed("./Output/mega_array_eng_" + number, energies)
        else:   # Saves data for mini arrays
            np.savez_compressed("./Output/mini_array_pos_" + number, position)
            np.savez_compressed("./Output/mini_array_eng_" + number, energies)

    print(time.time() - start)  # Time to run all calculations in seconds


if __name__ == "__main__":
    settings = Settings_hub.Settings()
    settings.read_settings()
    Pool_boundaries = Pool_Simulation_Setup(shape=settings.shape, x_dim=settings.x_scale, y_dim=settings.y_scale)
    grid_setup(Pool_boundaries, settings)
