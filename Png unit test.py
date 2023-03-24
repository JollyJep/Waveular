from PIL import Image
import Pool_Simulation as Pool_sim
import Grid_Creation_System as gcs
from matplotlib import pyplot as plt
import numpy as np
from numba import jit, njit, prange


def Pool_Simulation_Setup(shape="circular", x_dim=100, y_dim=100, z_dim=1, viscosity=1, density=1):
    Pool_boundaries = Pool_sim.Pool_Simulation(shape=shape, x_size=x_dim, y_size=y_dim, depth=z_dim, viscosity=viscosity, density=density)
    Pool_boundaries.pool_boundary_creator()
    return Pool_boundaries


def grid_setup(Pool_boundaries):
    grid = gcs.grid_creation(Pool_boundaries.x_size, Pool_boundaries.y_size)
    grid.grid_for_shape(Pool_boundaries)
    return grid
    #plot_x = []
    #plot_y = []
    #for x in range(grid.width):
    #    for y in range(grid.height):
    #        if grid.ref_grid[x][y] == True:
    #            plot_x.append(grid.grid[x][y][0])
    #            plot_y.append(grid.grid[x][y][1])
    #plt.scatter(plot_x, plot_y, s=2)
    #plt.show()


@njit(parallel=True)
def grid_to_png(ref_grid):
    test_png = np.zeros((len(ref_grid), len(ref_grid[0]), 4), dtype=np.uint8)
    for i in prange(len(ref_grid)):
        for j in range(len(ref_grid[0])):
            if ref_grid[i, j]:
                test_png[i, j] = np.array([255, 0, 0, 255])
            else:
                test_png[i, j] = np.array([255, 255, 255, 255])
    return test_png


def unit_test_main(shapes):
    for shape in shapes:
        pool_boundaries = Pool_Simulation_Setup(shape)
        grid = grid_setup(pool_boundaries)
        test_png = grid_to_png(grid.ref_grid)

