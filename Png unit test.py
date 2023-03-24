from PIL import Image
import Pool_Simulation as Pool_sim
import Grid_Creation_System as gcs
from matplotlib import pyplot as plt
import numpy as np
from numba import jit, njit, prange


def Pool_Simulation_Setup(shape="circular", x_dim=100, y_dim=100, z_dim=1, viscosity=1, density=1, debug=True):
    Pool_boundaries = Pool_sim.Pool_Simulation(shape=shape, x_size=x_dim, y_size=y_dim, depth=z_dim, viscosity=viscosity, density=density, debug=debug)
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
                test_png[i, j] = np.array([255, 255, 255, 0])
    return test_png


@njit(parallel=True)
def quick_analysis(test_png, original_png):
    num_pixels = len(original_png) * len(original_png[0])
    num_correct = 0
    for i in prange(len(original_png)):
        for j in range(len(original_png[0])):
            if np.isclose(original_png[i, j], test_png[i, j], atol=20).all():
                num_correct += 1
    return num_correct, num_pixels


def unit_test_main(shapes):
    for shape in shapes:
        pool_boundaries = Pool_Simulation_Setup(shape)
        grid = grid_setup(pool_boundaries)
        test_png = grid_to_png(grid.ref_grid)
        original_png = np.asarray(pool_boundaries.boundary.convert("RGBA"))
        num_correct, num_pixels = quick_analysis(test_png, original_png)
        print(f"\n----------------------------------------\nFor {shape}.png, the grid is correct in {num_correct} out of {num_pixels} resulting in {num_correct/num_pixels * 100}% of correct grid positions\n----------------------------------------\n")


shapes = ["Circle"]
unit_test_main(shapes)
