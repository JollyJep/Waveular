from PIL import Image
import Pool_Simulation as Pool_sim
import Grid_Creation_System as gcs
from matplotlib import pyplot as plt
import numpy as np
from numba import jit, njit, prange


def Pool_Simulation_Setup(shape="circular", x_dim=100, y_dim=100, z_dim=1, viscosity=1, density=1, debug=True): #Initialises pool with debug images
    Pool_boundaries = Pool_sim.Pool_Simulation(shape=shape, x_size=x_dim, y_size=y_dim, debug=debug)
    Pool_boundaries.pool_boundary_creator()
    return Pool_boundaries


def grid_setup(Pool_boundaries): # Initialises and creates grid from pngs
    grid = gcs.grid_creation()
    grid.grid_for_shape(Pool_boundaries)
    return grid

@njit(parallel=True)
def grid_to_png(ref_grid):  # Reformats grids to images
    test_png = np.zeros((len(ref_grid), len(ref_grid[0]), 4), dtype=np.uint8)
    for i in prange(len(ref_grid)): # Looping through all pixels
        for j in range(len(ref_grid[0])):
            if ref_grid[i, j]:
                test_png[i, j] = np.array([255, 0, 0, 255]) # If active particle, colour in red
            else:
                test_png[i, j] = np.array([255, 255, 255, 0]) # else colour in white alpha
    return test_png


def quick_analysis(test_png, original_png): # Tests pixels in newly created image to original. Black and red pixels are detected as the same as due to how the main code works they are functionally identical
    num_pixels = len(original_png) * len(original_png[0])   # Total number of pixels in both images
    num_correct = 0 # Number of pixels that are corrected in the new png as compared to the original
    for i in prange(len(original_png)):
        for j in range(len(original_png[0])):
            if np.isclose(original_png[i, j], test_png[i, j], atol=20).all():   #tests for red pixels in original and test image
                num_correct += 1
            elif (original_png[i, j] == np.array([0, 0, 0, 255])).all() and (test_png[i, j] == np.array([255, 0, 0, 255])).all(): #tests for fully black border pixel particles
                num_correct += 1
    return num_correct, num_pixels


def unit_test_main(shapes):
    for shape in shapes:
        pool_boundaries = Pool_Simulation_Setup(shape)
        grid = grid_setup(pool_boundaries)
        test_png = grid_to_png(grid.ref_grid)
        original_png = np.asarray(pool_boundaries.boundary.convert("RGBA")) # Load original image for testing against
        num_correct, num_pixels = quick_analysis(test_png, original_png)
        print(f"\n----------------------------------------\nFor {shape}.png, the grid is correct in {num_correct} out of {num_pixels} pixels, resulting in {num_correct/num_pixels * 100}% of correct grid positions\n----------------------------------------\n")


shapes = ["Circle", "Squiggle"] # Filenames without extensions to test
unit_test_main(shapes)
