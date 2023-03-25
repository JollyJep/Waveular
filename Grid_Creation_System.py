import numpy as np
from numba import njit, jit, prange


class grid_creation:
    """
        ----------------------------------------------------------------
        Description:
        Grid_Creation_System is used to convert from a png to a grid in the shape of the original png
        ----------------------------------------------------------------
        Inputs: pool (Object) - Contains initial attributes of the pool, including side dimensions and pool shape as an image
        ----------------------------------------------------------------
        Outputs: self.grid  (Numpy float64 array) - Array of points sitting in a grid (rectangular). Has format of np.size = [number of pixels in x, number of pixels in y, xyz position vector]
                self.ref_grid (Numpy bool array) - Array corresponding to shape of image from pool, used to determine what particles are stationary and what needs to be simulated
        ----------------------------------------------------------------
        """

    def grid_for_shape(self, pool):
        shape = pool.boundary  # Image from pool
        img = shape.convert("RGBA")  # Make sure the image is in a standard colour space
        self.width, self.height = img.size  # Pixel counts
        pixels = np.asarray(img)  # Make image easy to manipulate using numpy
        grid_x = np.full((self.width, self.height), 0)  # x and y linear 2d arrays
        grid_y = np.full((self.width, self.height), 0)
        ref_grid = np.full((self.width, self.height), True)  # Set up ref_grid to be the same shape as x and y
        grid_x, grid_y, ref_grid = self.quick_pixel(self.width, self.height, pixels, grid_x, grid_y, ref_grid)  # Sorts through and analyses each pixel. This is numba accelerated, as it was taking a massive amount of time otherwise
        grid = np.zeros((self.width, self.height, 3), np.float64)   # Creates final grid shape
        divisor = np.zeros((self.width, self.height, 3), np.float64)    # Converts from grid coordinates to particle coordinates
        for x in range(self.width): # Packs coordinates into grid shape
            for y in range(self.width):
                grid[x][y] = np.array([grid_x[x][y], grid_y[x][y], 0], np.float64)
                divisor[x][y] = np.array([1 / self.width * pool.x_size, 1 / self.width * pool.x_size, 1], np.float64)
        self.grid = grid * divisor  #Creates usable units for numbers at grid coordinates
        self.grid_base = grid   #not used
        self.ref_grid = ref_grid

    @staticmethod  # Numba hates classes, so functions must be used to appease it
    @njit(parallel=True)
    def quick_pixel(width, height, pixels, grid_x, grid_y, ref_grid):
        for x in prange(width): # prange indicates that for loop can be parallelised. Looping through every pixel in the image
            for y in range(height):
                if pixels[x, y][0] < 5 and pixels[x, y][3] == 255:  # detects black pixels
                    grid_x[x][y] = x
                    grid_y[x][y] = y
                    ref_grid[x][y] = True
                elif pixels[x, y][0] >= 5 and pixels[x, y][3] == 255 and pixels[x, y][1] == 0:  # detects only red pixels (ie not white)
                    grid_x[x][y] = x
                    grid_y[x][y] = y
                    ref_grid[x][y] = True
                else:   # Calls any white pixels or alpha pixels as fixed areas within the grid
                    grid_x[x][y] = x
                    grid_y[x][y] = y
                    ref_grid[x][y] = False
        return grid_x, grid_y, ref_grid
