import numpy as np
from numba import njit, jit, prange


class grid_creation:

    def __init__(self, max_grid_size_x=1000, max_grid_size_y=1000):
        self.grid_size_x = max_grid_size_x
        self.grid_size_y = max_grid_size_y

    def grid_for_shape(self, pool):
        shape = pool.boundary
        img = shape.convert("RGBA")
        self.width, self.height = img.size
        pixels = np.asarray(img)
        grid_x = np.full((self.width, self.height), 0)
        grid_y = np.full((self.width, self.height), 0)
        ref_grid = np.full((self.width, self.height), True)
        grid_x, grid_y, ref_grid = self.quick_pixel(self.width, self.height, pixels, grid_x, grid_y, ref_grid)
        grid = np.zeros((self.width, self.height, 3))
        divisor = np.zeros((self.width, self.height, 3))
        for x in range(self.width):
            for y in range(self.width):
                grid[x][y] = np.array([grid_x[x][y], grid_y[x][y], 0])
                divisor[x][y] = np.array([1/self.width * pool.x_size, 1/self.width * pool.x_size, 1])
        self.grid = grid * divisor
        self.grid_base = grid
        self.ref_grid = ref_grid


    @staticmethod
    @njit(parallel=True)
    def quick_pixel(width, height, pixels, grid_x, grid_y, ref_grid):
        for x in prange(width):
            for y in range(height):
                if pixels[x, y][0] < 5 and pixels[x, y][3] == 255:
                    grid_x[x][y] = x
                    grid_y[x][y] = y
                    ref_grid[x][y] = True
                elif pixels[x, y][0] >= 5 and pixels[x, y][3] == 255 and pixels[x, y][1] == 0:
                    grid_x[x][y] = x
                    grid_y[x][y] = y

                    ref_grid[x][y] = True
                else:
                    grid_x[x][y] = 0
                    grid_y[x][y] = 0
                    ref_grid[x][y] = False
        return grid_x, grid_y, ref_grid


