import numpy as np
import PIL
from PIL import Image

class Pool_Simulation:
    def __init__(self, shape="circular", x_size=1.0, y_size=1.0, depth=1.0, viscosity=0.01, density=1000, precision=10000, debug = False):
        self.shape = shape
        self.x_size = x_size
        self.y_size = y_size
        self.depth = depth
        self.viscosity = viscosity
        self.density = density
        self.pool_precision = precision
        self.debug = debug
        #self.initial_conditions = initial_conditions

    def __str__(self):
        return "Shape: {0}, X dimensions: {1}, Y dimensions: {2}, Depth: {3}, Viscosity: {4}, Density: {5}".format(
            self.shape, self.x_size, self.y_size, self.depth, self.viscosity, self.density
        )

    def pool_boundary_creator(self):
        primitives = ["circular"]
        if self.shape in primitives:
            if self.shape == "circular":
                self.boundary = Image.open("./Grid_Images/Circle.png")
            #if self.shape == "rectangle":
            #    self.boundary = np.array([np.array([-self.x_size, -self.x_size, self.x_size, self.x_size, -self.x_size]), np.array([-self.y_size, self.y_size, self.y_size, -self.y_size, -self.y_size])])
        elif self.debug:
            self.boundary = Image.open("./Grid_Images/", self.shape, ".png")
        else:
            file = input("What is the file name, excluding extension?\n")
            self.boundary = Image.open("./Grid_Images/", file, ".png")

