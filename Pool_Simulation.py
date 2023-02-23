import numpy as np

class Pool_Simulation:
    def __init__(self, shape="circular", x_size=1.0, y_size=1.0, depth=1.0, viscosity=0.01, density=1000, precision=10000):
        self.shape = shape
        self.x_size = x_size
        self.y_size = y_size
        self.depth = depth
        self.viscosity = viscosity
        self.density = density
        self.pool_precision = precision
        #self.initial_conditions = initial_conditions

    def __str__(self):
        return "Shape: {0}, X dimensions: {1}, Y dimensions: {2}, Depth: {3}, Viscosity: {4}, Density: {5}".format(
            self.shape, self.x_size, self.y_size, self.depth, self.viscosity, self.density
        )

    def pool_boundary_creator(self):
        primitives = ["circular", "rectangle"]
        if self.shape in primitives:
            if self.shape == "circular":
                theta = np.linspace(0, 2*np.pi, self.pool_precision)
                self.boundary = self.circle_to_numpy(theta)
            if self.shape == "rectangle":
                self.boundary = np.array([np.array([-self.x_size, -self.x_size, self.x_size, self.x_size, -self.x_size]), np.array([-self.y_size, self.y_size, self.y_size, -self.y_size, -self.y_size])])
        else:
            print("Arbritary pools not yet implemented")

    def circle_to_numpy(self, theta):
        x = self.x_size*np.cos(theta)
        y = self.y_size*np.sin(theta)
        return np.array([x, y])