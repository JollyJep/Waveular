class Pool_Simulation:
    def __init__(self, shape="circular", x_size=1.0, y_size=1.0, depth=1.0, viscosity=0.01, density=1000):
        self.shape = shape
        self.x_size = x_size
        self.y_size = y_size
        self.depth = depth
        self.viscosity = viscosity
        self.density = density

        #self.initial_conditions = initial_conditions

    def __str__(self):
        return "Shape: {0}, X dimensions: {1}, Y dimensions: {2}, Depth: {3}, Viscosity: {4}, Density: {5}".format(
            self.shape, self.x_size, self.y_size, self.depth, self.viscosity, self.density
        )

    def pool_boundary_creator(self):
        primitives = ["circular", "rectangle"]
        if self.shape in primitives:
            if self.shape == "circular":
                self.boundary = ["x**2+y**2 = {0}**2".format(self.x_size)]
            if self.shape == "rectangle":
                self.boundary = ["x=-{0}".format(self.x_size), "x={0}".format(self.x_size), "y=-{0}".format(self.y_size), "y={0}".format(self.y_size)]
        else:
            print("Arbritary pools not yet implemented")