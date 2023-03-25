
from PIL import Image


class Pool_Simulation:
    """
    ----------------------------------------------------------------
    Description:
    Pool_Simulation is used to define the png used as the pool shape.
    ----------------------------------------------------------------
    Inputs: shape (str) - Defines if a basic shape is used or a custom shape.
            x_size (float) - x length of simulation.
            y_size (float) - y length of simulation.
            debug (bool) - Launches unit test environment if True, isolates
                           processes without automation.
    ----------------------------------------------------------------
    Outputs: self.boundary (PIL Image) - Image loaded from disk to memory for further usage.
    ----------------------------------------------------------------
    """

    def __init__(self, shape="circular", x_size=1.0, y_size=1.0, debug=False):
        self.shape = shape
        self.x_size = x_size
        self.y_size = y_size
        self.debug = debug

    def pool_boundary_creator(self):
        primitives = ["circular"]  # list of inbuilt pools
        if self.shape in primitives:
            if self.shape == "circular":
                self.boundary = Image.open("./Grid_Images/Circle.png")
        elif self.debug:
            self.boundary = Image.open("./Grid_Images/" + self.shape + ".png")  # Automates image process for unit test
        else:
            file = input("What is the file name, excluding extension?\n")
            self.boundary = Image.open("./Grid_Images/" + file + ".png")
