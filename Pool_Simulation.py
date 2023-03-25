import numpy as np
import PIL
from PIL import Image


class Pool_Simulation:
    '''
    ----------------------------------------------------------------
    Description:
    Pool_Simulation is used to define the png used as the pool shape
    ----------------------------------------------------------------
    Inputs: shape (str) - Defines if a basic shape is used or a custom shape
            debug (bool) - Launches unit test environment if True, isolates
                processes without automation
    ----------------------------------------------------------------
    Outputs: self.boundary (PIL Image) - Image loaded from disk to memory for further usage
    ----------------------------------------------------------------
    '''

    def __init__(self, shape="circular", debug=False):
        self.shape = shape
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
