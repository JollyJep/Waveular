import numpy as np
import numba
from numba import njit
from numba import cuda
import scipy
import pandas as pd
from matplotlib import pyplot as plt

class object_kinematics:

    def __init__(self, k_object="sphere", density=5000, x_scale=1, y_scale=1, rotation=np.array([0, 0, 0]), height=1, velocity=np.NaN, kinematics=True):
        self.object = k_object
        self.density = density
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.rotation = rotation
        self.height_COM = height
        self.velocity = velocity
        self.kinematics = kinematics