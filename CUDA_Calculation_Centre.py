import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import scipy
import pandas as pd
from matplotlib import pyplot as plt
import PIL
import math



class CUDA_Calculations:

    def __init__(self):
        return

    def runner(self, output, pos_grid, k, l0, vector_difference,  force_local, ref_grid, coord_change, divisor, velocity, c):
        constants = np.array([k, l0, c])
        print(pos_grid)
        exit()
        #Allocate gpu memory
        output_gpu = cuda.mem_alloc(output.nbytes)
        pos_grid_gpu = cuda.mem_alloc(pos_grid.nbytes)
        constants_gpu = cuda.mem_alloc(constants.nbytes)
        vector_difference_gpu = cuda.mem_alloc(vector_difference.nbytes)
        force_local_gpu = cuda.mem_alloc(force_local.nbytes)
        ref_grid_gpu = cuda.mem_alloc(ref_grid.nbytes)
        coord_change_gpu = cuda.mem_alloc(coord_change.nbytes)
        divisor_gpu = cuda.mem_alloc(divisor.nbytes)
        velocity_gpu = cuda.mem_alloc(velocity.nbytes)

        #Copy variables from CPU RAM to VRAM
        cuda.memcpy_htod(output_gpu, output)
        cuda.memcpy_htod(pos_grid_gpu, pos_grid)
        cuda.memcpy_htod(constants_gpu, constants)
        cuda.memcpy_htod(vector_difference_gpu, vector_difference)
        cuda.memcpy_htod(force_local_gpu, force_local)
        cuda.memcpy_htod(ref_grid_gpu, ref_grid)
        cuda.memcpy_htod(coord_change_gpu, coord_change)
        cuda.memcpy_htod(divisor_gpu, divisor)
        cuda.memcpy_htod(velocity_gpu, velocity)

        force_calc = SourceModule(("""
                
        
        
        
                """))






    #@staticmethod
    #def hookes_law(output, pos_grid, k, l0, vector_difference,  Force_local, ref_grid, coord_change, divisor, velocity, c):





    def weight_calc(mass, g):
        return mass * g