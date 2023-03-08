import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import scipy
import pandas as pd
from matplotlib import pyplot as plt
import PIL
import math
import sys
from importlib import reload
reload(sys)

class CUDA_Calculations:

    def __init__(self):
        return

    def runner(self, output, pos_grid, k, l0, vector_difference, force_local, ref_grid, coord_change, divisor, velocity,
               c):
        constants = np.array([k, c])
        print(divisor)
        vector_difference = np.zeros(np.shape(output))
        force_local = np.zeros(np.shape(output))
        modulus = np.array(0, dtype=np.float64)

        # Allocate gpu memory
        output_gpu = cuda.mem_alloc(output.nbytes)
        pos_grid_gpu = cuda.mem_alloc(pos_grid.nbytes)
        constants_gpu = cuda.mem_alloc(constants.nbytes)
        l0_gpu = cuda.mem_alloc(l0.nbytes)
        ref_grid_gpu = cuda.mem_alloc(ref_grid.nbytes)
        coord_change_gpu = cuda.mem_alloc(coord_change.nbytes)
        divisor_gpu = cuda.mem_alloc(divisor.nbytes)
        velocity_gpu = cuda.mem_alloc(velocity.nbytes)
        vector_gpu = cuda.mem_alloc(vector_difference.nbytes)
        force_gpu = cuda.mem_alloc(force_local.nbytes)
        mod_gpu = cuda.mem_alloc(modulus.nbytes)

        # Copy variables from CPU RAM to VRAM
        cuda.memcpy_htod(output_gpu, output)
        cuda.memcpy_htod(pos_grid_gpu, pos_grid)
        cuda.memcpy_htod(constants_gpu, constants)
        cuda.memcpy_htod(l0_gpu, l0)
        cuda.memcpy_htod(ref_grid_gpu, ref_grid)
        cuda.memcpy_htod(coord_change_gpu, coord_change)
        cuda.memcpy_htod(divisor_gpu, divisor)
        cuda.memcpy_htod(velocity_gpu, velocity)
        cuda.memcpy_htod(vector_gpu, vector_difference)
        cuda.memcpy_htod(force_gpu, force_local)
        cuda.memcpy_htod(mod_gpu, modulus)
        line_1 = "__global__ void force_calc(float* output[{}][{}][3], float* pos_grid[{}][{}][3], float* constants[2], float* l0[8],  bool* ref_grid[{}][{}], int* coord_change[8][3], float* divisor[2], float* velocity[{}][{}][3], float* vector_difference[{}][{}][3], float* force_local[{}][{}][3], float* modulus[1])".format(len(output), len(output[0]),len(pos_grid), len(pos_grid[0]), len(ref_grid), len(ref_grid[0]), len(velocity), len(velocity[0]), len(vector_difference), len(vector_difference[0]), len(force_local), len(force_local[0]))
        force_calc = SourceModule((line_1 + """{
                    int idx = blockIdx.x * blockDim.x +threadIdx.x;
                	int idy = blockIdx.y * blockDim.y +threadIdx.y; //Cuda thread coordinates
                	for(int repeat =0; repeat<8; repeat++){;
                	    if(idx > sizeof(pos_grid[0])){
                
                	    }
                	    else{
                	    if(*ref_grid[idx][idy] == false){
                	    }
                	    else{
                	    float pos_grid_size_0 = static_cast<float>(sizeof(pos_grid[0]));
                	    float pos_grid_size_1 = static_cast<float>(sizeof(pos_grid[1]));
                        if(idx + static_cast<float>(*coord_change[repeat][0]) < 0 || idy + static_cast<float>(*coord_change[repeat][1]) < 0 || idx + static_cast<float>(*coord_change[repeat][0]) > pos_grid_size_0 - 1 || idy + static_cast<float>(*coord_change[repeat][1]) > pos_grid_size_1 - 1 || *ref_grid[idx + *coord_change[repeat][0]][idy + *coord_change[repeat][1]] == false){
                            *vector_difference[idx][idy][0] = static_cast<float>(*coord_change[repeat][0]) * *divisor[0];
                            *vector_difference[idx][idy][1] = static_cast<float>(*coord_change[repeat][1]) * *divisor[1];
                            *vector_difference[idx][idy][2] = *pos_grid[idx][idy][2];
                	    }
                	    else{
                	        *vector_difference[idx][idy][0] = *pos_grid[idx + *coord_change[repeat][0]][idy + *coord_change[repeat][1]][0] - *pos_grid[idx][idy][0];
                	        *vector_difference[idx][idy][1] = *pos_grid[idx + *coord_change[repeat][0]][idy + *coord_change[repeat][1]][1] - *pos_grid[idx][idy][1];
                	        *vector_difference[idx][idy][2] = *pos_grid[idx + *coord_change[repeat][0]][idy + *coord_change[repeat][1]][2] - *pos_grid[idx][idy][2];
                	    }
                	    float modulus;
                	    modulus = pow(pow(*vector_difference[idx][idy][0], 2) + pow(*vector_difference[idx][idy][1], 2) + pow(*vector_difference[idx][idy][2],2), 0.5);
                	    *force_local[idx][idy][0] = *constants[0] * (modulus -*l0[repeat]) * *vector_difference[idx][idy][0] * 1/ modulus; //c++ doesn't have native vector support, hence single value multiplication
                	    *force_local[idx][idy][1] = *constants[0] * (modulus -*l0[repeat]) * *vector_difference[idx][idy][1] * 1/ modulus;
                	    *force_local[idx][idy][2] = *constants[0] * (modulus -*l0[repeat]) * *vector_difference[idx][idy][2] * 1/ modulus;
                        *output[idx][idy][0] = *output[idx][idy][0] + *force_local[idx][idy][0];
                        *output[idx][idy][1] = *output[idx][idy][1] + *force_local[idx][idy][1];
                        *output[idx][idy][2] = *output[idx][idy][2] + *force_local[idx][idy][2];
                	}
                	}
                	}
                	
                }
                """))
        run = force_calc.get_function("force_calc")
        blockdim = (3, 3, 1)
        griddim = (1, 1)
        run(output_gpu, pos_grid_gpu, constants_gpu, l0_gpu, ref_grid_gpu, coord_change_gpu, divisor_gpu, velocity_gpu, vector_gpu, force_gpu, block=blockdim, grid=griddim)
        cuda.memcpy_dtoh(l0, l0_gpu)
        print(output)
    # @staticmethod
    # def hookes_law(output, pos_grid, k, l0, vector_difference,  Force_local, ref_grid, coord_change, divisor, velocity, c):

    def weight_calc(mass, g):
        return mass * g
