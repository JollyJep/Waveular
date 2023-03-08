__global__ void force_calc(float* output[1001][1001][3], float* pos_grid[1001][1001][3], float* constants[2], float* l0[8],  bool* ref_grid[1001][1001], int* coord_change[8][3], float* divisor[2], float* velocity[1001][1001][3], float* vector_difference[1001][1001][3], float* force_local[1001][1001][3], float* modulus[1]){
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
	    *modulus = pow(pow(*vector_difference[idx][idy][0], 2) + pow(*vector_difference[idx][idy][1], 2) + pow(*vector_difference[idx][idy][2],2), 0.5);
	    *force_local[idx][idy][0] = *constants[0] * (*modulus -*l0[repeat]) * *vector_difference[idx][idy][0] * 1/ modulus; //c++ doesn't have native vector support, hence single value multiplication
	    *force_local[idx][idy][1] = *constants[0] * (*modulus -*l0[repeat]) * *vector_difference[idx][idy][1] * 1/ modulus;
	    *force_local[idx][idy][2] = *constants[0] * (*modulus -*l0[repeat]) * *vector_difference[idx][idy][2] * 1/ modulus;
        *output[idx][idy][0] = *output[idx][idy][0] + *force_local[idx][idy][0];
        *output[idx][idy][1] = *output[idx][idy][1] + *force_local[idx][idy][1];
        *output[idx][idy][2] = *output[idx][idy][2] + *force_local[idx][idy][2];
	}
	}
	}
	
}