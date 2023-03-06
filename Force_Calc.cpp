__global__ void force_calc(float* output, float* pos_grid, float* constants, bool* ref_grid, float* coord_change, float* divisor, float* velocity){
	int idx = blockIdx.x * blockDim.x +threadIdx.x;
	int idy = blockIdx.y * blockDim.y +threadIdx.y; //Cuda thread coordinates
	float vector_difference[3] = {0, 0, 0};
	float force_local[3] = {0, 0, 0};
	for(int repeat =0; repeat<8; repeat++){
	    if(idx > sizeof(pos_grid[0])){
	    return void;
	    }
	    if(ref_grid[idx][idy] == false){
	    return void;
	    }
	    if(idx + coord_change[repeat][0] < 0 || idy + coord_change[repeat][1] < 0 || idx+coord_change[repeat][0] > sizeof(pos_grid[0]) - 1 || idy + coord_change[repeat][1] > sizeof(pos_grid[1]) - 1 || !(ref_grid[idx+coord_change[repeat][0]][idy+coord_change[repeat][1]])){
            vector_difference = {coord_change[repeat][0] * divisor[0], coord_change[repeat][0] * divisor[1], pos_grid[idx][idy][2]};
	    }
	    else{
	        vector_difference = {pos_grid[idx+coord_change[repeat][0]][idy + coord_change[repeat][1]][0] - pos_grid[idx][idy][0], pos_grid[idx+coord_change[repeat][0]][idy + coord_change[repeat][1]][1] - pos_grid[idx][idy][1], pos_grid[idx+coord_change[repeat][0]][idy + coord_change[repeat][1]][2] - pos_grid[idx][idy][2]};
	    }
	    float modulus;
	    modulus = pow(pow(vector_difference[0] ** 2 + vector_difference[1], 2) + pow(vector_difference[2],2), 0.5);
	    force_local[0] = constants[0] * (modulus -constants[1][repeat]) * vector_difference[0] * 1/ modulus; //c++ doesn't have native vector support, hence single value multiplication
	    force_local[1] = constants[0] * (modulus -constants[1][repeat]) * vector_difference[1] * 1/ modulus;
	    force_local[2] = constants[0] * (modulus -constants[1][repeat]) * vector_difference[2] * 1/ modulus;
        output[i][j][0] = output[i][j][0] + Force_local[0];
        output[i][j][1] = output[i][j][1] + Force_local[1];
        output[i][j][2] = output[i][j][2] + Force_local[2];
	}
	
}