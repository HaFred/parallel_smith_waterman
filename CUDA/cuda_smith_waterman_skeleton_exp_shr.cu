/**
 * Name: HONG Ziyang
 * Student id: 20528735
 * ITSC email: zhongad@connect.ust.hk
 *
*/

#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <fstream>

using namespace std;

#include "cuda_smith_waterman.h"	
/*
 *  You can add helper functions and variables as you wish.
 */
// #define dev_idx(x, y, n) utils::dev_idx(x, y, n)
using namespace utils;

//expanding
__global__ void expand_score(int a_len, int b_len, int *d_score, int *d_local_max, char *d_a, char *d_b, int re_row, int col_amount) { // re_row device dia rank, re_i
	int threadID = blockDim.x * blockIdx.x + threadIdx.x; 		
	int num_threads = blockDim.x * gridDim.x; // for 1D, how many threads & blocks per have
	int re_i = re_row; // realigned i j
	for (int u = threadID; u < col_amount - 2; u += num_threads) {
		int re_j = u + 1; 
		int j = re_j;		
		int i = re_row - j; // i j are original ele index
		// realign the score for coalesced access
		d_score[dev_idx(re_i, re_j, a_len + 1)] = max(0,
							  max(d_score[dev_idx(re_i - 2, re_j - 1, a_len + 1)] + sub_mat(d_a[i - 1], d_b[j - 1]),
							  max(d_score[dev_idx(re_i - 1, re_j, a_len + 1)] - GAP,
							  d_score[dev_idx(re_i - 1, re_j - 1, a_len + 1)] - GAP)));
		// }
		d_local_max[threadID] = max(d_local_max[threadID], d_score[dev_idx(re_i, re_j, a_len + 1)]);	
	}
}

// not doing multi-for loop in the kernel functions, so no need to do _syncthreads()

// keep & shrinking	
__global__ void shrink_score(int a_len, int b_len, int *d_score, int *d_local_max, char *d_a, char *d_b, int re_row, int col_amount) { // d_rerow device dia rank, re_i
	int threadID = blockDim.x * blockIdx.x + threadIdx.x; 		
	int num_threads = blockDim.x * gridDim.x;	
	int re_i = re_row; 
	if (re_row <= b_len) { //keeping
		for (int u = threadID; u < col_amount - 1; u += num_threads) { 
			int re_j = u;	
			int i = col_amount - 1 - u; // i j are original ele index
			int j = re_row - i;		
			// realign the score for coalesced access
			if (re_i == a_len + 1) {
			d_score[dev_idx(re_i, re_j, a_len + 1)] = max(0,
								  max(d_score[dev_idx(re_i - 2, re_j, a_len + 1)] + sub_mat(d_a[i - 1], d_b[j - 1]),
								  max(d_score[dev_idx(re_i - 1, re_j, a_len + 1)] - GAP,
								  d_score[dev_idx(re_i - 1, re_j + 1, a_len + 1)] - GAP)));
			} else {
				d_score[dev_idx(re_i, re_j, a_len + 1)] = max(0,
								  max(d_score[dev_idx(re_i - 2, re_j + 1, a_len + 1)] + sub_mat(d_a[i - 1], d_b[j - 1]),
								  max(d_score[dev_idx(re_i - 1, re_j, a_len + 1)] - GAP,
								  d_score[dev_idx(re_i - 1, re_j + 1, a_len + 1)] - GAP)));					  							  
			}
			d_local_max[threadID] = max(d_local_max[threadID], d_score[dev_idx(re_i, re_j, a_len + 1)]);
		}
	} else { //shrinking
		for (int u = threadID; u < col_amount; u += num_threads) { 
			int re_j = u;	
			int i = a_len - u; // i j are original ele index
			int j = re_row - i;		
			// realign the score for coalesced access
			if (a_len == b_len && re_row == a_len + 1) {
				d_score[dev_idx(re_i, re_j, a_len + 1)] = max(0,
								  max(d_score[dev_idx(re_i - 2, re_j, a_len + 1)] + sub_mat(d_a[i - 1], d_b[j - 1]),
								  max(d_score[dev_idx(re_i - 1, re_j, a_len + 1)] - GAP,
								  d_score[dev_idx(re_i - 1, re_j + 1, a_len + 1)] - GAP)));
			} else {
				d_score[dev_idx(re_i, re_j, a_len + 1)] = max(0,
								  max(d_score[dev_idx(re_i - 2, re_j + 1, a_len + 1)] + sub_mat(d_a[i - 1], d_b[j - 1]),
								  max(d_score[dev_idx(re_i - 1, re_j, a_len + 1)] - GAP,
								  d_score[dev_idx(re_i - 1, re_j + 1, a_len + 1)] - GAP)));
			}								  							  
			d_local_max[threadID] = max(d_local_max[threadID], d_score[dev_idx(re_i, re_j, a_len + 1)]);
		}
	}	
}

// keep & shrinking	with redundant mem free, to avoid OOM
__global__ void free_shrink_score(int a_len, int b_len, int *d_score, int *d_local_max, char *d_a, char *d_b, int re_row, int col_amount) { // d_rerow device dia rank, re_i
	int threadID = blockDim.x * blockIdx.x + threadIdx.x; 		
	int num_threads = blockDim.x * gridDim.x;	
	int re_i = re_row; 
	if (re_row + (a_len + 1 - 2) <= b_len) { //keeping
		for (int u = threadID; u < col_amount - 1; u += num_threads) { 
			int re_j = u;	
			int i = col_amount - 1 - u; // i j are original ele index
			int j = re_row + (a_len + 1 - 2) - i;		
			// realign the score for coalesced access
			if (re_i == a_len + 1 - (a_len + 1 - 2)) {
			d_score[dev_idx(re_i, re_j, a_len + 1)] = max(0,
								  max(d_score[dev_idx(re_i - 2, re_j, a_len + 1)] + sub_mat(d_a[i - 1], d_b[j - 1]),
								  max(d_score[dev_idx(re_i - 1, re_j, a_len + 1)] - GAP,
								  d_score[dev_idx(re_i - 1, re_j + 1, a_len + 1)] - GAP)));
			} else {
				d_score[dev_idx(re_i, re_j, a_len + 1)] = max(0,
								  max(d_score[dev_idx(re_i - 2, re_j + 1, a_len + 1)] + sub_mat(d_a[i - 1], d_b[j - 1]),
								  max(d_score[dev_idx(re_i - 1, re_j, a_len + 1)] - GAP,
								  d_score[dev_idx(re_i - 1, re_j + 1, a_len + 1)] - GAP)));					  							  
			}
			d_local_max[threadID] = max(d_local_max[threadID], d_score[dev_idx(re_i, re_j, a_len + 1)]);
		}
	} else { //shrinking
		for (int u = threadID; u < col_amount; u += num_threads) { 
			int re_j = u;	
			int i = a_len - u; // i j are original ele index
			int j = re_row + (a_len + 1 - 2) - i;		
			// realign the score for coalesced access
			if (a_len == b_len && re_row + (a_len + 1 - 2) == a_len + 1) {
				d_score[dev_idx(re_i, re_j, a_len + 1)] = max(0,
								  max(d_score[dev_idx(re_i - 2, re_j, a_len + 1)] + sub_mat(d_a[i - 1], d_b[j - 1]),
								  max(d_score[dev_idx(re_i - 1, re_j, a_len + 1)] - GAP,
								  d_score[dev_idx(re_i - 1, re_j + 1, a_len + 1)] - GAP)));
			} else {
				d_score[dev_idx(re_i, re_j, a_len + 1)] = max(0,
								  max(d_score[dev_idx(re_i - 2, re_j + 1, a_len + 1)] + sub_mat(d_a[i - 1], d_b[j - 1]),
								  max(d_score[dev_idx(re_i - 1, re_j, a_len + 1)] - GAP,
								  d_score[dev_idx(re_i - 1, re_j + 1, a_len + 1)] - GAP)));
			}								  							  
			d_local_max[threadID] = max(d_local_max[threadID], d_score[dev_idx(re_i, re_j, a_len + 1)]);
		}
	}	
}

__global__ void update_global_max (int *d_global_max, int *d_local_max) {
	int num_threads = blockDim.x * gridDim.x;
/* 	//debug
	int threadID = blockDim.x * blockIdx.x + threadIdx.x; 		
	printf("Report the thread d_local_max = %d from block %d, thread %d\n", d_local_max[threadID] , blockIdx.x, threadIdx.x); */
    if (threadIdx.x == 0){
        *d_global_max = INT_MIN;
        for (int i = 0; i < num_threads; i++){
            if (d_local_max[i] > *d_global_max){
                *d_global_max = d_local_max[i];
            }
        }
    }
}

//debug
/* __global__ void print_d_score (int *d_score, int num_diagonal, int a_len) {
	int threadID = blockDim.x * blockIdx.x + threadIdx.x; 		
	if (threadID == 0) {
		// printf("\t");
		// for (int i = 0; i < a_len + 1; i++) {
			// cout << b[i] << "\t";
		// }
		printf("\n");
		for (int i = 0; i < 20; i++) {
			// cout << a[i - 1] << "\t";
			for (int j = 0; j < a_len + 1; j++) {
						printf("%d\t", d_score[dev_idx(i, j, a_len + 1)]);
			}
			printf("\n");
		}	
	}	
} */
//debug
/* int print_h_score (int *h_score, int num_diagonal, int a_len) {
	std::ofstream outputf("output_cuda.txt", std::ofstream::out);
	// outputf << dist[0];
	// printf("\n");
	outputf << "\n";
	for (int i = 0; i < num_diagonal; i++) {
		// cout << a[i - 1] << "\t";
		for (int j = 0; j < a_len + 1; j++) {
					// printf("%d\t", d_score[dev_idx(i, j, a_len + 1)]);
			outputf << h_score[i * (a_len + 1) + j] << "\t";
		}
		// printf("\n");
		outputf << "\n";
	}	
	outputf << endl;
	return 0;
} */

int smith_waterman(int blocks_per_grid, int threads_per_block, char *a, char *b, int a_len, int b_len) {
	/*
	 *  Please fill in your codes here.
	 */
	dim3 blocks(blocks_per_grid);
	dim3 threads(threads_per_block);
	
	int num_diagonal = b_len + a_len + 1; // original diagonals, aka the re_row length. 0 padding, a_len + 1 & b_len + 1
	int num_threads = blocks_per_grid * threads_per_block;
	
	// realign the score matrix for coalesced access
	// int *score = (int *)malloc(sizeof(int) * (a_len + 1) * (b_len + 1));
	int *h_score = (int*)calloc(num_diagonal * (a_len + 1), sizeof(int));	
	char *d_a, *d_b;
	int *d_score, *d_local_max, *d_global_max, *global_max;	
	global_max = (int*) malloc(sizeof(int));
	cudaMalloc(&d_score, sizeof(int) * (num_diagonal * (a_len + 1)));
	cudaMalloc(&d_local_max, sizeof(int) * num_threads);
	cudaMalloc(&d_global_max, sizeof(int));
	cudaMemset(d_local_max, 0, sizeof(int) * num_threads);
	if (a_len > b_len) {
		cudaMalloc(&d_a, sizeof(char) * b_len);
		cudaMalloc(&d_b, sizeof(char) * a_len);
		cudaMemcpy(d_a, b, sizeof(char) * b_len, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, a, sizeof(char) * a_len, cudaMemcpyHostToDevice);
		int temp = b_len;
		b_len = a_len;
		a_len = temp;

	} else {
		cudaMalloc(&d_a, sizeof(char) * a_len);
		cudaMalloc(&d_b, sizeof(char) * b_len);
		cudaMemcpy(d_a, a, sizeof(char) * a_len, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, sizeof(char) * b_len, cudaMemcpyHostToDevice);
	}
    // cudaMemcpy(d_a, a, sizeof(char) * a_len, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_b, b, sizeof(char) * b_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_score, h_score, sizeof(int) * (num_diagonal * (a_len + 1)), cudaMemcpyHostToDevice);
		
	int re_row = 0; //readability
	int col_amount = 0;
	int shrink_len = 0;
	bool freeFlag = false;
	shrink_len = a_len + 1;
	
	//expanding Phase
	for (re_row = 2; re_row < a_len + 1; re_row++){ // fred: the re_row = dev_j is the rank_dia, row of realign		
		col_amount = re_row + 1;  
		expand_score <<< blocks, threads >>> (a_len, b_len, d_score, d_local_max, d_a, d_b, re_row, col_amount);
	}
	//shrinking Phase
	if (a_len > 11000 && b_len > 11000) { // if necessary free the redundant mem
		int *temp_d_score;
		freeFlag = true;
		shrink_len = 2;
		num_diagonal = num_diagonal - (a_len + 1 - 2);
		cudaMalloc(&temp_d_score, sizeof(int) * (num_diagonal * (a_len + 1)));
		cudaMemcpy(temp_d_score, d_score + (a_len - 1) * (a_len + 1), sizeof(int) * 2 * (a_len + 1), cudaMemcpyDeviceToDevice);
		cudaFree(d_score);
		// int *d_score = temp_d_score;
		for (re_row = shrink_len; re_row < num_diagonal; re_row++){	
			if (re_row > b_len - (a_len + 1 - 2))
				col_amount = num_diagonal - re_row;
			else
				col_amount = a_len + 1;
		free_shrink_score <<< blocks, threads >>> (a_len, b_len, temp_d_score, d_local_max, d_a, d_b, re_row, col_amount);
		cudaFree(temp_d_score);
		}
	} else { // not freeing the mem
		for (re_row = shrink_len; re_row < num_diagonal; re_row++){	
			if (re_row > b_len)
				col_amount = num_diagonal - re_row;
			else
				col_amount = a_len + 1;
		shrink_score <<< blocks, threads >>> (a_len, b_len, d_score, d_local_max, d_a, d_b, re_row, col_amount);
		}
	}	
	//update
	update_global_max <<< blocks, threads >>> (d_global_max, d_local_max);
	cudaMemcpy(global_max, d_global_max, sizeof(int), cudaMemcpyDeviceToHost);
	//debug
	// print_d_score <<< blocks, threads >>> (d_score, num_diagonal, a_len);
	// cudaMemcpy(h_score, d_score, sizeof(int) * (num_diagonal * (a_len + 1)), cudaMemcpyDeviceToHost);
	// print_h_score(h_score, num_diagonal, a_len);
	free(h_score);
	cudaFree(d_a);
	cudaFree(d_b);
	if (freeFlag == false)
		cudaFree(d_score);
	// else {
		// cudaFree(d_score);
	// }
	cudaFree(d_local_max);
	cudaFree(d_global_max);
	 
	return *global_max;
}