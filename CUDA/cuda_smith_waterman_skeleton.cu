#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>
#include "cuda_smith_waterman.h"

using namespace std;

__global__ void kernel(int *score_prev, int *score, int *score_next, int i, int offset, int range, int width, char *a, char *b, int a_len, int b_len, int *local_max) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;   // fred: just need these prev_dia, this_dia, next_dia rather than whole score, and we calculate the next_dia
    int num_threads = blockDim.x * gridDim.x;

    for (int j = offset + tid; j < offset + range; j += num_threads) {
        score_next[j] = max(0,
                        max(score_prev[j - 1] + sub_mat(a[i - j], b[j - 1]),
                        max(score[j - 1] - GAP,
                            score[j] - GAP)));
        local_max[tid] = max(local_max[tid], score_next[j]);
    }
}

int smith_waterman(int blocks_per_grid, int threads_per_block, char *a, char *b, int a_len, int b_len) {
    char *d_a, *d_b;
    int *d_score_prev, *d_score, *d_score_next;
    int *d_max;

    int n_diagonal = a_len + b_len - 1;
    int width = min(a_len, b_len);
    int *h_max = (int *)malloc(sizeof(int) * blocks_per_grid * threads_per_block);

    GPUErrChk(cudaMalloc(&d_score_prev, sizeof(int) * (width + 1)));
    GPUErrChk(cudaMalloc(&d_score, sizeof(int) * (width + 1)));
    GPUErrChk(cudaMalloc(&d_score_next, sizeof(int) * (width + 1)));
    GPUErrChk(cudaMalloc(&d_a, sizeof(char) * a_len));
    GPUErrChk(cudaMalloc(&d_b, sizeof(char) * b_len));
    GPUErrChk(cudaMalloc(&d_max, sizeof(int) * blocks_per_grid * threads_per_block));

    GPUErrChk(cudaMemset(d_score_prev, 0, sizeof(int) * (width + 1)));
    GPUErrChk(cudaMemset(d_score, 0, sizeof(int) * (width + 1)));
    GPUErrChk(cudaMemcpy(d_a, a, sizeof(char) * a_len, cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemcpy(d_b, b, sizeof(char) * b_len, cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemset(d_max, 0, sizeof(int) * blocks_per_grid * threads_per_block));

    for (int i = 1; i <= n_diagonal; i++) {
        int offset = max(0, i - max(a_len, b_len)) + 1;
        int range = min(min(i, n_diagonal - i + 1), width); // the shape of the new score, go up & down
        
        GPUErrChk(cudaMemset(d_score_next, 0, sizeof(int) * (width + 1)));
        kernel<<<blocks_per_grid, threads_per_block>>>(d_score_prev, d_score, d_score_next, i, offset, range, width, d_a, d_b, a_len, b_len, d_max);
        
        int *tmp = d_score_prev;
        d_score_prev = d_score;
        d_score = d_score_next;
        d_score_next = tmp;      // change the ptr linking, not to interfere with the value being done on these 3 memory snippets
    }
    GPUErrChk(cudaMemcpy(h_max, d_max, sizeof(int) * blocks_per_grid * threads_per_block, cudaMemcpyDeviceToHost));
    int max_val = *std::max_element(h_max, h_max + blocks_per_grid * threads_per_block); // mem level max

    cudaFree(d_score_prev);
    cudaFree(d_score);
    cudaFree(d_score_next);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_max);

    free(h_max);

    return max_val;
}