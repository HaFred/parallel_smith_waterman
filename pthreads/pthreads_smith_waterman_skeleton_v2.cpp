#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <vector>
#include <queue>
#include <iostream>
#include <memory.h>
#include <cstring>

#include <pthread.h>
#include "pthreads_smith_waterman.h"

using namespace std;

int glb_max_value = 0;

struct ThreadParameters {
	int num_threads;
	int thread_id;

	int a_len;
	int b_len;

	char *a;
	char *b;
	int **score;
	int *max_score;

	pthread_barrier_t *barrier;
};


void *thread_work(void *t){
	auto *parameters = (ThreadParameters *) t;
	int num_threads = parameters->num_threads;
	int tid = parameters->thread_id;

	int a_len = parameters->a_len;
	int b_len = parameters->b_len;

	int block_l = tid * (b_len / num_threads) + 1;
	int block_r = block_l + (b_len / num_threads);

	char *a = parameters->a;
	char *b = parameters->b;
	int **score = parameters->score;
	int *max_score = parameters->max_score;

	int n_diagonal = a_len + num_threads - 1;
	max_score[tid] = -1;

	for (int i = 0; i <= n_diagonal; i++) {
		int row = i - tid;
		if (row >= 1 && row <= a_len) {
			for (int j = block_l; j <= block_r; j++) {
				score[row][j] = max(0,
						max(score[row - 1][j - 1] + sub_mat(a[row - 1], b[j - 1]),
								max(score[row - 1][j] - GAP,
										score[row][j - 1] - GAP)));
				max_score[tid] = max(max_score[tid], score[row][j]);
			}
		}
		pthread_barrier_wait(parameters->barrier);
	}

	if(tid == 0){
		glb_max_value = *std::max_element(max_score, max_score + num_threads);
	}
}

int smith_waterman(int num_threads, char *a, char *b, int a_len, int b_len){
	int **score = (int **)malloc(sizeof(int*) * (a_len + 1));
	for (int i = 0; i <= a_len; i++) {
		score[i] = (int*)calloc(b_len + 1, sizeof(int));
	}

	int *max_score = (int *) malloc(sizeof(int) * (num_threads));

	pthread_attr_t attr;
	vector<pthread_t> threads(num_threads);

	pthread_barrier_t barrier;
	pthread_barrier_init(&barrier, nullptr, num_threads);

	/* For portability, explicitly create threads in a joinable state */
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	vector<ThreadParameters> parameters(num_threads);     //std vector helps with the mem allocation, parameters is a real pointer with ThreadParameters type struct, size: num_threads

	ThreadParameters parameters_template{};
	parameters_template.num_threads = num_threads;

	parameters_template.a_len = a_len;
	parameters_template.b_len = b_len;
	parameters_template.a = a;
	parameters_template.b = b;
	parameters_template.score = score;
	parameters_template.max_score = max_score;

	parameters_template.barrier = &barrier;


	for (int i = 0; i < num_threads; i++) {
		parameters_template.thread_id = i;

		parameters[i] = parameters_template;  // initialization for every para struct for each thread
		assert(pthread_create(&threads[i], &attr, thread_work, (void *) &parameters[i]) == 0);
	}

	for (int i = 0; i < num_threads; i++) {
		pthread_join(threads[i], NULL);
	}

	pthread_barrier_destroy(&barrier);

	free(score);
	free(max_score);

	return glb_max_value;
}