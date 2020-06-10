/**
 *
 * Author: Please comment the DEBUG macro. My testing shows that it has slightly better performance than mpi version.
*/

#include <pthread.h>
#include "pthreads_smith_waterman.h"
#include <semaphore.h>  /* Semaphores are not part of Pthreads */
#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>


// #define DEBUG
/*
 *  You can add helper functions and variables as you wish.
 */
int counter;
sem_t* thread_sem;
sem_t* barrier_sem;
sem_t count_sem; 
int *temp_score; // array storing exchanging value of every dia
int *temp_max; // array storing 
 
/* In addition to semaphore, maybe barriers could also make it */

int find_global_maximum(int *global_max, int p){
    int max = 0;
    for (int i = 0; i < p; i++){
        if(global_max[i] > max){
            max = global_max[i];
        }
    }
    return max;	
} 


//  pthreads_smith_waterman declaration, thread fucntion
void* pthreads_smith_waterman(void* t_parameter);

// the struct we need to pass to each thread.
struct thread_parameter
{
	int num_threads;
	int n_diagonal;
	char *a;
	char *b;
	int a_len;
	int b_len;
    int *global_max;  // have to pass in the addr, so that the value can be modified
	// int **score;
	int rank;
};

void *pthreads_smith_waterman(void *t_parameter) {
	struct thread_parameter *paramater = (struct thread_parameter *) t_parameter;
	int num_threads = paramater->num_threads;
	int my_rank = paramater->rank;
	int n_diagonal = paramater->n_diagonal;
	char *a = paramater->a;
	char *b = paramater->b;
	int a_len = paramater->a_len;
	int b_len = paramater->b_len;
	int *global_max = paramater->global_max;
	// int **score = paramater->score;	 // it's wrong if use this, will make it critical section, and while using the `score` it just f**ks up... 
	// fredmay4: actually, as long as we make the score matrix globally [alen*blen], rather then below [alen*width], it could be shared by every thread and won't be written on the same element by 2 different thread. Refer to the sol
	// fred: but I think the sol is unnecessary, since we don't need score matrix at all. We just wanna print out the max. allocate the mem in thread_func is better
	int max_score = 0; 
	int dest = (my_rank + 1) % num_threads;
	int block_l = my_rank * b_len / num_threads, block_r = (my_rank + 1) * b_len / num_threads;
    int width = block_r - block_l;  // with the rounding down of cpp, it works perfectly ito process assignments, no matter a_len>or<b_len
	
	
	// declare & init score matrix for each thread
	// score: only store the score computed by current thread `my_rank` and one adjacent value from the previous thread `my_rank - 1`
    int **score = (int **)malloc(sizeof(int*) * (a_len + 1));
    for (int i = 0; i <= a_len; i++) {
        score[i] = (int*)calloc(width + 1, sizeof(int));
    }
	
	for (int i = 0; i < n_diagonal; i++) { 	
		int row = i - my_rank + 1;
		if (row > 0 && row <= a_len) {   // limit those out-of-bound
			for (int j = 1; j <= width; j++) { // the score[][0] is reserved 4 prevrank
				score[row][j] = max(0,
								max(score[row - 1][j - 1] + sub_mat(a[row - 1], b[block_l + j - 1]), 
								max(score[row - 1][j] - GAP,
									score[row][j - 1] - GAP)));
				max_score = max(max_score, score[row][j]);
			}

			if (my_rank < num_threads - 1) {
				temp_score[dest] = score[row][width];
				sem_post(&thread_sem[dest]);  /* "Unlock" the semaphore of dest */
				// cout<<"the rank"<< my_rank << " sent out its temp"<<endl;
			}
		}
		if (my_rank > 0 && row + 1 > 0 && row + 1 <= a_len) {
			sem_wait(&thread_sem[my_rank]);  /* Wait for our semaphore to be unlocked */
			score[row+1][0] = temp_score[my_rank];
	#	ifdef DEBUG			
		cout<<"the rank"<< my_rank <<" received "<<temp_score[my_rank]<<" for dia"<<i<<endl;
	# 	endif
		
			}
			
		// semaphore barrier in each diagonal, need every rank reach here
		sem_wait(&count_sem);
		if (counter == num_threads - 1) {
			 counter = 0;
			 sem_post(&count_sem);
			 for (int j = 0; j < num_threads - 1; j++)
				sem_post(&barrier_sem[i]);
	#	ifdef DEBUG	
	std::cout<<"Sync done for dia"<<i<<"\n"<<std::endl;
	# 	endif
		
		} else {
			 counter++;
			 sem_post(&count_sem);
			 sem_wait(&barrier_sem[i]);    // where barrier works
		}
	}
	
	// max for each thread
	temp_max[my_rank] = max_score;
		
	*global_max = find_global_maximum(temp_max, num_threads);
		
    for (int i = 0; i <= a_len; i++) {
        free(score[i]);
    }
	free(score);
	
}
int smith_waterman(int num_threads, char *a, char *b, int a_len, int b_len){
    /*
     *  Please fill in your codes here.
     */
    pthread_t *thread_handles = (pthread_t*) calloc(num_threads, sizeof(pthread_t));
    struct thread_parameter **paramater = (struct thread_parameter**) calloc(num_threads, sizeof(struct thread_parameter*));
	int thread; // rank
	int code;
	int n_diagonal = a_len + num_threads - 1;    // num of block dia
	thread_sem = (sem_t*) malloc(num_threads*sizeof(sem_t)); // it is array, have to explictly allocate	
	barrier_sem = (sem_t*) malloc(n_diagonal*sizeof(sem_t));
	int *global_max;
	global_max = (int*) malloc(sizeof(int));
	temp_score =(int*) calloc(num_threads, sizeof(int));
	temp_max =(int*) calloc(num_threads, sizeof(int));
	
	for (thread = 0; thread < num_threads; thread++) {
	  // temp_score[thread] = NULL;
	  sem_init(&thread_sem[thread], 0, 0);
   }
	for (int i = 0; i < n_diagonal; i++)
      sem_init(&barrier_sem[i], 0, 0);
	sem_init(&count_sem, 0, 1);
	for (thread = 0; thread < num_threads; thread++){
		paramater[thread] = (struct thread_parameter*) malloc (sizeof(struct thread_parameter));
		paramater[thread]->num_threads = num_threads;
		paramater[thread]->n_diagonal = n_diagonal;
		paramater[thread]->rank = thread;
		paramater[thread]->a = a;
		paramater[thread]->b = b;
		paramater[thread]->a_len = a_len;
		paramater[thread]->b_len = b_len;
		paramater[thread]->global_max = global_max;
		code = pthread_create(&thread_handles[thread], (pthread_attr_t*) NULL, pthreads_smith_waterman, (void*) paramater[thread]);
	#	ifdef DEBUG	
		if (code){
		std::cout <<"ERROR; return code from pthread_create() is %d\n"<<code <<std::endl;
		exit(-1);
		}	  
	# 	endif
	}


	for (thread = 0; thread < num_threads; thread++) {
	  pthread_join(thread_handles[thread], NULL);
	  free(paramater[thread]);
	}
	for (int i = 0; i < n_diagonal; i++)
      sem_destroy(&barrier_sem[i]);
	sem_destroy(&count_sem);  
	#	ifdef DEBUG	
	std::cout << "\nThread joined, start free\n"<<std::endl;
	# 	endif
	free(paramater);
	free(temp_score);
	free(temp_max);
	for (thread = 0; thread < num_threads; thread++) {
	  sem_destroy(&thread_sem[thread]);
	}
	free(thread_sem);  
	free(thread_handles);
    // free(score);	
	
	#	ifdef DEBUG	
	std::cout << "All destroyed and free\n"<<std::endl;
	# 	endif	
	 	 
	return *global_max;
}

