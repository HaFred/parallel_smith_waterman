#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>

#include "mpi_smith_waterman.h"

int smith_waterman(int my_rank, int p, MPI_Comm comm, char *a, char *b, int a_len, int b_len) {
    MPI_Bcast(&a_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&b_len, 1, MPI_INT, 0, comm);
    if (my_rank > 0) {
        a = (char *)malloc(sizeof(char) * (a_len + 1));
        b = (char *)malloc(sizeof(char) * (b_len + 1));
    }
    MPI_Bcast(a, a_len + 1, MPI_CHAR, 0, comm);
    MPI_Bcast(b, b_len + 1, MPI_CHAR, 0, comm);

    // block boundary: [l, r)
	
	// fred: this block method is brilliant, it divide the whole matrix as p part. It is easier to locate a certain score element in score[][] comparing to my original separated idea.
	// the separated idea: 1. hard to locate, needs a lot of conditional statement.
	// 2. increase the MPI communication API overhead, too frequent
	
    int block_l = my_rank * b_len / p, block_r = (my_rank + 1) * b_len / p;
    int width = block_r - block_l;  // fred: with the rounding down of cpp, it works perfectly ito process assignments, no matter a_len>or<b_len

    int **score = (int **)malloc(sizeof(int*) * (a_len + 1));
    for (int i = 0; i <= a_len; i++) {
        score[i] = (int*)calloc(width + 1, sizeof(int));
    }

    int n_diagonal = a_len + p - 1;
    int max_score = 0;
    for (int i = 0; i < n_diagonal; i++) { 
        int row = i - my_rank + 1;
        if (row > 0 && row <= a_len) {   // limit those out-of-bound
            for (int j = 1; j <= width; j++) {
                score[row][j] = max(0,
                                max(score[row - 1][j - 1] + sub_mat(a[row - 1], b[block_l + j - 1]),
                                max(score[row - 1][j] - GAP,
                                    score[row][j - 1] - GAP)));
                max_score = max(max_score, score[row][j]);
            }
            if (my_rank < p - 1) {
                MPI_Send(&score[row][width], 1, MPI_INT, my_rank + 1, 0, comm);
            }
        }
        if (my_rank > 0 && row + 1 > 0 && row + 1 <= a_len) {
            MPI_Recv(&score[row + 1][0], 1, MPI_INT, my_rank - 1, 0, comm, MPI_STATUS_IGNORE); // you must receive under the same i, every my_rank got only previous rank shared score, which is already enough 
        }
    }

    int global_max = 0;
    MPI_Reduce(&max_score, &global_max, 1, MPI_INT, MPI_MAX, 0, comm);
    
    if (my_rank > 0) {
        free(a);
        free(b);
    }
    for (int i = 0; i <= a_len; i++) {
        free(score[i]);
    }
    free(score);
    
    return global_max;
}
