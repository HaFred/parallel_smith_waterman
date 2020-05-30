#include "mpi_smith_waterman.h"

/* Author: There is another way of doing the Smith_waterman algorithm concurrently in a `divide_and_conquer` fashion. But the output max_score will vary in that case, which is not as expected, refer to (https://www.docin.com/p-1202801738.html). In the following implementation, we do it as hinted in the class tutorial. */

/* Improvement could be rewrite the find_score function, try using anti-diagnoal-wise, but it takes time to verify it. In this case, we just need to Bcast the previous_ant, not the whole score matrix.*/

/* If we have to print out the whole score matrix, it takes more time. */

/*
 *  You can add helper functions and variables as you wish.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <cmath>
#include <vector>
#include <climits>
#include <cstring>

using std::vector;
using std::string;


int find_score(int i, int j, int **loc_score, char *a, char *b){ //row i, col j
	int intersect_score = -1;
	i = i + 1;
	j = j + 1;
	intersect_score = max(0,
		  max(loc_score[i - 1][j - 1] + sub_mat(a[i - 1], b[j - 1]), 
		  max(loc_score[i - 1][j] - GAP,
			  loc_score[i][j - 1] - GAP)));
		
	return intersect_score;
}


int smith_waterman(int my_rank, int p, MPI_Comm comm, char *a, char *b, int a_len, int b_len) {
    /*
     *  Please fill in your codes here.
     */
	
	// Bcast the a&b and their lengths which were copied in processor#0, might not needed?
		
    // main loop
    int max_score = 0;
	// int ant_max_score[a_len + b_len - 1] = {0}; // the max for the current anti-diagnoal
	
	cout << "testing first" << "\t"<<endl;
	
	// int ant_max_score[a_len + b_len - 1];
	int *ant_max_score;
	ant_max_score = (int *) malloc(sizeof(int) * (a_len + b_len - 1));
	int loc_max_score = 0; // for a certain anti-diagnoal, it is allocated for each process
			
	// complete the score matrix below
	// num_node_dia = (int *) calloc(a_len + b_len - 1, sizeof(int));
	int num_node_dia;
	int row;
	int col;
	int *recv_cnts;
	int *recv_disp;
	bool a_b_flag;
	recv_cnts = (int *) calloc(p, sizeof(int));
	recv_disp = (int *) calloc(p, sizeof(int));	
	int temp_a_len;
	char temp_a_str[a_len];
	
	a_b_flag = a_len < b_len ? true : false; // a shorter than b flag	

	// init score matrix
    int **score = new int*[a_len + 1];
	// allocate local memory
	// score = (int *) malloc(sizeof(int) * a_len *b_len);	
    for (int i = 0; i <= a_len; i++) {
        score[i] = new int[b_len + 1];
        for (int j = 0; j <= b_len; j++) {
            score[i][j] = 0;
        }
    }
	
	// start the anti-diagnoal parallel part
	for (int ant = 2; ant <= a_len + b_len - 1; ant++) { //ant is anti-diagnoal rank	
		if (ant < b_len) {
			num_node_dia = ant;
		} else if (ant > a_len) {
			num_node_dia = a_len + b_len - ant;
		} else {
			num_node_dia = b_len;
		}
		
		// set up the receive cnts & disp for each process 
		int offset = 0;
		int *loc_piece_dia;
		for (int i = 0; i < p; ++i) {
			if (i != p - 1) {
				recv_cnts[i] = ceil(float(num_node_dia) / float(p)); //avoid the last one being the bottleneck, the ceil() seems better
				recv_disp[i] = offset;
			} else {
				recv_cnts[i] = num_node_dia - ceil(float(num_node_dia) / float(p)) * i; //avoid the last one being the bottleneck
				recv_disp[i] = offset;				
			}
			offset += recv_cnts[i];
			if (my_rank == i) // set up the mem for loc_piece_dia
				loc_piece_dia = (int *) malloc(sizeof(int) * recv_cnts[i]);
		}		
		
		// start to formulate the score matrix and max_score
		for (int i = 0; i <= num_node_dia - 1; i++) {  // might change
		
			if ((i < recv_disp[my_rank]) && (i >= recv_disp[my_rank - 1])) {
				if (ant < b_len) {
					row = i;
					col = ant - i + 1;
				} else {
					row = i + ant - b_len;
					col = b_len - i + 1;
				}
			}	
			// loc_score = find_score(row, col, score, a, b);
			loc_piece_dia[recv_cnts[i]] = find_score(row, col, score, a, b);
			loc_max_score = max(loc_max_score, loc_piece_dia[recv_cnts[i]]);
			score[row][col] = loc_max_score;
			//MPI_Gatherv(loc_piece_dia, 1, MPI_INT, score, recv_cnts, recv_disp, MPI_INT, 0, comm);	
			MPI_Bcast(score, a_len * b_len, MPI_INT, my_rank, comm);
		}
		
		// debug
		cout << "testing" << "\t"<<endl;
		
		MPI_Allreduce(&loc_max_score, &ant_max_score + ant - 1, 1, MPI_INT, MPI_MAX, comm);	
		max_score = max(max(ant_max_score[ant], ant_max_score[ant-1]), 0); // every process will do it and the return value of the whole sw is still the max, no worries to make it processor#0

			
		// Coz it is multi-process, the score matrix has to be contructed with MPI_Gather. And don't Bcast the element of score matrix element-wise, it comsumes. Better do it anti-diagnoal-wise.	


	} 
		 
	// todo: free memory
	free(ant_max_score);
	free(recv_cnts);
	free(recv_disp);

    return max_score;
}

