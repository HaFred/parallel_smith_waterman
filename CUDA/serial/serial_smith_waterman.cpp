#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <stdio.h>

#include "smith_waterman.h"

using namespace std;

int print_h_score (int *h_score, int b_len, int a_len) {
	std::ofstream outputf("output_serial.txt", std::ofstream::out);
	// outputf << dist[0];
	// printf("\n");
	outputf << "\n";
	for (int i = 0; i < a_len + 1; i++) {
		// cout << a[i - 1] << "\t";
		for (int j = 0; j < b_len + 1; j++) {
					// printf("%d\t", d_score[dev_idx(i, j, a_len + 1)]);
			outputf << h_score[i * (a_len + 1) + j] << "\t";
		}
		// printf("\n");
		outputf << "\n";
	}	
	outputf << endl;
	return 0;
}

int smith_waterman(char *a, char *b, int a_len, int b_len) {
    // init score matrix
//    int **score = new int*[a_len + 1];
//    for (int i = 0; i <= a_len; i++) {
//        score[i] = new int[b_len + 1];
//        for (int j = 0; j <= b_len; j++) {
//            score[i][j] = 0;
//        }
//    }

    int *score = (int *)malloc(sizeof(int) * (a_len + 1) * (b_len + 1));
    for(int i = 0; i <= a_len; i++){
        for(int j = 0; j <= b_len; j++){
            score[idx(i, j, b_len + 1)] = 0;
        }
    }

    // main loop
    int max_score = 0;

    for (int i = 1; i <= a_len; i++) {
//        for (int j = 1; j <= b_len; j++) {
//            score[i][j] = max(0,
//                          max(score[i - 1][j - 1] + sub_mat(a[i - 1], b[j - 1]),
//                          max(score[i - 1][j] - GAP,
//                              score[i][j - 1] - GAP)));
//            max_score = max(max_score, score[i][j]);

        for (int j = 1; j <= b_len; j++) {
            score[idx(i, j, b_len + 1)] = max(0,
                              max(score[idx(i - 1, j - 1, b_len + 1)] + sub_mat(a[i - 1], b[j - 1]),
                                      max(score[idx(i - 1, j, b_len + 1)] - GAP,
                                      score[idx(i, j - 1, b_len + 1)] - GAP)));
            max_score = max(max_score, score[idx(i, j, b_len + 1)]);
        }
    }
	
	// debug
	int num_diagonal = b_len + a_len + 1; // original diagonals, aka the re_row length. 0 padding, a_len + 1 & b_len + 1
	print_h_score(score, b_len, a_len);


//    for (int i = 0; i <= a_len; i++) {
//        delete [] score[i];
//    }
//    delete [] score;

    

    free(score);

    return max_score;
}