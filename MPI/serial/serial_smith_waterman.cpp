#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <cassert>

#include "smith_waterman.h"

#define DEBUG 0


using namespace std;
// using namespace std::chrono;

int smith_waterman(char *a, char *b, int a_len, int b_len) {
	
	// auto t_start = chrono::high_resolution_clock::now();
	
	// where these two are equivelent
	// int* ip = new int; // dynamincally allocate the address
	// int* ip = (int *) malloc(sizeof(int));
	
    // declare & init score matrix
    int **score = new int*[a_len + 1];
    for (int i = 0; i <= a_len; i++) {
        score[i] = new int[b_len + 1];
        for (int j = 0; j <= b_len; j++) {
            score[i][j] = 0;
        }
    }

#ifdef DEBUG
    cout << "\t";
    for (int i = 0; i < b_len; i++) {
        cout << b[i] << "\t";
    }
    cout << endl;
#endif

    // main loop
    int max_score = 0;
    for (int i = 1; i <= a_len; i++) {
#ifdef DEBUG
        cout << a[i - 1] << "\t";
#endif
        for (int j = 1; j <= b_len; j++) {
            score[i][j] = max(0,
                          max(score[i - 1][j - 1] + sub_mat(a[i - 1], b[j - 1]), 
                          max(score[i - 1][j] - GAP,
                              score[i][j - 1] - GAP)));
            max_score = max(max_score, score[i][j]);

#ifdef DEBUG
            cout << score[i][j] << "\t";
#endif
        }
#ifdef DEBUG
        cout << endl;
#endif
		

    }

/* 	auto t_end = chrono::high_resolution_clock::now();
#ifdef T_DEBUG 

	chrono::duration<double> diff = t_end - t_start;
	cerr << "Time: " << diff.count() << " s" << endl;
		
#endif */

    for (int i = 0; i <= a_len; i++) {
        delete [] score[i];
    }
    delete [] score;

    return max_score;
}
