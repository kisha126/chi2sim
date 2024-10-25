#ifndef CHI_SQUARE_MC_H
#define CHI_SQUARE_MC_H

#include <stdlib.h>
#include <math.h>
#include <string.h>

// Function declarations
double* compute_fact(int n);
int** rcont(int* nrowt, int* ncolt, double* fact, int nrow, int ncol);
double chi_square_stat(int** observed, double** expected, int nrow, int ncol);
double monte_carlo_pvalue(int** observed, int nrow, int ncol, int simulations);

#endif
