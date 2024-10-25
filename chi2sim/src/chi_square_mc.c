#include "chi_square_mc.h"

// Helper function to generate uniform random number between 0 and 1
static double runif() {
    return (double)rand() / RAND_MAX;
}

// Compute factorial lookup table
double* compute_fact(int n) {
    double* fact = (double*)malloc((n + 1) * sizeof(double));
    fact[0] = 0.0;
    for (int i = 1; i <= n; i++) {
        fact[i] = fact[i-1] + log(i);
    }
    return fact;
}

// Generate random contingency table
int** rcont(int* nrowt, int* ncolt, double* fact, int nrow, int ncol) {
    // Allocate matrix
    int** matrix = (int**)malloc(nrow * sizeof(int*));
    for (int i = 0; i < nrow; i++) {
        matrix[i] = (int*)calloc(ncol, sizeof(int));
    }

    int* jwork = (int*)malloc((ncol - 1) * sizeof(int));

    int ntotal = 0;
    for (int i = 0; i < nrow; i++) ntotal += nrowt[i];

    int nr_1 = nrow - 1;
    int nc_1 = ncol - 1;
    int ib = 0;

    // Copy column totals
    for (int j = 0; j < nc_1; j++) {
        jwork[j] = ncolt[j];
    }

    int jc = ntotal;

    // Main algorithm
    for (int l = 0; l < nr_1; l++) {
        int ia = nrowt[l];
        int ic = jc;
        jc -= ia;

        for (int m = 0; m < nc_1; m++) {
            int id = jwork[m];
            int ie = ic;
            ib = ie - ia;
            int ii = ib - id;
            ic -= id;

            if (ie == 0) {
                for (int j = m; j < nc_1; j++) {
                    matrix[l][j] = 0;
                }
                ia = 0;
                break;
            }

            // Generate random entry
            double U = runif();
            int nlm;
            do {
                nlm = (int)(ia * (id / (double)ie) + 0.5);
                double x = exp(fact[ia] + fact[ib] + fact[ic] + fact[id]
                                   - fact[ie] - fact[nlm]
                                   - fact[id - nlm] - fact[ia - nlm] - fact[ii + nlm]);

                                   if (x >= U) break;
                                   if (x == 0.0) {
                                       free(jwork);
                                       for (int i = 0; i < nrow; i++) free(matrix[i]);
                                       free(matrix);
                                       return NULL; // Algorithm failure
                                   }

                                   double sumprb = x;
                                   double y = x;
                                   int nll = nlm;
                                   int lsp;

                                   do {
                                       double j = (id - nlm) * (double)(ia - nlm);
                                       lsp = (nlm == ia || nlm == id);

                                       if (!lsp) {
                                           nlm++;
                                           x *= j / ((double)nlm * (ii + nlm));
                                           sumprb += x;
                                           if (sumprb >= U) goto L160;
                                       }

                                       int lsm;
                                       do {
                                           j = nll * (double)(ii + nll);
                                           lsm = (nll == 0);

                                           if (!lsm) {
                                               nll--;
                                               y *= j / ((double)(id - nll) * (ia - nll));
                                               sumprb += y;
                                               if (sumprb >= U) {
                                                   nlm = nll;
                                                   goto L160;
                                               }
                                               if (!lsp) break;
                                           }
                                       } while (!lsm);

                                   } while (!lsp);

                                   U = sumprb * runif();

            } while (1);

            L160:
                matrix[l][m] = nlm;
            ia -= nlm;
            jwork[m] -= nlm;
        }
        matrix[l][nc_1] = ia;
    }

    // Fill in last row
    for (int m = 0; m < nc_1; m++) {
        matrix[nr_1][m] = jwork[m];
    }
    matrix[nr_1][nc_1] = ib - matrix[nr_1][nc_1 - 1];

    free(jwork);
    return matrix;
}

// Compute chi-square statistic
double chi_square_stat(int** observed, double** expected, int nrow, int ncol) {
    double chi_sq = 0.0;
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            double diff = observed[i][j] - expected[i][j];
            chi_sq += (diff * diff) / expected[i][j];
        }
    }
    return chi_sq;
}

// Monte Carlo simulation for p-value
double monte_carlo_pvalue(int** observed, int nrow, int ncol, int simulations) {
    // Get row and column totals
    int* row_totals = (int*)malloc(nrow * sizeof(int));
    int* col_totals = (int*)malloc(ncol * sizeof(int));
    int total = 0;

    for (int i = 0; i < nrow; i++) {
        row_totals[i] = 0;
        for (int j = 0; j < ncol; j++) {
            row_totals[i] += observed[i][j];
        }
        total += row_totals[i];
    }

    for (int j = 0; j < ncol; j++) {
        col_totals[j] = 0;
        for (int i = 0; i < nrow; i++) {
            col_totals[j] += observed[i][j];
        }
    }

    // Compute expected frequencies
    double** expected = (double**)malloc(nrow * sizeof(double*));
    for (int i = 0; i < nrow; i++) {
        expected[i] = (double*)malloc(ncol * sizeof(double));
        for (int j = 0; j < ncol; j++) {
            expected[i][j] = (row_totals[i] * col_totals[j]) / (double)total;
        }
    }

    // Compute observed chi-square statistic
    double observed_chi_sq = chi_square_stat(observed, expected, nrow, ncol);

    // Compute factorial lookup table
    double* fact = compute_fact(total);

    // Run simulations
    int count = 0;
    for (int sim = 0; sim < simulations; sim++) {
        int** random_table = rcont(row_totals, col_totals, fact, nrow, ncol);
        if (random_table == NULL) continue;

        double sim_chi_sq = chi_square_stat(random_table, expected, nrow, ncol);
        if (sim_chi_sq >= observed_chi_sq) count++;

        // Free random table
        for (int i = 0; i < nrow; i++) free(random_table[i]);
        free(random_table);
    }

    // Cleanup
    free(fact);
    free(row_totals);
    free(col_totals);
    for (int i = 0; i < nrow; i++) free(expected[i]);
    free(expected);

    return (double)count / simulations;
}
