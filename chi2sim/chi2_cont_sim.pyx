# cython wrapper for the C implementation
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

# Declare the C functions from our header file
cdef extern from "src/chi_square_mc.h":
    # Declare the C functions we want to use
    double* compute_fact(int n)
    int** rcont(int* nrowt, int* ncolt, double* fact, int nrow, int ncol)
    double chi_square_stat(int** observed, double** expected, int nrow, int ncol)
    double monte_carlo_pvalue(int** observed, int nrow, int ncol, int simulations)

def chi2_cont_sim(np.ndarray[int, ndim=2] table not None, int n_sim=10000):
    """
    Perform Chi-square test using Monte Carlo simulation for contingency tables.

    Parameters:
    -----------
    table : numpy.ndarray
        2D contingency table of observed frequencies
    n_sim : int, optional
        Number of Monte Carlo simulations (default: 10000)

    Returns:
    --------
    dict
        Dictionary containing Chi-square statistic, p-value, and other test statistics
    """
    if table.ndim != 2:
        raise ValueError("Table must be 2-dimensional")

    cdef int nrow = table.shape[0]
    cdef int ncol = table.shape[1]

    # Convert numpy array to C array
    cdef int** c_table = <int**>malloc(nrow * sizeof(int*))
    if not c_table:
        raise MemoryError("Failed to allocate memory for table")

    for i in range(nrow):
        c_table[i] = <int*>malloc(ncol * sizeof(int))
        if not c_table[i]:
            # Clean up already allocated memory
            for j in range(i):
                free(c_table[j])
            free(c_table)
            raise MemoryError("Failed to allocate memory for table row")

        for j in range(ncol):
            c_table[i][j] = table[i, j]

    try:
        # Step 1: Compute Chi-square statistic
        # We first need the expected values. Using rcont to get expected table
        # (as rcont generates tables based on given row/column sums).
        # Generate the expected table
        row_sums = np.sum(table, axis=1)
        col_sums = np.sum(table, axis=0)
        total = np.sum(row_sums)

        # Convert row_sums and col_sums to C arrays
        cdef int* c_row_sums = <int*>malloc(nrow * sizeof(int))
        cdef int* c_col_sums = <int*>malloc(ncol * sizeof(int))
        cdef double** expected_table = <double**>malloc(nrow * sizeof(double*))

        if not c_row_sums or not c_col_sums or not expected_table:
            raise MemoryError("Failed to allocate memory for expected table")

        for i in range(nrow):
            c_row_sums[i] = row_sums[i]
            expected_table[i] = <double*>malloc(ncol * sizeof(double))
            if not expected_table[i]:
                for j in range(i):
                    free(expected_table[j])
                free(expected_table)
                raise MemoryError("Failed to allocate memory for expected table rows")

        for j in range(ncol):
            c_col_sums[j] = col_sums[j]

        # Calculate expected frequencies: (row_sum[i] * col_sum[j]) / total
        for i in range(nrow):
            for j in range(ncol):
                expected_table[i][j] = (c_row_sums[i] * c_col_sums[j]) / total

        # Step 2: Compute the Chi-square statistic
        chi_square = chi_square_stat(c_table, expected_table, nrow, ncol)

        # Step 3: Perform Monte Carlo simulation for p-value
        p_value = monte_carlo_pvalue(c_table, nrow, ncol, n_sim)
        
    finally:
        # Clean up C memory allocations
        for i in range(nrow):
            free(c_table[i])
            free(expected_table[i])
        free(c_table)
        free(expected_table)
        free(c_row_sums)
        free(c_col_sums)

    # Return the Chi-square statistic, p-value, and number of simulations
    return {
        'statistic': chi_square,
        'p_value': p_value,
        'n_sim': n_sim
    }
    