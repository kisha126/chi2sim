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
        Number of Monte Carlo n_sim (default: 10000)

    Returns:
    --------
    dict
        Dictionary containing p-value and other test statistics
    """
    if table.ndim != 2:
        raise ValueError("Table must be 2-dimensional")

    cdef int nrow = table.shape[0]
    cdef int ncol = table.shape[1]

    # Convert numpy array to C array
    cdef int** c_table = <int**>malloc(nrow * sizeof(int*))
    cdef double** expected_table = <double**>malloc(nrow * sizeof(double*))  # Allocate expected table
    if not c_table or not expected_table:
        raise MemoryError("Failed to allocate memory for table or expected table")

    cdef int i, j
    cdef double row_sum, col_sum, total_sum

    # Calculate row sums, column sums, and total sum
    cdef double* row_totals = <double*>malloc(nrow * sizeof(double))
    cdef double* col_totals = <double*>malloc(ncol * sizeof(double))
    if not row_totals or not col_totals:
        raise MemoryError("Failed to allocate memory for row or column totals")

    total_sum = 0
    for i in range(nrow):
        row_totals[i] = 0
        for j in range(ncol):
            row_totals[i] += table[i, j]
        total_sum += row_totals[i]

    for j in range(ncol):
        col_totals[j] = 0
        for i in range(nrow):
            col_totals[j] += table[i, j]

    # Populate c_table and calculate expected_table
    for i in range(nrow):
        c_table[i] = <int*>malloc(ncol * sizeof(int))
        expected_table[i] = <double*>malloc(ncol * sizeof(double))  # Allocate expected table row
        if not c_table[i] or not expected_table[i]:
            # Clean up already allocated memory
            for k in range(i):
                free(c_table[k])
                free(expected_table[k])
            free(c_table)
            free(expected_table)
            raise MemoryError("Failed to allocate memory for table or expected table row")

        for j in range(ncol):
            c_table[i][j] = table[i, j]
            expected_table[i][j] = (row_totals[i] * col_totals[j]) / total_sum  # Calculate expected value

    try:
        # Call the C function
        p_value = monte_carlo_pvalue(c_table, nrow, ncol, n_sim)
        chi2_stat = chi_square_stat(c_table, expected_table, nrow, ncol)
    finally:
        # Clean up
        for i in range(nrow):
            free(c_table[i])
            free(expected_table[i])
        free(c_table)
        free(expected_table)
        free(row_totals)
        free(col_totals)

    return {
        'statistic' : chi2_stat,
        'p_value': p_value,
        'n_sim': n_sim
    }
    