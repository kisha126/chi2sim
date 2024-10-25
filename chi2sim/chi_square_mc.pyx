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

def chi2_cont_sim(np.ndarray[int, ndim=2] table not None, int simulations=10000):
    """
    Perform Chi-square test using Monte Carlo simulation for contingency tables.

    Parameters:
    -----------
    table : numpy.ndarray
        2D contingency table of observed frequencies
    simulations : int, optional
        Number of Monte Carlo simulations (default: 10000)

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
        # Call the C function
        p_value = monte_carlo_pvalue(c_table, nrow, ncol, simulations)
    finally:
        # Clean up
        for i in range(nrow):
            free(c_table[i])
        free(c_table)

    return {
        'p_value': p_value,
        'simulations': simulations
    }
