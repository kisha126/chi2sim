# chi2sim

Chi-square test with Monte Carlo simulation for contingency tables.
The package and its documentation are still under construction.

## Description

The `chi2sim` package is a Python implementation that contains the translation of R's H tests for contingency tables when `simulate.p.value = TRUE`, originally written in C and based on Patefield's (1981) FORTRAN algorithm. The package provides a fast and reliable method to compute p-values for chi-square tests using Monte Carlo simulation.

## Installation

```bash
pip install chi2sim
```

## Usage

```python
import numpy as np
from chi2sim import chi2_cont_sim

# Example contingency table
table = np.array([
    [10, 5],
    [20, 15]
], dtype=int)

# Perform chi-square test with Monte Carlo simulation
result = chi2_cont_sim(table)
print(result)
```

## Features

- Similar to SciPy's `scipy.stats.chi2_contingency`, but returns 
- Monte Carlo simulation for p-value approximation
- Easy-to-use Python interface

## Requirements

- Python >= 3.9
- NumPy >= 1.15.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Citation

If you use the `chi2sim` package in your study, please don't forget to cite the following literatures:

-  Hope, A. C. A. (1968). A simplified Monte Carlo significance test procedure. Journal of the Royal Statistical Society Series B, 30, 582–598. doi:10.1111/j.2517-6161.1968.tb00759.x.

-  Patefield, W. M. (1981). Algorithm AS 159: An efficient method of generating r x c tables with given row and column totals. Applied Statistics, 30, 91–97. doi:10.2307/2346669.
