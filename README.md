# chi2sim

Chi-square test with Monte Carlo simulation for contingency tables.
This is still under construction.

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

- Fast C implementation of contingency table generation
- Monte Carlo simulation for p-value approximation
- Easy-to-use Python interface

## Requirements

- Python >= 3.9
- NumPy >= 1.15.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Citation

If you use the `chi2sim` package in your work, please cite the following:

Hope, A. C. A. (1968). A simplified Monte Carlo significance test procedure. Journal of the Royal Statistical Society Series B, 30, 582–598. doi:10.1111/j.2517-6161.1968.tb00759.x.


