# chi2sim

Chi-square test with Monte Carlo simulation for contingency tables.

## Installation

```bash
pip install chi2sim
```

## Usage

```python
import numpy as np
from chi2sim import chi_square_test

# Example contingency table
table = np.array([
    [10, 5],
    [20, 15]
], dtype=int)

# Perform chi-square test with Monte Carlo simulation
result = chi_square_test(table, simulations=10000)
print(f"P-value: {result['p_value']}")
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
