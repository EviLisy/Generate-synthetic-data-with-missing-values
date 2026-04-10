# mar_missing

Generate synthetic datasets with Missing At Random (MAR) missing values.

## Overview

`mar_missing` provides a flexible tool to introduce structured missingness patterns into complete datasets. It uses logistic regression models to generate missing-at-random (MAR) patterns that depend on observed variables, enabling realistic evaluation of imputation methods.

## Installation

```bash
pip install .
```

Or from source:

```bash
git clone <repository-url>
cd Generate-synthetic-data-with-missing-values
pip install .
```

## Quick Start

```python
import pandas as pd
import numpy as np
from mar_missing import MAR

# Create a complete dataset
df = pd.DataFrame({
    'x1': np.random.normal(size=100),
    'x2': np.random.normal(size=100),
    'x3': np.random.normal(size=100),
    'x4': np.random.normal(size=100),
})

# Case 1: Introduce 30% missingness in x1, x2 (determined by x3, x4)
mar = MAR(
    dataset=df,
    missing_rate=0.3,
    target_vars=['x1', 'x2'],      # Variables to introduce missingness into
    deter_vars=['x3', 'x4'],        # Variables that drive missingness
    weights=[0.5, -0.3],             # Logistic model coefficients
    seed=42
)
df_missing = mar.apply(get_statistics=False)

# Case 2: Use provided probability model
def my_model(X):
    return 1.0 / (1.0 + np.exp(-X.sum(axis=1)))

mar = MAR(
    dataset=df,
    missing_rate=0.25,
    target_vars=['x1', 'x2'],
    deter_vars=['x3', 'x4'],
    model=my_model,
    seed=42
)
df_missing = mar.apply(get_statistics=True)

# Case 3: Random splits with different missingness mechanisms
mar = MAR(
    dataset=df,
    missing_rate=0.2,
    d_deter=2,           # Number of determining variables
    split=3,             # Split data into 3 groups
    weights_range=(-1.0, 1.0),
    seed=42
)
df_missing = mar.apply(get_statistics=False)
```

## API Reference

### `MAR(dataset, missing_rate, ...)`

#### Parameters

- **dataset** : `pd.DataFrame`
  - Complete dataset (must not contain missing values)
  
- **missing_rate** : `float` in [0, 1]
  - Overall proportion of missing values in the output
  
- **same_miss_prob** : `bool`, default=True
  - If True, all target variables share the same missing probability
  - If False, missingness is randomly distributed among target variables
  
- **target_vars** : `str` or `list of str`, optional
  - Variables to introduce missingness into
  - If provided with `d_deter`, raises error
  
- **deter_vars** : `str` or `list of str`, optional
  - Variables that drive the missingness mechanism
  
- **d_deter** : `int`, optional
  - Number of determining variables to randomly select
  - Cannot be used with explicit `target_vars` or `deter_vars`
  
- **split** : `int`, optional
  - If provided, split dataset into `split` groups with independent missingness mechanisms
  - Requires `d_deter` to be specified
  
- **model** : callable, optional
  - Function computing P(missing | deter_vars)
  - Output shape: (n_samples,) or (n_samples, n_target_vars)
  
- **weights** : `list` or `np.ndarray`, optional
  - Coefficients for logistic model (used if `model` is None)
  
- **weights_range** : `tuple of float`, optional
  - (min, max) range for randomly sampling weights
  
- **seed** : `int`, optional
  - Random seed for reproducibility

#### Methods

- **apply(get_statistics=True)** → `pd.DataFrame`
  - Applies missingness pattern and returns dataset with NaN values
  - If `get_statistics=True`, prints missingness summary

## Reproducibility

### Current Behavior

`mar_missing` uses NumPy's global random state. **Reproducibility is guaranteed only when call sequences are controlled:**

```python
# Reproducible
mar1 = MAR(dataset=df, ..., seed=123)
out1 = mar1.apply()
mar2 = MAR(dataset=df, ..., seed=123)
out2 = mar2.apply()
assert out1.isna().equals(out2.isna())  # True

# Not guaranteed to be reproducible
mar1 = MAR(dataset=df, ..., seed=123)
mar2 = MAR(dataset=df, ..., seed=123)  # RNG state has advanced
out1 = mar1.apply()
out2 = mar2.apply()
# Results may differ due to global RNG state
```

### Future Plan

Adding more missingness pattern class like MNAR and MCAR to this repository.

## Testing

Run the test suite:

```bash
pytest tests/ -q
```

### Test Coverage

- **test_mar_validation.py** — Input validation and error handling
- **test_mar_split_behavior.py** — Split semantics and variable partitioning
- **test_mar_reproducibility.py** — Reproducibility with controlled call sequences

## Examples

See [notebooks/](notebooks/) for additional examples:
- [Class for generating missing data.ipynb](notebooks/Class%20for%20generating%20missing%20data.ipynb) — Original research implementation
- [generate data under missingness assumption.ipynb](notebooks/generate%20data%20under%20missingness%20assumption.ipynb) — Data generation workflows

## License

See LICENSE file.

## Citation

If you use `mar_missing` in research, please cite the original work:

```bibtex
@misc{mar_missing,
  title={Generate Synthetic Data with Missing At Random Values},
  author={...},
  year={2024},
  url={...}
}
```

## Contact

For issues, questions, or contributions, please open an issue on GitHub.
