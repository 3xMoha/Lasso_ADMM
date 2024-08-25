# Custom ADMM Lasso Implementation

Custom implementation of the Lasso regression using the Alternating Direction Method of Multipliers (ADMM) and results contrast with the Lasso implementation in scikit-learn.

## Environment Setup

To run the code in this repository you can set up an environment using either `pip` or `conda`.

### With pip

1. **Create a virtual environment**:
   ```bash
   python -m venv myenv
   ```
2. **Activate the environment**:
  ```bash
  source myenv/bin/activate
  ```
3. **Install the packages**:
   ```bash
   pip install -r requirements.txt
   ```
### With CONDA

1. **Create an environment with the packages**
   ```bash
   conda env create -f conda_lasso_env.yaml
   ```
2. **Activate the environment**:
  ```bash
  conda activate lasso_env
  ```

## Execution

```bash
python lasso_admm.py
```
This script automatically calls the lasso_admm function in test mode.

## Expected output

For the synthetic data, created from a standard normal distribution with a given set of parameters it is presented:

- Real synthetic coefficients from where we create our target variable
- Number of iterations till convergence
- The coefficients estimated using the constructed ADMM
- The coefficients estimated using Scikit-learn Lasso

```
Test 1: Synthetic Data
Real synthetic coefficients:
[ 1.5 -2.   0.   0.   0.   0.   0.   0.   3.   0. ]
Converged in 387 iterations.
Estimated coefficients using custom ADMM:
[ 1.563 -2.021 -0.059 -0.044  0.016 -0.042  0.    -0.     3.02   0.063]

Estimated coefficients using scikit-learn Lasso:
[ 1.57  -2.026 -0.065 -0.052  0.022 -0.049 -0.    -0.005  3.022  0.069]

Difference in coefficients:
[0.007 0.005 0.005 0.008 0.006 0.007 0.    0.004 0.002 0.006]
```

From the scikit-lear dataset, we provide the same as before with the exception of the  real coefficient since it is not possible.

```
Test 2: Scikit-Learn Diabetes Dataset
Converged in 13 iterations.

Estimated coefficients using custom ADMM on Diabetes dataset:
[ -0.476 -11.406  24.727  15.429 -37.643  22.648   4.789   8.416  35.721
   3.217]

Estimated coefficients using scikit-learn Lasso on Diabetes dataset:
[ -0.476 -11.407  24.727  15.429 -37.662  22.662   4.797   8.419  35.728
   3.217]

Difference in coefficients on Sk-L Diabetes dataset:
[0.    0.    0.    0.    0.018 0.014 0.009 0.003 0.007 0.   ]
```
## Description of the main function

```python
def lasso_admm(X, y, lambd, rho, num_iter, tol):
    """
    Lasso regression using ADMM.
    
    Args:
    - X: numpy array of shape (n_samples, n_features), data matrix.
    - y: numpy array of shape (n_samples,), target variable.
    - lambd: float, regularization parameter (lambda).
    - rho: float, Lagrangian parameter.
    - num_iter: int, maximum number of iterations.
    - tol: float, tolerance for convergence.
    
    Returns:
    - beta: numpy array of shape (n_features,), the estimated coefficients.
    """
```



