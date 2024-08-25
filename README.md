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
