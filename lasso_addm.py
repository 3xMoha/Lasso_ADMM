import numpy as np
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

def soft_thresholding(x, threshold):
    """
    Soft-thresholding operator.
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def lasso_admm(X, y, lambd, rho, num_iter=1000, tol=1e-6):
    """
    Lasso using ADMM.
    
    Args:
    - X: numpy array of shape (n_samples, n_features), data matrix.
    - y: numpy array of shape (n_samples,), target var.
    - lambd: regularization parameter (lambda).
    - rho: Lagrangian parameter.
    - num_iter: maximum number of iter.
    - tol: tolerance for convergence.
    
    Returns:
    - beta: numpy array of shape (n_features,), the estimated coefficients.
    """
    # Initialization
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    z = np.zeros(n_features)
    u = np.zeros(n_features)

    # Precompute (X^T X + rho I)^{-1} 
    XTy = X.T @ y
    XT_X = X.T @ X
    inv_matrix = np.linalg.inv(XT_X + rho * np.eye(n_features))

    for iteration in range(num_iter):
        
        # Update beta
        beta = inv_matrix @ (XTy + rho * z - u)

        # Update z 
        z_old = z.copy()
        z = soft_thresholding(beta + u, lambd / rho)

        # Update dual variable u
        u += rho*(beta - z)

        # Check convergence
        if np.linalg.norm(z - z_old) < tol:
            print(f'Converged in {iteration + 1} iterations.')
            break

    return beta

# Testing agains Scikit-learn's Lasso ADDM
if __name__ == "__main__":
    # Set NumPy print options for better readability
    np.set_printoptions(precision=3, suppress=True)

    # First test with synthetic data
    print("Test 1: Synthetic Data")
    np.random.seed(98545)
    X = np.random.randn(100, 10)
    true_beta = np.array([1.5, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3, 0.0])
    y = X @ true_beta + 0.5 * np.random.randn(100)

    # Parameters
    lambd = 1.0
    rho = 1.0

    print("Real synthetic coefficients:")
    print(true_beta)

    # Fit model using custom ADMM implementation
    estimated_beta_admm = lasso_admm(X, y, lambd, rho)
    print("Estimated coefficients using custom ADMM:")
    print(estimated_beta_admm)

    # Fit model using scikit-learn's Lasso
    alpha = lambd / (2 * len(y))
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000, tol=1e-6)
    lasso.fit(X, y)
    estimated_beta_sklearn = lasso.coef_
    print("\nEstimated coefficients using scikit-learn Lasso:")
    print(estimated_beta_sklearn)

    # Compare results
    print("\nDifference in coefficients:")
    print(np.abs(estimated_beta_admm - estimated_beta_sklearn))

    # Second test with real data: Diabetes dataset
    print("\nTest 2: Scikit-Learn Diabetes Dataset")
    # Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Parameters for Lasso
    lambd = 0.1
    rho = 1.0

    # Fit model using custom ADMM implementation
    estimated_beta_admm = lasso_admm(X, y, lambd, rho)
    print("\nEstimated coefficients using custom ADMM on Diabetes dataset:")
    print(estimated_beta_admm)

    # Fit model using scikit-learn's Lasso
    alpha = lambd / (2 * len(y))
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000, tol=1e-6)
    lasso.fit(X, y)
    estimated_beta_sklearn = lasso.coef_
    print("\nEstimated coefficients using scikit-learn Lasso on Diabetes dataset:")
    print(estimated_beta_sklearn)

    # Compare results
    print("\nDifference in coefficients on Sk-L Diabetes dataset:")
    print(np.abs(estimated_beta_admm - estimated_beta_sklearn))
