"""
Linear Algebra Review – AQA Assignment
=======================================
Key operations used in quantitative finance:
  - Portfolio return and variance with vectors/matrices
  - PCA for dimensionality reduction / factor extraction
  - Cholesky decomposition for correlated simulation
"""

import numpy as np


# ── 1. Portfolio arithmetic ──────────────────────────────────────────────────

def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """Compute expected portfolio return: r_p = w · r."""
    return float(weights @ expected_returns)


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute portfolio variance: σ²_p = wᵀ Σ w."""
    return float(weights @ cov_matrix @ weights)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute portfolio standard deviation: σ_p = √(wᵀ Σ w)."""
    return float(np.sqrt(portfolio_variance(weights, cov_matrix)))


# ── 2. Covariance matrix from return matrix ──────────────────────────────────

def sample_covariance_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Compute sample covariance matrix from a returns matrix.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        T observations of N asset returns.

    Returns
    -------
    np.ndarray, shape (N, N)
    """
    return np.cov(returns.T, ddof=1)


# ── 3. PCA ────────────────────────────────────────────────────────────────────

def pca(cov_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Principal Component Analysis via eigendecomposition.

    Returns
    -------
    eigenvalues  : np.ndarray, sorted descending
    eigenvectors : np.ndarray, columns are principal components
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """Fraction of total variance explained by each principal component."""
    return eigenvalues / eigenvalues.sum()


# ── 4. Cholesky decomposition for correlated random draws ────────────────────

def correlated_samples(
    cov_matrix: np.ndarray, n_samples: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate correlated random samples using Cholesky decomposition.

    Parameters
    ----------
    cov_matrix : np.ndarray, shape (N, N)
    n_samples  : int
    rng        : optional numpy random Generator for reproducibility

    Returns
    -------
    np.ndarray, shape (n_samples, N) – correlated samples
    """
    if rng is None:
        rng = np.random.default_rng()
    L = np.linalg.cholesky(cov_matrix)
    n_assets = cov_matrix.shape[0]
    z = rng.standard_normal((n_samples, n_assets))
    return z @ L.T


# ── 5. Minimum-variance portfolio weights ────────────────────────────────────

def minimum_variance_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Closed-form minimum-variance portfolio weights (long/short, fully invested).

    w* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)
    """
    ones = np.ones(cov_matrix.shape[0])
    inv_cov = np.linalg.inv(cov_matrix)
    w = inv_cov @ ones
    return w / (ones @ w)
