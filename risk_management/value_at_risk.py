"""
Value at Risk (VaR) – AQA Assignment
======================================
Implements three standard VaR methods:
  1. Historical Simulation
  2. Parametric (Variance-Covariance) assuming normality
  3. Monte Carlo Simulation

Also computes Conditional VaR (CVaR / Expected Shortfall).
"""

import numpy as np


# ── 1. Historical VaR ─────────────────────────────────────────────────────────

def var_historical(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Historical simulation VaR.

    VaR = −percentile(returns, (1 − confidence) × 100)

    Parameters
    ----------
    returns    : 1-D array of historical daily returns
    confidence : confidence level, e.g. 0.95 for 95% VaR

    Returns
    -------
    VaR as a positive number (loss convention)
    """
    return float(-np.percentile(returns, (1 - confidence) * 100))


def cvar_historical(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Historical Conditional VaR (Expected Shortfall).

    CVaR = −mean(returns | returns ≤ −VaR)
    """
    var = var_historical(returns, confidence)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


# ── 2. Parametric VaR ────────────────────────────────────────────────────────

def var_parametric(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Parametric (Gaussian) VaR.

    VaR = −(μ − z_α σ)   where z_α = Φ⁻¹(1 − confidence)

    Note: assumes returns are normally distributed.
    """
    import scipy.stats as stats

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z = stats.norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def cvar_parametric(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Parametric CVaR (Expected Shortfall under normality).

    CVaR = −μ + σ × φ(z_α) / (1 − confidence)
    """
    import scipy.stats as stats

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z = stats.norm.ppf(1 - confidence)
    # Expected value of the left tail of a standard normal
    es_std = -stats.norm.pdf(z) / (1 - confidence)
    return float(-(mu + sigma * es_std))


# ── 3. Monte Carlo VaR ────────────────────────────────────────────────────────

def var_monte_carlo(
    returns: np.ndarray,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Monte Carlo VaR: simulate future returns by sampling from a fitted
    normal distribution and take the (1 − confidence) quantile.

    Parameters
    ----------
    returns      : historical return series (used to estimate μ, σ)
    confidence   : confidence level
    n_simulations: number of simulated scenarios

    Returns
    -------
    VaR as a positive number
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    simulated = rng.normal(loc=mu, scale=sigma, size=n_simulations)
    return float(-np.percentile(simulated, (1 - confidence) * 100))


def cvar_monte_carlo(
    returns: np.ndarray,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Monte Carlo CVaR (Expected Shortfall).
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    simulated = rng.normal(loc=mu, scale=sigma, size=n_simulations)
    var = -np.percentile(simulated, (1 - confidence) * 100)
    tail = simulated[simulated <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


# ── 4. Scaling VaR ────────────────────────────────────────────────────────────

def scale_var(daily_var: float, holding_period: int) -> float:
    """
    Scale 1-day VaR to a multi-day holding period.

    Uses the square-root-of-time rule (assumes i.i.d. returns):
      VaR_T = VaR_1 × √T
    """
    return float(daily_var * np.sqrt(holding_period))
