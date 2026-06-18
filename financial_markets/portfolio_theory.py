"""
Portfolio Theory – AQA Assignment
==================================
Implements:
  - Efficient frontier via Monte Carlo simulation
  - Minimum-variance and maximum-Sharpe portfolios
  - CAPM beta and alpha estimation
"""

import numpy as np
import scipy.optimize as opt


# ── 1. Portfolio metrics ──────────────────────────────────────────────────────

def portfolio_performance(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> tuple[float, float, float]:
    """
    Compute annualised return, volatility, and Sharpe ratio for a portfolio.

    Returns
    -------
    (annual_return, annual_volatility, sharpe_ratio)
    """
    ret = float(weights @ expected_returns) * trading_days
    vol = float(np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(trading_days))
    sr = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return ret, vol, sr


# ── 2. Efficient frontier via Monte Carlo ────────────────────────────────────

def efficient_frontier_monte_carlo(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Simulate random portfolios to approximate the efficient frontier.

    Returns
    -------
    dict with keys:
      weights      – (n_portfolios, N) array of portfolio weights
      returns      – annualised returns
      volatilities – annualised volatilities
      sharpes      – Sharpe ratios
    """
    if rng is None:
        rng = np.random.default_rng()

    n_assets = len(expected_returns)
    all_weights = np.zeros((n_portfolios, n_assets))
    all_returns = np.zeros(n_portfolios)
    all_vols = np.zeros(n_portfolios)
    all_sharpes = np.zeros(n_portfolios)

    for i in range(n_portfolios):
        w = rng.random(n_assets)
        w /= w.sum()
        all_weights[i] = w
        all_returns[i], all_vols[i], all_sharpes[i] = portfolio_performance(
            w, expected_returns, cov_matrix, risk_free_rate, trading_days
        )

    return {
        "weights": all_weights,
        "returns": all_returns,
        "volatilities": all_vols,
        "sharpes": all_sharpes,
    }


# ── 3. Optimised portfolios ───────────────────────────────────────────────────

def _neg_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    trading_days: int,
) -> float:
    _, _, sr = portfolio_performance(
        weights, expected_returns, cov_matrix, risk_free_rate, trading_days
    )
    return -sr


def max_sharpe_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> np.ndarray:
    """
    Find portfolio weights that maximise the Sharpe ratio.

    Constraints: weights sum to 1; all weights ≥ 0 (long-only).
    """
    n = len(expected_returns)
    w0 = np.ones(n) / n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 1)] * n
    result = opt.minimize(
        _neg_sharpe,
        w0,
        args=(expected_returns, cov_matrix, risk_free_rate, trading_days),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def minimum_variance_portfolio(
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Find portfolio weights that minimise variance.

    Constraints: weights sum to 1; all weights ≥ 0 (long-only).
    """
    n = cov_matrix.shape[0]
    w0 = np.ones(n) / n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 1)] * n

    def portfolio_var(w: np.ndarray) -> float:
        return float(w @ cov_matrix @ w)

    result = opt.minimize(
        portfolio_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


# ── 4. CAPM ───────────────────────────────────────────────────────────────────

def capm_beta_alpha(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> tuple[float, float]:
    """
    Estimate CAPM beta and alpha via OLS regression.

    Model: r_i − r_f = α + β (r_m − r_f) + ε

    Returns
    -------
    (beta, alpha)
    """
    excess_asset = asset_returns - risk_free_rate
    excess_market = market_returns - risk_free_rate

    # OLS: β = Cov(r_i, r_m) / Var(r_m)
    beta = float(np.cov(excess_asset, excess_market)[0, 1] / np.var(excess_market, ddof=1))
    alpha = float(np.mean(excess_asset) - beta * np.mean(excess_market))
    return beta, alpha


def capm_expected_return(
    beta: float,
    risk_free_rate: float,
    market_expected_return: float,
) -> float:
    """Security Market Line: E[r_i] = r_f + β (E[r_m] − r_f)."""
    return risk_free_rate + beta * (market_expected_return - risk_free_rate)
