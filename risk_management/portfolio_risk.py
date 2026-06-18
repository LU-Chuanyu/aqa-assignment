"""
Portfolio Risk Analytics – AQA Assignment
==========================================
Implements:
  - Portfolio Beta, Tracking Error, Information Ratio
  - Drawdown analysis (max drawdown, average drawdown, recovery time)
  - Rolling volatility and rolling Sharpe ratio
"""

import numpy as np
import pandas as pd


# ── 1. Benchmark-relative metrics ────────────────────────────────────────────

def portfolio_beta(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """
    Compute portfolio beta: β = Cov(r_p, r_b) / Var(r_b).
    """
    cov = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    var_b = np.var(benchmark_returns, ddof=1)
    return float(cov / var_b) if var_b > 0 else float("nan")


def tracking_error(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    trading_days: int = 252,
) -> float:
    """
    Annualised tracking error: std(r_p − r_b) × √252.
    """
    active = portfolio_returns - benchmark_returns
    return float(np.std(active, ddof=1) * np.sqrt(trading_days))


def information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    trading_days: int = 252,
) -> float:
    """
    Annualised Information Ratio: active_return / tracking_error.
    """
    active = portfolio_returns - benchmark_returns
    te = tracking_error(portfolio_returns, benchmark_returns, trading_days)
    if te == 0:
        return float("nan")
    annual_active = float(np.mean(active) * trading_days)
    return float(annual_active / te)


# ── 2. Drawdown analysis ──────────────────────────────────────────────────────

def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Compute the drawdown series from an equity curve.

    drawdown_t = (peak_t − equity_t) / peak_t
    """
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown (positive number representing a loss)."""
    dd = drawdown_series(equity_curve)
    return float(dd.min())


def average_drawdown(equity_curve: pd.Series) -> float:
    """Average drawdown across all periods below the peak."""
    dd = drawdown_series(equity_curve)
    below_peak = dd[dd < 0]
    return float(below_peak.mean()) if len(below_peak) > 0 else 0.0


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """
    Longest number of consecutive periods the equity curve stayed below
    its previous peak (drawdown duration).
    """
    dd = drawdown_series(equity_curve)
    in_drawdown = (dd < 0).astype(int)

    max_dur = 0
    current_dur = 0
    for v in in_drawdown:
        if v == 1:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0

    return max_dur


# ── 3. Rolling analytics ──────────────────────────────────────────────────────

def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    trading_days: int = 252,
) -> pd.Series:
    """
    Annualised rolling volatility.
    """
    return returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> pd.Series:
    """
    Rolling Sharpe ratio (annualised).
    """
    excess = returns - risk_free_rate / trading_days
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std(ddof=1)
    return (roll_mean / roll_std) * np.sqrt(trading_days)
