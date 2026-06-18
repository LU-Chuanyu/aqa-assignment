"""
Pairs Trading (Statistical Arbitrage) – AQA Assignment
========================================================
Implements:
  - Cointegration testing (Engle-Granger ADF on spread)
  - Hedge ratio estimation (OLS)
  - Z-score based signal generation
  - Half-life of mean reversion (Ornstein-Uhlenbeck fit)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


# ── 1. Cointegration test ─────────────────────────────────────────────────────

def test_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
    significance: float = 0.05,
) -> dict:
    """
    Engle-Granger two-step cointegration test.

    Steps:
      1. Regress log(A) on log(B) to estimate hedge ratio γ.
      2. Test residuals (spread) for stationarity with ADF.

    Returns
    -------
    dict with keys:
      hedge_ratio  – OLS coefficient γ
      spread       – residual series (A − γ B)
      adf_stat     – ADF test statistic
      p_value      – p-value of ADF test
      is_cointegrated – True if p < significance
    """
    log_a = np.log(price_a)
    log_b = np.log(price_b)

    # OLS regression: log(A) = α + γ log(B) + ε
    X = sm.add_constant(log_b)
    model = sm.OLS(log_a, X).fit()
    hedge_ratio = float(model.params.iloc[1])
    spread = log_a - hedge_ratio * log_b

    adf_result = adfuller(spread.dropna(), autolag="AIC")
    adf_stat = float(adf_result[0])
    p_value = float(adf_result[1])

    return {
        "hedge_ratio": hedge_ratio,
        "spread": spread,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "is_cointegrated": p_value < significance,
    }


# ── 2. Z-score of spread ──────────────────────────────────────────────────────

def spread_zscore(spread: pd.Series, window: int | None = None) -> pd.Series:
    """
    Compute the rolling z-score of the spread.

    If ``window`` is None, use the full-history mean and std (in-sample).
    """
    if window is None:
        mu = spread.mean()
        sigma = spread.std(ddof=1)
    else:
        mu = spread.rolling(window).mean()
        sigma = spread.rolling(window).std(ddof=1)
    return (spread - mu) / sigma


# ── 3. Half-life of mean reversion ───────────────────────────────────────────

def half_life(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion from an Ornstein-Uhlenbeck fit.

    Model: Δspread_t = κ (μ − spread_{t-1}) + ε
    Half-life = −ln(2) / κ  ≈  ln(2) / speed_of_reversion

    A shorter half-life means faster mean-reversion.
    """
    spread_lag = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()

    X = sm.add_constant(spread_lag)
    model = sm.OLS(delta_spread, X).fit()
    kappa = float(model.params.iloc[1])

    if kappa >= 0:
        return float("inf")   # non-mean-reverting

    return float(-np.log(2) / kappa)


# ── 4. Signal generation ──────────────────────────────────────────────────────

def pairs_signals(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """
    Generate long/short signals from z-score of spread.

    Rules:
      - Long spread (long A, short B)  when z < −entry_z
      - Short spread (short A, long B) when z > +entry_z
      - Close position                 when |z| < exit_z

    Returns
    -------
    pd.Series of signals: +1 (long spread), -1 (short spread), 0 (flat)
    """
    signals = pd.Series(0.0, index=zscore.index)
    position = 0.0

    for t in zscore.index:
        z = zscore.loc[t]
        if np.isnan(z):
            signals.loc[t] = 0.0
            continue

        if position == 0:
            if z < -entry_z:
                position = 1.0
            elif z > entry_z:
                position = -1.0
        elif position == 1 and z > -exit_z:
            position = 0.0
        elif position == -1 and z < exit_z:
            position = 0.0

        signals.loc[t] = position

    return signals
