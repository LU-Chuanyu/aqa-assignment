"""
Time Series Analysis – AQA Assignment
=======================================
Implements:
  - Stationarity testing (ADF)
  - Autocorrelation / PACF helpers
  - ARIMA model fitting and forecasting
  - GARCH(1,1) volatility modelling
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA


# ── 1. Stationarity ───────────────────────────────────────────────────────────

def adf_test(series: pd.Series, max_diff: int = 2) -> dict:
    """
    Run ADF test and auto-difference until stationarity is achieved.

    Returns
    -------
    dict with: n_diffs (number of differences applied), adf_stat, p_value,
               stationary (bool), stationary_series
    """
    s = series.dropna().copy()
    for d in range(max_diff + 1):
        result = adfuller(s, autolag="AIC")
        adf_stat = float(result[0])
        p_value = float(result[1])
        if p_value < 0.05:
            return {
                "n_diffs": d,
                "adf_stat": adf_stat,
                "p_value": p_value,
                "stationary": True,
                "stationary_series": s,
            }
        s = s.diff().dropna()

    return {
        "n_diffs": max_diff,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "stationary": False,
        "stationary_series": s,
    }


# ── 2. Autocorrelation ────────────────────────────────────────────────────────

def compute_acf_pacf(
    series: pd.Series, nlags: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation function (ACF) and partial autocorrelation
    function (PACF) values.

    Returns
    -------
    (acf_values, pacf_values) – each of length nlags+1
    """
    acf_vals = acf(series.dropna(), nlags=nlags, fft=True)
    pacf_vals = pacf(series.dropna(), nlags=nlags)
    return acf_vals, pacf_vals


def ljung_box_test(series: pd.Series, lags: int = 10) -> dict:
    """
    Ljung-Box portmanteau test for autocorrelation.

    H₀: no serial autocorrelation up to ``lags`` lags.

    Returns
    -------
    dict with: lb_statistic, p_value (at the final lag)
    """
    result = sm.stats.acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    return {
        "lb_statistic": float(result["lb_stat"].iloc[-1]),
        "p_value": float(result["lb_pvalue"].iloc[-1]),
    }


# ── 3. ARIMA ──────────────────────────────────────────────────────────────────

def fit_arima(
    series: pd.Series,
    order: tuple[int, int, int] = (1, 0, 1),
) -> dict:
    """
    Fit an ARIMA(p, d, q) model.

    Parameters
    ----------
    series : time series (typically stationary returns or differenced prices)
    order  : (p, d, q) – AR order, differencing order, MA order

    Returns
    -------
    dict with: model_result, aic, bic, residuals
    """
    model = ARIMA(series.dropna(), order=order)
    fitted = model.fit()
    return {
        "model_result": fitted,
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "residuals": fitted.resid,
    }


def arima_forecast(
    model_result,
    steps: int = 5,
) -> pd.Series:
    """
    Produce out-of-sample forecast from a fitted ARIMA model.

    Returns
    -------
    pd.Series of forecasted values
    """
    forecast = model_result.forecast(steps=steps)
    return forecast


# ── 4. GARCH(1,1) ─────────────────────────────────────────────────────────────

def fit_garch(returns: pd.Series) -> dict:
    """
    Fit a GARCH(1,1) model to capture volatility clustering.

    Model: σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}

    Requires the ``arch`` package (install with: pip install arch).

    Returns
    -------
    dict with: omega, alpha, beta, conditional_volatility
    """
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError(
            "The 'arch' package is required for GARCH modelling. "
            "Install it with: pip install arch"
        ) from exc

    am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")

    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    cond_vol = res.conditional_volatility / 100   # rescale back

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "conditional_volatility": cond_vol,
        "model_result": res,
    }
