"""
Probability & Statistics Review – AQA Assignment
=================================================
Core statistical tools for quantitative finance:
  - Descriptive statistics and return analysis
  - Hypothesis tests (t-test, normality tests)
  - Confidence intervals
  - Log vs. simple returns
"""

import numpy as np
import scipy.stats as stats


# ── 1. Return calculations ────────────────────────────────────────────────────

def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple (arithmetic) returns: r_t = (P_t − P_{t-1}) / P_{t-1}."""
    return np.diff(prices) / prices[:-1]


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log (continuously compounded) returns: r_t = ln(P_t / P_{t-1})."""
    return np.diff(np.log(prices))


def annualize_return(daily_return_mean: float, trading_days: int = 252) -> float:
    """Annualize a mean daily return: (1 + r_daily)^252 − 1."""
    return (1 + daily_return_mean) ** trading_days - 1


def annualize_volatility(daily_vol: float, trading_days: int = 252) -> float:
    """Annualize daily volatility: σ_annual = σ_daily × √252."""
    return daily_vol * np.sqrt(trading_days)


# ── 2. Descriptive statistics ─────────────────────────────────────────────────

def descriptive_stats(returns: np.ndarray) -> dict:
    """
    Compute key descriptive statistics for a return series.

    Returns
    -------
    dict with keys: mean, std, skewness, kurtosis, min, max, median
    """
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns, ddof=1)),
        "skewness": float(stats.skew(returns)),
        "kurtosis": float(stats.kurtosis(returns)),   # excess kurtosis
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "median": float(np.median(returns)),
    }


# ── 3. Sharpe ratio ───────────────────────────────────────────────────────────

def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Annualised Sharpe ratio.

    SR = (mean(r) − r_f) / std(r)  × √252
    """
    excess = returns - risk_free_rate / trading_days
    if np.std(excess, ddof=1) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(trading_days))


# ── 4. Hypothesis tests ───────────────────────────────────────────────────────

def ttest_mean_zero(returns: np.ndarray) -> tuple[float, float]:
    """
    One-sample t-test: H₀: mean return = 0.

    Returns
    -------
    (t_statistic, p_value)
    """
    result = stats.ttest_1samp(returns, popmean=0.0)
    return float(result.statistic), float(result.pvalue)


def normality_test_jarque_bera(returns: np.ndarray) -> tuple[float, float]:
    """
    Jarque-Bera test for normality.

    H₀: returns are normally distributed.

    Returns
    -------
    (jb_statistic, p_value)
    """
    result = stats.jarque_bera(returns)
    return float(result.statistic), float(result.pvalue)


def adf_test_stationary(series: np.ndarray) -> tuple[float, float]:
    """
    Augmented Dickey-Fuller test for stationarity.

    H₀: series has a unit root (non-stationary).
    Small p-value → reject H₀ → series is stationary.

    Returns
    -------
    (adf_statistic, p_value)
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series, autolag="AIC")
    return float(result[0]), float(result[1])


# ── 5. Confidence intervals ───────────────────────────────────────────────────

def mean_confidence_interval(
    returns: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Confidence interval for the mean return.

    Returns
    -------
    (lower_bound, upper_bound)
    """
    n = len(returns)
    mean = np.mean(returns)
    se = stats.sem(returns)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(mean - t_crit * se), float(mean + t_crit * se)


# ── 6. Correlation and covariance ────────────────────────────────────────────

def rolling_correlation(
    series_a: np.ndarray, series_b: np.ndarray, window: int
) -> np.ndarray:
    """
    Compute rolling Pearson correlation between two return series.

    Returns an array of the same length as the inputs (NaN-padded for the
    first ``window − 1`` observations).
    """
    s_a = pd.Series(series_a)
    s_b = pd.Series(series_b)
    return s_a.rolling(window).corr(s_b).to_numpy()
