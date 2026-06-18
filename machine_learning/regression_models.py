"""
Regression Models for Finance – AQA Assignment
================================================
Implements:
  - OLS factor model (Fama-French style)
  - Ridge and Lasso regression for return prediction
  - Cross-validation framework
  - Feature engineering helpers
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# ── 1. Feature engineering ────────────────────────────────────────────────────

def make_features(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Create standard momentum, reversal, and volatility features.

    Parameters
    ----------
    prices  : pd.DataFrame of close prices
    returns : pd.DataFrame of daily returns (same columns as prices)

    Returns
    -------
    pd.DataFrame of features (same index as returns)
    """
    features = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        r = returns[col]

        # Short-term reversal (1-day)
        features[f"{col}_reversal_1d"] = r.shift(1)

        # Momentum (1-month ≈ 21 trading days)
        features[f"{col}_mom_21d"] = r.shift(1).rolling(21).sum()

        # Momentum (3-month ≈ 63 trading days)
        features[f"{col}_mom_63d"] = r.shift(1).rolling(63).sum()

        # Rolling volatility (21-day)
        features[f"{col}_vol_21d"] = r.shift(1).rolling(21).std(ddof=1)

        # RSI (14-day)
        features[f"{col}_rsi_14"] = _rsi(r.shift(1), window=14)

    return features


def _rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    gain = returns.clip(lower=0).rolling(window).mean()
    loss = (-returns.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ── 2. OLS factor model ───────────────────────────────────────────────────────

def ols_factor_model(
    asset_returns: pd.Series,
    factors: pd.DataFrame,
) -> dict:
    """
    Regress asset returns on factors (e.g., Fama-French factors).

    Returns
    -------
    dict with: alpha, betas (dict), r_squared, residuals
    """
    y = asset_returns.values
    X = factors.values
    X_const = np.column_stack([np.ones(len(X)), X])

    model = LinearRegression(fit_intercept=False)
    model.fit(X_const, y)

    coefficients = model.coef_
    alpha = float(coefficients[0])
    betas = {col: float(coefficients[i + 1]) for i, col in enumerate(factors.columns)}
    residuals = y - model.predict(X_const)
    r_sq = float(r2_score(y, model.predict(X_const)))

    return {
        "alpha": alpha,
        "betas": betas,
        "r_squared": r_sq,
        "residuals": residuals,
    }


# ── 3. Ridge / Lasso with time-series cross-validation ───────────────────────

def regularised_model_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "ridge",
    alpha_values: list[float] | None = None,
    n_splits: int = 5,
) -> dict:
    """
    Fit a Ridge or Lasso model with time-series cross-validation to select alpha.

    Parameters
    ----------
    X          : feature matrix (dates × features)
    y          : target returns
    model_type : "ridge" or "lasso"
    alpha_values: regularisation strengths to search over
    n_splits   : number of time-series CV folds

    Returns
    -------
    dict with: best_alpha, best_r2, model (fitted on full data)
    """
    if alpha_values is None:
        alpha_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    best_alpha = alpha_values[0]
    best_score = -np.inf

    X_arr = X.fillna(0).values
    y_arr = y.values

    for a in alpha_values:
        fold_scores = []
        for train_idx, test_idx in tscv.split(X_arr):
            X_train = scaler.fit_transform(X_arr[train_idx])
            X_test = scaler.transform(X_arr[test_idx])
            y_train = y_arr[train_idx]
            y_test = y_arr[test_idx]

            if model_type == "ridge":
                m = Ridge(alpha=a)
            else:
                m = Lasso(alpha=a, max_iter=5000)

            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            fold_scores.append(r2_score(y_test, pred))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = a

    # Fit final model on full dataset
    X_scaled = scaler.fit_transform(X_arr)
    if model_type == "ridge":
        final_model = Ridge(alpha=best_alpha)
    else:
        final_model = Lasso(alpha=best_alpha, max_iter=5000)
    final_model.fit(X_scaled, y_arr)

    return {
        "best_alpha": best_alpha,
        "best_r2": best_score,
        "model": final_model,
        "scaler": scaler,
    }
