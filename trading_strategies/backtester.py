"""
Backtesting Framework – AQA Assignment
=======================================
A simple vectorised backtesting engine.
Usage:
  1. Subclass BaseStrategy and implement generate_signals().
  2. Pass it to Backtester.run() with a price DataFrame.
  3. Inspect BacktestResult for performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── 1. Result dataclass ───────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Container for all backtesting outputs."""

    equity_curve: pd.Series
    returns: pd.Series
    signals: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metrics = compute_performance_metrics(self.returns)


# ── 2. Performance metrics ────────────────────────────────────────────────────

def compute_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict:
    """Compute standard performance metrics from a daily return series."""
    r = returns.dropna()

    if len(r) == 0:
        return {}

    annual_return = float((1 + r.mean()) ** trading_days - 1)
    annual_vol = float(r.std(ddof=1) * np.sqrt(trading_days))
    sharpe = (
        (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
    )

    # Sortino ratio (downside deviation)
    downside = r[r < 0]
    downside_std = float(downside.std(ddof=1) * np.sqrt(trading_days)) if len(downside) > 1 else float("nan")
    sortino = (
        (annual_return - risk_free_rate) / downside_std
        if downside_std and downside_std > 0
        else float("nan")
    )

    # Max drawdown
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    calmar = (
        annual_return / abs(max_drawdown) if max_drawdown != 0 else float("nan")
    )

    # Trade-level stats
    sign_changes = (np.sign(r) != np.sign(r.shift(1))).sum()
    hit_rate = float((r > 0).mean())

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "hit_rate": hit_rate,
        "n_observations": len(r),
    }


# ── 3. Base strategy ──────────────────────────────────────────────────────────

class BaseStrategy:
    """
    Abstract base class for all trading strategies.

    Subclasses must implement ``generate_signals``.
    """

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate position signals from price data.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV or close prices, indexed by date.

        Returns
        -------
        pd.DataFrame with same index as prices.
            Columns correspond to assets; values are +1 (long), -1 (short), or 0.
        """
        raise NotImplementedError


# ── 4. Backtester ─────────────────────────────────────────────────────────────

class Backtester:
    """
    Vectorised backtester.

    Parameters
    ----------
    transaction_cost : float
        One-way cost as a fraction of trade value (e.g. 0.001 = 10 bps).
    initial_capital : float
        Starting portfolio value.
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

    def run(
        self,
        strategy: BaseStrategy,
        prices: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run the strategy on the given price data.

        Parameters
        ----------
        strategy : BaseStrategy
        prices   : pd.DataFrame of close prices (dates × assets)

        Returns
        -------
        BacktestResult
        """
        signals = strategy.generate_signals(prices)

        # Align signals and compute asset returns
        asset_returns = prices.pct_change()

        # Lag signals by 1 day to avoid look-ahead bias
        positions = signals.shift(1).fillna(0)

        # Compute transaction costs: cost on every position change
        turnover = positions.diff().abs()
        cost = (turnover * self.transaction_cost).sum(axis=1)

        # Portfolio return = weighted sum of asset returns minus costs
        portfolio_returns = (positions * asset_returns).sum(axis=1) - cost

        # Build equity curve
        equity = self.initial_capital * (1 + portfolio_returns).cumprod()

        return BacktestResult(
            equity_curve=equity,
            returns=portfolio_returns,
            signals=signals,
        )
