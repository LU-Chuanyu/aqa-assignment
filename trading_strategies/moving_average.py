"""
Moving Average Crossover Strategy – AQA Assignment
====================================================
Implements:
  - Simple Moving Average (SMA) crossover
  - Exponential Moving Average (EMA) crossover
Both return position signals compatible with the Backtester.
"""

import pandas as pd
from trading_strategies.backtester import BaseStrategy


class SMACrossover(BaseStrategy):
    """
    Simple Moving Average crossover strategy.

    Signal logic:
      - Long  (+1) when fast SMA > slow SMA
      - Short (-1) when fast SMA < slow SMA  [if ``allow_short=True``]
      - Flat  ( 0) otherwise
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
        allow_short: bool = False,
    ) -> None:
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.allow_short = allow_short

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        prices : pd.DataFrame of close prices (dates × assets)

        Returns
        -------
        pd.DataFrame of signals (+1, -1, 0) with same index/columns as prices
        """
        fast_ma = prices.rolling(self.fast_window).mean()
        slow_ma = prices.rolling(self.slow_window).mean()

        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        signals[fast_ma > slow_ma] = 1
        if self.allow_short:
            signals[fast_ma < slow_ma] = -1

        return signals


class EMACrossover(BaseStrategy):
    """
    Exponential Moving Average crossover strategy.

    Signal logic:
      - Long  (+1) when fast EMA > slow EMA
      - Short (-1) when fast EMA < slow EMA  [if ``allow_short=True``]
    """

    def __init__(
        self,
        fast_span: int = 12,
        slow_span: int = 26,
        allow_short: bool = False,
    ) -> None:
        if fast_span >= slow_span:
            raise ValueError("fast_span must be less than slow_span")
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.allow_short = allow_short

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        prices : pd.DataFrame of close prices (dates × assets)

        Returns
        -------
        pd.DataFrame of signals (+1, -1, 0) with same index/columns as prices
        """
        fast_ema = prices.ewm(span=self.fast_span, adjust=False).mean()
        slow_ema = prices.ewm(span=self.slow_span, adjust=False).mean()

        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        signals[fast_ema > slow_ema] = 1
        if self.allow_short:
            signals[fast_ema < slow_ema] = -1

        return signals
