"""Strategies turn a price series into a target position in [-1, 1] for each day.

A position decided using day t's close is traded at the next bar by the engine,
so strategies never need to worry about look-ahead bias themselves.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


class Strategy:
    name = "strategy"

    def positions(self, prices: pd.Series) -> pd.Series:  # pragma: no cover
        raise NotImplementedError


@dataclass
class BuyAndHold(Strategy):
    name: str = "Buy & Hold"

    def positions(self, prices):
        return pd.Series(1.0, index=prices.index)


@dataclass
class SmaCrossover(Strategy):
    """Long when the fast moving average is above the slow one, else flat (or short)."""
    fast: int = 50
    slow: int = 200
    allow_short: bool = False

    def __post_init__(self):
        if self.fast >= self.slow:
            raise ValueError("fast window must be shorter than slow window")
        self.name = f"SMA {self.fast}/{self.slow}"

    def positions(self, prices):
        fast = prices.rolling(self.fast).mean()
        slow = prices.rolling(self.slow).mean()
        pos = np.where(fast > slow, 1.0, -1.0 if self.allow_short else 0.0)
        pos = pd.Series(pos, index=prices.index)
        pos[slow.isna()] = 0.0
        return pos


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


@dataclass
class RsiMeanReversion(Strategy):
    """Buy when RSI is oversold, exit when it recovers; only trade above the trend filter."""
    window: int = 2
    entry: float = 10
    exit: float = 70
    trend: int = 200

    def __post_init__(self):
        self.name = f"RSI({self.window}) mean reversion"

    def positions(self, prices):
        r = rsi(prices, self.window)
        uptrend = prices > prices.rolling(self.trend).mean()
        pos = np.zeros(len(prices))
        holding = False
        for i, (ri, up) in enumerate(zip(r.to_numpy(), uptrend.to_numpy())):
            if not holding and up and ri < self.entry:
                holding = True
            elif holding and (ri > self.exit or not up):
                holding = False
            pos[i] = 1.0 if holding else 0.0
        return pd.Series(pos, index=prices.index)
