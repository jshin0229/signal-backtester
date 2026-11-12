"""signal-backtester: a small, readable event-free (vectorized) backtesting engine."""
from .data import load_prices
from .engine import Backtest, BacktestResult
from .strategies import SmaCrossover, RsiMeanReversion, BuyAndHold
from .metrics import summarize

__all__ = ["load_prices", "Backtest", "BacktestResult", "SmaCrossover",
           "RsiMeanReversion", "BuyAndHold", "summarize"]
