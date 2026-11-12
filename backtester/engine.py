"""Vectorized backtest engine.

Rules (kept deliberately simple and explicit):
- Signals are computed on close of day t and filled at close of day t+1 (no look-ahead).
- Each change in position pays ``cost_bps`` basis points on the traded amount.
"""
from dataclasses import dataclass, field
import pandas as pd

from .metrics import summarize, drawdown
from .strategies import Strategy


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    stats: dict = field(default_factory=dict)

    @property
    def drawdown(self) -> pd.Series:
        return drawdown(self.equity)


@dataclass
class Backtest:
    prices: pd.Series
    cost_bps: float = 5.0
    initial_capital: float = 10_000.0

    def run(self, strategy: Strategy) -> BacktestResult:
        target = strategy.positions(self.prices).clip(-1, 1)
        held = target.shift(1).fillna(0.0)          # trade on the next bar
        asset_ret = self.prices.pct_change().fillna(0.0)
        turnover = held.diff().abs().fillna(held.abs())
        strat_ret = held * asset_ret - turnover * self.cost_bps / 10_000
        equity = self.initial_capital * (1 + strat_ret).cumprod()
        return BacktestResult(strategy.name, equity, strat_ret, held,
                              summarize(equity, strat_ret, held))

    def compare(self, *strategies: Strategy) -> pd.DataFrame:
        results = [self.run(s) for s in strategies]
        return pd.DataFrame({r.name: r.stats for r in results}).T, results
