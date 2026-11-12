import numpy as np
import pandas as pd
import pytest

from backtester import Backtest, BuyAndHold, SmaCrossover, RsiMeanReversion
from backtester.strategies import rsi


def prices(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_buy_and_hold_matches_asset_return_without_costs():
    p = prices(np.linspace(100, 200, 300))
    r = Backtest(p, cost_bps=0).run(BuyAndHold())
    # signal on day 0's close is held from day 1 on, so it earns every move after day 0
    assert r.equity.iloc[-1] / r.equity.iloc[0] == pytest.approx(p.iloc[-1] / p.iloc[0])


def test_no_lookahead_signal_is_traded_next_bar():
    p = prices([100, 100, 100, 150, 150])      # jump on day 3
    class JumpDay(BuyAndHold):
        def positions(self, prices):
            return pd.Series([0, 0, 0, 1, 1], index=prices.index, dtype=float)
    r = Backtest(p, cost_bps=0).run(JumpDay())
    # signal appears on the jump day, so the jump itself must NOT be captured
    assert r.equity.iloc[-1] == pytest.approx(10_000)


def test_costs_reduce_returns():
    p = prices(100 + np.sin(np.arange(400) / 5) * 10)
    s = SmaCrossover(5, 20)
    free = Backtest(p, cost_bps=0).run(s).equity.iloc[-1]
    costly = Backtest(p, cost_bps=25).run(s).equity.iloc[-1]
    assert costly < free


def test_sma_flat_until_slow_window_ready():
    p = prices(np.arange(1, 301))
    pos = SmaCrossover(10, 50).positions(p)
    assert (pos.iloc[:49] == 0).all() and pos.iloc[-1] == 1


def test_sma_rejects_bad_windows():
    with pytest.raises(ValueError):
        SmaCrossover(200, 50)


def test_rsi_bounds():
    p = prices(100 + np.random.default_rng(0).normal(0, 1, 500).cumsum())
    r = rsi(p, 14)
    assert r.between(0, 100).all()


def test_rsi_strategy_only_long_or_flat():
    p = prices(100 + np.random.default_rng(1).normal(0, 1, 800).cumsum() + np.arange(800) * .05)
    pos = RsiMeanReversion().positions(p)
    assert set(pos.unique()) <= {0.0, 1.0}
