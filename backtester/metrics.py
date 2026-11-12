import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std()
    return 0.0 if sd == 0 else np.sqrt(TRADING_DAYS) * excess.mean() / sd


def sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0].std()
    return 0.0 if not downside else np.sqrt(TRADING_DAYS) * returns.mean() / downside


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def summarize(equity: pd.Series, returns: pd.Series, positions: pd.Series) -> dict:
    trades = int((positions.diff().abs() > 0).sum())
    return {
        "CAGR": cagr(equity),
        "Volatility": returns.std() * np.sqrt(TRADING_DAYS),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "Max drawdown": drawdown(equity).min(),
        "Time in market": float((positions != 0).mean()),
        "Trades": trades,
    }
