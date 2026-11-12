"""Command line entry point:  python -m backtester SPY --start 2005-01-01"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import load_prices, Backtest, SmaCrossover, RsiMeanReversion, BuyAndHold


def fmt(df):
    out = df.copy()
    for c in ["CAGR", "Volatility", "Max drawdown", "Time in market"]:
        out[c] = out[c].map("{:.1%}".format)
    for c in ["Sharpe", "Sortino"]:
        out[c] = out[c].map("{:.2f}".format)
    out["Trades"] = out["Trades"].astype(int)
    return out


def plot(results, ticker, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        ax1.plot(r.equity, label=r.name, lw=1.4)
        ax2.plot(r.drawdown, lw=1)
    ax1.set_yscale("log"); ax1.set_ylabel("Equity ($, log scale)")
    ax1.set_title(f"{ticker}: strategy equity curves (5 bps costs, next-bar fills)")
    ax1.legend(frameon=False); ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for a in (ax1, ax2):
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest simple strategies on one ticker.")
    p.add_argument("ticker", nargs="?", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--plot", default=None, help="save equity chart to this path")
    a = p.parse_args(argv)

    prices = load_prices(a.ticker, a.start, a.end)
    bt = Backtest(prices, cost_bps=a.cost_bps)
    table, results = bt.compare(BuyAndHold(), SmaCrossover(50, 200), RsiMeanReversion())
    print(f"\n{a.ticker.upper()}  {prices.index[0].date()} -> {prices.index[-1].date()}\n")
    print(fmt(table).to_string())
    if a.plot:
        Path(a.plot).parent.mkdir(parents=True, exist_ok=True)
        plot(results, a.ticker.upper(), a.plot)
        print(f"\nSaved chart to {a.plot}")
    return table


if __name__ == "__main__":
    main()
