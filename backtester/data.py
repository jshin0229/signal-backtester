"""Price loading with a local CSV cache so repeated runs don't hit the network."""
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(ticker: str, start: str = "2010-01-01", end: str | None = None,
                use_cache: bool = True) -> pd.Series:
    """Return a daily adjusted close price series for ``ticker``."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{ticker.upper()}.csv"
    if use_cache and cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf
        df = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker!r}")
        s = df["Close"].squeeze().rename(ticker.upper())
        s.to_csv(cache)
    s = s.loc[start:end] if end else s.loc[start:]
    return s.dropna().rename(ticker.upper())
