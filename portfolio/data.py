from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional, Union
import pandas as pd

DATA_CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"
SECTOR_ETFS: "dict[str, str]" = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}


def download_prices(
    tickers: Union[str, "list[str]"], start: str, end: Optional[str] = None
) -> pd.DataFrame:
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [tickers]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw["Close"] if "Close" in raw.columns else raw
    if len(tickers) == 1:
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
        else:
            prices = prices.rename(columns={prices.columns[0]: tickers[0]})
    prices = prices.reindex(columns=[t for t in tickers if t in prices.columns])
    return prices.dropna(how="any")


def _cache_key(tickers: "list[str]", start: str, end: Optional[str]) -> str:
    raw = "|".join(sorted(tickers)) + f"|{start}|{end}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_prices(
    tickers: Union[str, "list[str]", "dict[str, str]"],
    start: str,
    end: Optional[str] = None,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    if isinstance(tickers, dict):
        tickers = list(tickers.keys())
    elif isinstance(tickers, str):
        tickers = [tickers]
    cache = Path(cache_dir) if cache_dir is not None else DATA_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"prices_{_cache_key(tickers, start, end)}.csv"
    if path.exists() and (not refresh):
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
        return prices.reindex(columns=[t for t in tickers if t in prices.columns])
    prices = download_prices(tickers, start=start, end=end)
    prices.to_csv(path)
    return prices


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")
