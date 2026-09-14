from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.data import DATA_CACHE_DIR, SECTOR_ETFS, load_prices, prices_to_returns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch & clean the Week 8 sector-ETF universe.")
    p.add_argument("--start", default="2018-07-01")
    p.add_argument("--end", default=None, help="default: through today")
    p.add_argument("--refresh", action="store_true", help="force a fresh download")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = list(SECTOR_ETFS.keys())
    print("=" * 70)
    print(
        f"Fetching {len(tickers)} sector ETFs from Yahoo Finance ({args.start} -> {args.end or 'today'})"
    )
    print("=" * 70)
    prices = load_prices(tickers, args.start, args.end, refresh=args.refresh)
    prices = prices.dropna(how="any")
    returns = prices_to_returns(prices)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(DATA_CACHE_DIR / "universe_prices.csv")
    returns.to_csv(DATA_CACHE_DIR / "universe_returns.csv")
    print("\nUniverse (documented sourcing):")
    for t in tickers:
        flag = "" if t in prices.columns else "  [MISSING]"
        print(f"  {t:5s}  {SECTOR_ETFS[t]}{flag}")
    print(f"\n  assets:        {prices.shape[1]}")
    print(
        f"  price rows:    {prices.shape[0]}  ({prices.index.min().date()} -> {prices.index.max().date()})"
    )
    print(f"  return rows:   {returns.shape[0]}")
    print(f"  any missing:   {bool(prices.isna().any().any())}")
    print(f"\n  saved: {DATA_CACHE_DIR / 'universe_prices.csv'}")
    print(f"  saved: {DATA_CACHE_DIR / 'universe_returns.csv'}")


if __name__ == "__main__":
    main()
