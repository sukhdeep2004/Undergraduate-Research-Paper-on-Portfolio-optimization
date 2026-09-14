from __future__ import annotations
from typing import Iterable, Mapping, Optional
import pandas as pd


def screen_universe(
    returns: pd.DataFrame,
    *,
    exclude_tickers: Optional[Iterable[str]] = None,
    exclude_sectors: Optional[Iterable[str]] = None,
    sector_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    drop: set[str] = set(exclude_tickers or [])
    if exclude_sectors:
        if sector_map is None:
            raise ValueError("exclude_sectors requires a sector_map (ticker -> sector).")
        wanted = {s.lower() for s in exclude_sectors}
        drop |= {t for (t, sec) in sector_map.items() if sec.lower() in wanted}
    kept = [c for c in returns.columns if c not in drop]
    if len(kept) < 2:
        raise ValueError(
            f"Screen left {len(kept)} asset(s); need at least 2. Dropped: {sorted(drop)}"
        )
    return returns.loc[:, kept].copy()


def build_sector_groups(
    columns: Iterable[str], sector_map: Mapping[str, str]
) -> "dict[str, tuple[list[int], None]]":
    cols = list(columns)
    groups: dict[str, list[int]] = {}
    for i, ticker in enumerate(cols):
        sector = sector_map.get(ticker)
        if sector is None:
            continue
        groups.setdefault(sector, []).append(i)
    return {sector: (idx, None) for (sector, idx) in groups.items()}


def equal_sector_targets(
    columns: Iterable[str], sector_map: Mapping[str, str]
) -> "dict[str, tuple[list[int], float]]":
    groups = build_sector_groups(columns, sector_map)
    g = len(groups)
    if g == 0:
        raise ValueError("No sectors matched the given columns.")
    target = 1.0 / g
    return {sector: (idx, target) for (sector, (idx, _)) in groups.items()}
