#!/usr/bin/env python3
"""
CLI: Kalshi weekly WTI vs Brent data pipeline (markets, trades, panels, diagnostics).

Run from this directory:
    python run_wti_brent_pipeline.py
    python run_wti_brent_pipeline.py --no-refresh --max-events 2
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from wti_brent_pipeline import run_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Kalshi KXWTIW vs KXBRENTW weekly pipeline")
    p.add_argument(
        "--no-refresh",
        action="store_true",
        help="Do not refresh market lists from API before reading cache (faster)",
    )
    p.add_argument("--wti-series", default="KXWTIW", help="WTI weekly series ticker")
    p.add_argument("--brent-series", default="KXBRENTW", help="Brent weekly series ticker")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output root (default: btc15m_backtest/outputs/wti_brent)",
    )
    p.add_argument(
        "--market-limit",
        type=int,
        default=5000,
        help="Max settled markets to consider per series (DataFetcher limit)",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Process only the N most recent matched Fridays (for testing)",
    )
    p.add_argument(
        "--freqs",
        default="1m,5m,15m",
        help="Comma-separated bar sizes: 1m,5m,15m",
    )
    p.add_argument(
        "--min-both-pct",
        type=float,
        default=0.5,
        help="overlap_stats ready_for_cointegration: min fraction of minutes with prints on both sides",
    )
    p.add_argument(
        "--min-overlap-minutes",
        type=int,
        default=60,
        help="ready_for_cointegration: minimum overlap length in minutes (1m bins)",
    )
    args = p.parse_args()

    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.join(_SCRIPT_DIR, "outputs", "wti_brent")

    freqs = tuple(x.strip() for x in str(args.freqs).split(",") if x.strip())
    for f in freqs:
        if f not in ("1m", "5m", "15m"):
            print(f"ERROR: unsupported freq {f!r}", file=sys.stderr)
            sys.exit(1)

    summary = run_pipeline(
        out_dir=out_dir,
        wti_series=args.wti_series,
        brent_series=args.brent_series,
        refresh_markets=not args.no_refresh,
        market_limit=args.market_limit,
        max_matched_events=args.max_events,
        freqs=freqs,
        min_both_pct=args.min_both_pct,
        min_overlap_minutes=args.min_overlap_minutes,
    )
    print(summary)


if __name__ == "__main__":
    main()
