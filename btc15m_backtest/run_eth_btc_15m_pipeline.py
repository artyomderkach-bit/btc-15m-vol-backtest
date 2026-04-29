#!/usr/bin/env python3
"""
CLI: Kalshi KXETH15M vs KXBTC15M 15-minute binary pipeline (2-second aligned grid).

Run from this directory:
    python run_eth_btc_15m_pipeline.py --no-refresh --max-events 5
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from eth_btc_15m_pipeline import run_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Kalshi KXETH15M vs KXBTC15M 15m binary pipeline (2s grid)")
    p.add_argument("--no-refresh", action="store_true", help="Do not refresh market lists from API first")
    p.add_argument("--eth-series", default="KXETH15M", help="ETH 15m series ticker")
    p.add_argument("--btc-series", default="KXBTC15M", help="BTC 15m series ticker")
    p.add_argument("--out-dir", default=None, help="Output root (default: outputs/eth_btc_15m)")
    p.add_argument("--market-limit", type=int, default=20000, help="Max settled markets per series")
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Process only the N most recent matched 15-min windows (for testing)",
    )
    p.add_argument("--chart-n", type=int, default=9, help="Most recent matched windows to plot (3x3 default)")
    p.add_argument("--min-both-pct", type=float, default=0.5, help="overlap_stats: min fraction of bins with prints on both sides")
    p.add_argument(
        "--min-overlap-bins",
        type=int,
        default=60,
        help="overlap_stats: min number of 2s bins in window for ready_for_cointegration",
    )
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(_SCRIPT_DIR, "outputs", "eth_btc_15m")
    summary = run_pipeline(
        out_dir=out_dir,
        eth_series=args.eth_series,
        btc_series=args.btc_series,
        refresh_markets=not args.no_refresh,
        market_limit=args.market_limit,
        max_matched_events=args.max_events,
        render_chart_n=args.chart_n,
        min_both_pct=args.min_both_pct,
        min_overlap_bins=args.min_overlap_bins,
    )
    print(summary)


if __name__ == "__main__":
    main()
