#!/usr/bin/env python3
"""
Run monotonicity (ladder) arbitrage backtest — grouped multi-market path.

Usage (from btc15m_backtest directory):
    python run_monotonicity_backtest.py KXBTC15M 500 1000 --refresh 0
    python run_monotonicity_backtest.py --help

Does not modify the legacy run_backtest.py / Engine single-market flow.
"""
import argparse
import os
import sys

from monotonicity_config import MonotonicityConfig
from monotonicity_engine import MonotonicityEngine


def main():
    p = argparse.ArgumentParser(description="Monotonicity ladder backtest")
    p.add_argument("series", nargs="?", default="KXBTC15M", help="Kalshi series ticker")
    p.add_argument("num_markets", nargs="?", type=int, default=500, help="Max markets to fetch")
    p.add_argument("bankroll", nargs="?", type=float, default=1000.0, help="Starting bankroll")
    p.add_argument("--refresh", type=int, default=0, help="1 = refresh markets from API")
    p.add_argument("--min-net-violation", type=float, default=None, help="Min net edge after costs")
    p.add_argument("--min-raw-violation", type=float, default=None)
    p.add_argument("--max-contracts", type=int, default=None, help="Max contracts per leg")
    p.add_argument("--silent", action="store_true")
    args = p.parse_args()

    cfg = MonotonicityConfig()
    if args.min_net_violation is not None:
        cfg.min_net_violation = args.min_net_violation
    if args.min_raw_violation is not None:
        cfg.min_raw_violation = args.min_raw_violation
    if args.max_contracts is not None:
        cfg.max_contracts_per_leg = args.max_contracts

    eng = MonotonicityEngine(
        series_ticker=args.series,
        bankroll=args.bankroll,
        num_markets=args.num_markets,
        refresh_markets=(args.refresh == 1),
        config=cfg,
    )
    results = eng.run(silent=args.silent)

    df = results.get("df")
    if df is not None and not df.empty:
        _dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(_dir, f"trades_monotonicity_{args.series}.csv")
        df.to_csv(path, index=False)
        print(f"\nTrade log saved to {path}")

    return results


if __name__ == "__main__":
    main()
