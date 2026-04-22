#!/usr/bin/env python3
"""
Run stink-bid-at-open backtest for BTC 15m style markets.

Usage:
    python run_stink_bid_backtest.py KXBTC15M --num-markets 500 --bankroll 1000
    python run_stink_bid_backtest.py KXBTC15M --bad-prices-yes 0.01,0.02,0.03 --sides YES,NO
"""
from __future__ import annotations

import argparse
import os

from stink_bid_config import StinkBidConfig
from stink_bid_engine import StinkBidEngine


def _parse_csv_floats(v: str):
    if v is None or not str(v).strip():
        return tuple()
    out = []
    for x in str(v).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return tuple(out)


def _parse_sides(v: str):
    if v is None or not str(v).strip():
        return ("YES",)
    return tuple(x.strip().upper() for x in str(v).split(",") if x.strip())


def main():
    p = argparse.ArgumentParser(description="BTC 15m stink-bid backtest")
    p.add_argument("series", nargs="?", default="KXBTC15M")
    p.add_argument("--num-markets", type=int, default=500)
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--refresh", type=int, default=0)

    p.add_argument("--opening-delay-ms", type=int, default=0)
    p.add_argument("--cancel-timeout-ms", type=int, default=500)
    p.add_argument("--num-levels", type=int, default=5)
    p.add_argument("--max-contracts-per-level", type=int, default=1)
    p.add_argument("--max-notional-per-market", type=float, default=50.0)
    p.add_argument("--max-open-orders", type=int, default=10)
    p.add_argument("--volume-fill-pct", type=float, default=0.10)
    p.add_argument("--replace-canceled", type=int, default=0)

    p.add_argument("--sides", type=str, default="YES")
    p.add_argument("--bad-price-mode", type=str, default="absolute")
    p.add_argument("--bad-prices-yes", type=str, default="0.01,0.02,0.03,0.05,0.10")
    p.add_argument("--bad-prices-no", type=str, default="0.01,0.02,0.03,0.05,0.10")
    p.add_argument("--pct-offsets", type=str, default="0.20,0.30,0.40,0.50,0.70")

    p.add_argument("--exit-mode", type=str, default="time_exit")
    p.add_argument("--time-exit-minutes", type=int, default=5)
    p.add_argument("--assume-fee-per-contract", type=float, default=0.0)
    p.add_argument("--silent", action="store_true")
    args = p.parse_args()

    cfg = StinkBidConfig(
        series_ticker=args.series,
        opening_delay_ms=args.opening_delay_ms,
        cancel_timeout_ms=args.cancel_timeout_ms,
        sides=_parse_sides(args.sides),
        num_levels=args.num_levels,
        bad_price_mode=args.bad_price_mode,
        bad_prices_yes=_parse_csv_floats(args.bad_prices_yes),
        bad_prices_no=_parse_csv_floats(args.bad_prices_no),
        pct_offsets_below_best_bid=_parse_csv_floats(args.pct_offsets),
        max_contracts_per_level=args.max_contracts_per_level,
        max_notional_per_market=args.max_notional_per_market,
        max_open_orders=args.max_open_orders,
        volume_fill_pct=args.volume_fill_pct,
        replace_canceled=bool(int(args.replace_canceled)),
        exit_mode=args.exit_mode,
        time_exit_minutes=args.time_exit_minutes,
        assume_fee_per_contract=args.assume_fee_per_contract,
    )

    eng = StinkBidEngine(
        series_ticker=args.series,
        bankroll=args.bankroll,
        num_markets=args.num_markets,
        refresh_markets=(args.refresh == 1),
        config=cfg,
    )
    res = eng.run(silent=args.silent)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    trades_path = os.path.join(out_dir, f"stink_bid_trades_{args.series}.csv")
    summary_path = os.path.join(out_dir, f"stink_bid_summary_{args.series}.csv")

    df = res.get("df")
    if df is not None and not df.empty:
        df.to_csv(trades_path, index=False)
    summary_df = res.get("summary_df")
    if summary_df is not None and not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)

    if not args.silent:
        print(f"\nTrade log saved to {trades_path}")
        print(f"Summary saved to {summary_path}")

    return res


if __name__ == "__main__":
    main()
