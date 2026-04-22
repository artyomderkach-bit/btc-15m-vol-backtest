#!/usr/bin/env python3
"""
Parameter sweep for stink-bid strategy.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from itertools import product

from stink_bid_config import StinkBidConfig
from stink_bid_engine import StinkBidEngine


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep stink-bid strategy parameters")
    p.add_argument("series", nargs="?", default="KXBTC15M")
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--num-markets", type=int, default=200)
    p.add_argument("--refresh", type=int, default=0)
    args = p.parse_args()

    base = StinkBidConfig(series_ticker=args.series)
    eng = StinkBidEngine(
        series_ticker=args.series,
        bankroll=float(args.bankroll),
        num_markets=int(args.num_markets),
        refresh_markets=bool(int(args.refresh)),
        config=base,
    )

    cancel_timeouts = [250, 500, 1000, 2000, 5000]
    sides = [("YES",), ("NO",), ("YES", "NO")]
    contracts = [1, 2, 5]
    levels = [3, 5, 10]
    open_delays = [0, 100, 250, 500, 1000]
    exit_modes = ["hold_to_expiry", "time_exit"]

    rows = []
    combos = list(product(cancel_timeouts, sides, contracts, levels, open_delays, exit_modes))
    for i, (cto, sd, cpl, lvl, dly, exm) in enumerate(combos, start=1):
        cfg = replace(
            base,
            cancel_timeout_ms=cto,
            sides=sd,
            max_contracts_per_level=cpl,
            num_levels=lvl,
            opening_delay_ms=dly,
            exit_mode=exm,
        )
        res = eng.run(config=cfg, silent=True)
        row = {
            "cancel_timeout_ms": cto,
            "sides": "|".join(sd),
            "contracts_per_level": cpl,
            "num_levels": lvl,
            "opening_delay_ms": dly,
            "exit_mode": exm,
        }
        row.update(res["summary"])
        rows.append(row)
        if i % 50 == 0:
            print(f"  Completed {i}/{len(combos)}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"stink_bid_sweep_{args.series}.csv")
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Sweep written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
