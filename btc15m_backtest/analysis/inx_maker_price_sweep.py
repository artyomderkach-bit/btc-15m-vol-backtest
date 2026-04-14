#!/usr/bin/env python3
"""
Sweep maker-style buy/TP sell prices for KXINXU (hourly) using the same Strategy as BTC 15m.

Uses forward-filled minute candles (fill_minute_gaps) and scales entry cutoff vs a 60m window.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def run_one(
    buy: float,
    sell: float,
    *,
    num_markets: int,
    entry_cutoff_seconds: int,
    volume_fill_pct: float = 0.10,
) -> dict:
    eng = Engine(
        series_ticker="KXINXU",
        bankroll=1000.0,
        risk_pct=0.01,
        num_markets=num_markets,
        volume_fill_pct=volume_fill_pct,
        buy_price=buy,
        sell_price=sell,
        tp_fill_rate=1.0,
        refresh_markets=False,
        entry_cutoff_seconds=entry_cutoff_seconds,
        fill_minute_gaps=True,
    )
    return eng.run(silent=True)


def main():
    num_markets = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    # ~20m bid resting time on a 60m contract (BTC default 240s on 15m ≈ 27% → 960s ≈ 16m)
    entry_cutoff_default = 1200

    # Pairs: (buy, sell) — sell must exceed buy for long YES/NO lottery + TP story
    pairs = [
        (0.10, 0.40),
        (0.10, 0.35),
        (0.08, 0.30),
        (0.08, 0.35),
        (0.12, 0.40),
        (0.15, 0.40),
        (0.15, 0.45),
        (0.18, 0.45),
        (0.20, 0.50),
        (0.20, 0.45),
        (0.05, 0.25),
        (0.25, 0.50),
        (0.30, 0.55),
    ]

    rows = []
    print(f"KXINXU maker sweep | markets={num_markets} | entry_cutoff_sec={entry_cutoff_default} | vol_fill={0.10}", flush=True)

    for buy, sell in pairs:
        if sell <= buy:
            continue
        r = run_one(buy, sell, num_markets=num_markets, entry_cutoff_seconds=entry_cutoff_default)
        row = {
            "buy": buy,
            "sell": sell,
            "net_pnl": r["net_pnl"],
            "total_return_pct": r["total_return_pct"],
            "markets_traded": r["markets_traded"],
            "markets_filled": r["markets_filled"],
            "fill_rate_pct": (100.0 * r["markets_filled"] / r["markets_traded"]) if r["markets_traded"] else 0.0,
            "tp_count": r["tp_count"],
            "tp_pct_of_filled": r["tp_pct"],
            "settled_count": r["settled_count"],
            "settle_win_rate": r["settle_win_rate"],
            "sharpe": r["sharpe"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "note": f"entry_cutoff_{entry_cutoff_default}s",
        }
        rows.append(row)
        print(
            f"  buy={buy:.2f} sell={sell:.2f} | pnl=${r['net_pnl']:.2f} ret={r['total_return_pct']:.2f}% "
            f"filled={r['markets_filled']}/{r['markets_traded']} tp={r['tp_count']} sharpe={r['sharpe']:.2f}",
            flush=True,
        )

    # Baseline 10/40 with shorter cutoff (BTC-like 240s) for comparison
    r240 = run_one(0.10, 0.40, num_markets=num_markets, entry_cutoff_seconds=240)
    rows.append(
        {
            "buy": 0.10,
            "sell": 0.40,
            "net_pnl": r240["net_pnl"],
            "total_return_pct": r240["total_return_pct"],
            "markets_traded": r240["markets_traded"],
            "markets_filled": r240["markets_filled"],
            "fill_rate_pct": (100.0 * r240["markets_filled"] / r240["markets_traded"]) if r240["markets_traded"] else 0.0,
            "tp_count": r240["tp_count"],
            "tp_pct_of_filled": r240["tp_pct"],
            "settled_count": r240["settled_count"],
            "settle_win_rate": r240["settle_win_rate"],
            "sharpe": r240["sharpe"],
            "max_drawdown_pct": r240["max_drawdown_pct"],
            "note": "entry_cutoff_240s",
        }
    )
    print(
        f"  buy=0.10 sell=0.40 (cutoff 240s only) | pnl=${r240['net_pnl']:.2f} filled={r240['markets_filled']}/{r240['markets_traded']}",
        flush=True,
    )

    path = os.path.join(OUT_DIR, "inx_maker_price_sweep.csv")
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}", flush=True)

    best = max(rows, key=lambda x: x["net_pnl"])
    print(f"Best net_pnl row: buy={best.get('buy')} sell={best.get('sell')} pnl={best['net_pnl']:.2f} note={best.get('note', '')}", flush=True)


if __name__ == "__main__":
    main()
