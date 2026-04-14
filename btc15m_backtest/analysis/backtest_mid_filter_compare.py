#!/usr/bin/env python3
"""
Compare baseline vs mid-price filter [0.15, 0.85] on bar 4.

  * baseline     — orders at market open, 4 min entry cutoff (current_strat style).
  * delayed      — wait 4 bars, then skip if mid outside band; entry window extended
                   by 4 min (deployable, causal).
  * oracle       — skip before open if bar-4 mid would be outside band, but orders
                   still rest from t=0 (NOT deployable: uses future of first 4 minutes).
                   Shows how much of the observational pattern is “selection on mid”
                   without changing when you join the queue.

Matches current_strat-style params: 10¢ buy, 40¢ TP, 4 min entry after rest, 2% risk.

Usage:
  python3 analysis/backtest_mid_filter_compare.py [NUM_MARKETS]
"""
from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from engine import Engine


def _run(label: str, *, mode: str, num_markets: int) -> dict:
    eng = Engine(
        series_ticker="KXBTC15M",
        bankroll=1000.0,
        risk_pct=0.02,
        num_markets=num_markets,
        volume_fill_pct=0.10,
        buy_price=0.10,
        sell_price=0.40,
        tp_fill_rate=1.0,
        refresh_markets=False,
        entry_cutoff_seconds=240,
        mid_filter_min=0.15,
        mid_filter_max=0.85,
        mid_filter_bars=4,
        mid_filter_mode=mode,
    )
    r = eng.run(silent=True)
    df = r["df"]
    skipped_d = skipped_o = 0
    if not df.empty and "action" in df.columns:
        skipped_d = int((df["action"] == "skip_mid_filter").sum())
        skipped_o = int((df["action"] == "skip_mid_oracle").sum())
    skipped = skipped_d + skipped_o
    return {
        "label": label,
        "mode": mode,
        "final_bankroll": r["final_bankroll"],
        "net_pnl": r["net_pnl"],
        "total_return_pct": r["total_return_pct"],
        "markets_traded": r["markets_traded"],
        "markets_filled": r["markets_filled"],
        "tp_count": r["tp_count"],
        "max_drawdown_pct": r["max_drawdown_pct"],
        "sharpe": r["sharpe"],
        "skipped_mid": skipped,
    }


def main():
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    print(f"Running {num} markets per variant (cached API data)…\n")

    rows = [
        _run("baseline (no mid filter)", mode="off", num_markets=num),
        _run("delayed: bar-4 mid, then trade + extended cutoff", mode="delayed", num_markets=num),
        _run("oracle: skip from t=0 using bar-4 mid (research only)", mode="oracle", num_markets=num),
    ]

    for row in rows:
        print(f"--- {row['label']} ---")
        print(f"  Final bankroll:     ${row['final_bankroll']:,.2f}")
        print(f"  Net P&L:            ${row['net_pnl']:+,.2f}")
        print(f"  Total return:       {row['total_return_pct']:+.2f}%")
        print(f"  Markets traded:     {row['markets_traded']}")
        print(f"  Markets filled:     {row['markets_filled']}")
        print(f"  TP hits:            {row['tp_count']}")
        print(f"  Skipped (mid rule): {row['skipped_mid']}")
        print(f"  Max DD %:           {row['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe (ann.):      {row['sharpe']:.2f}")
        print()

    base = rows[0]
    for row in rows[1:]:
        d_pnl = row["net_pnl"] - base["net_pnl"]
        d_ret = row["total_return_pct"] - base["total_return_pct"]
        print(f"Delta ({row['mode']} − baseline): P&L ${d_pnl:+,.2f}  |  return {d_ret:+.2f} pp")


if __name__ == "__main__":
    main()
