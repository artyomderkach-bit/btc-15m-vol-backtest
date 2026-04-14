#!/usr/bin/env python3
"""
Run backtest on the N most recent settled markets that close strictly before
END_DATE_EXCL (UTC midnight). Then copy the trade log to current_strat_full_tradelog.csv
and regenerate current_strat_full_pnl.png.

Default END_DATE_EXCL=2026-04-07 → includes all closes through 2026-04-06 UTC
(~within one calendar day of 2026-04-05 when run from early April).

Usage:
  python3 run_backtest_through_date.py
  python3 run_backtest_through_date.py 2026-04-07 10000
"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone

_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)
sys.path.insert(0, _script_dir)

from engine import Engine


def exclusive_end_ts(date_yyyy_mm_dd: str) -> float:
    """Unix seconds: first instant of that calendar day in UTC (exclusive upper bound for closes)."""
    dt = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def main():
    end_date_excl = sys.argv[1] if len(sys.argv) > 1 else "2026-04-07"
    num_markets = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    bankroll = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0
    vol_fill = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
    tp_fill = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

    max_close_exclusive_ts = exclusive_end_ts(end_date_excl)
    print(f"Backtest window: most recent {num_markets} settled markets with close_time < "
          f"{end_date_excl} 00:00 UTC (unix {max_close_exclusive_ts:.0f})")
    print()

    engine = Engine(
        series_ticker="KXBTC15M",
        bankroll=bankroll,
        risk_pct=0.01,
        num_markets=num_markets,
        volume_fill_pct=vol_fill,
        sell_price=0.40,
        tp_fill_rate=tp_fill,
        refresh_markets=True,
        max_close_exclusive_ts=max_close_exclusive_ts,
    )
    results = engine.run()
    engine.fetcher.close()

    df = results["df"]
    if df.empty:
        print("No trades — not updating charts.", file=sys.stderr)
        sys.exit(1)

    trades_path = os.path.join(_script_dir, "trades_KXBTC15M.csv")
    df.to_csv(trades_path, index=False)
    print(f"\nTrade log saved to {trades_path}")

    chart_log = os.path.join(_script_dir, "charts", "btc_main", "current_strat_full_tradelog.csv")
    os.makedirs(os.path.dirname(chart_log), exist_ok=True)
    shutil.copyfile(trades_path, chart_log)
    print(f"Copied to {chart_log} (for PnL chart)")

    if df["ts"].notna().any():
        last_ts = float(df["ts"].dropna().max())
        last_utc = datetime.fromtimestamp(last_ts, tz=timezone.utc)
        print(f"Last trade row ts (UTC): {last_utc:%Y-%m-%d %H:%M:%S}")

    import plot_current_strat_pnl_vix
    plot_current_strat_pnl_vix.main()
    print("Done.")


if __name__ == "__main__":
    main()
