#!/usr/bin/env python3
"""
Run backtest starting from March 11 2026 00:00 UTC with $300 bankroll.
Saves to trades_KXBTC15M_recent.csv.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine

_dir = os.path.dirname(os.path.abspath(__file__))

START_DATE = datetime(2026, 3, 11, 0, 0, 0)
START_TS = int(START_DATE.timestamp())
BANKROLL = 300.0

engine = Engine(
    series_ticker='KXBTC15M',
    bankroll=BANKROLL,
    risk_pct=0.01,
    num_markets=5000,
    volume_fill_pct=0.10,
    sell_price=0.40,
    tp_fill_rate=1.0,
    refresh_markets=False,
    start_after_ts=START_TS,
)

results = engine.run()

df = results['df']
if not df.empty:
    csv_path = os.path.join(_dir, 'trades_KXBTC15M_recent.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nRecent trade log saved to {csv_path}")
else:
    print("\nNo trades in recent period.")
