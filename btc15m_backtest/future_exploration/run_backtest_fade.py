#!/usr/bin/env python3
"""
Run the AdaptiveLimitFadeStrategy backtest.

Reuses the existing Engine (data fetching, candle parsing, reporting) but swaps
in the fade strategy before running.

Usage (from repo root, after cd btc15m_backtest):
    python future_exploration/run_backtest_fade.py                          # 7000 markets, $1000
    python future_exploration/run_backtest_fade.py 1000                     # 1000 markets
    python future_exploration/run_backtest_fade.py 7000 0.10 1000 0.003     # custom vol threshold
"""
import sys
import os
import re

_FE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_FE)
sys.path[:0] = [_ROOT, _FE]

from engine import Engine
from strategy_fade import AdaptiveLimitFadeStrategy

_dir = _FE


def _parse_int(s, default):
    try:
        return int(re.sub(r'[^\d-]', '', s) or '0') or default
    except (ValueError, TypeError):
        return default


def _parse_float(s, default):
    try:
        cleaned = re.sub(r'[^\d.]', '', s)
        return float(cleaned) if cleaned else default
    except (ValueError, TypeError):
        return default


NUM_MARKETS   = _parse_int(sys.argv[1], 7000) if len(sys.argv) > 1 else 7000
VOL_FILL_PCT  = _parse_float(sys.argv[2], 0.10) if len(sys.argv) > 2 else 0.10
BANKROLL      = _parse_float(sys.argv[3], 1000.0) if len(sys.argv) > 3 else 1000.0
VOL_THRESHOLD = _parse_float(sys.argv[4], 0.80) if len(sys.argv) > 4 else 0.80
TP_FILL_RATE  = _parse_float(sys.argv[5], 1.0) if len(sys.argv) > 5 else 1.0
REFRESH       = _parse_int(sys.argv[6], 0) if len(sys.argv) > 6 else 0

SERIES = 'KXBTC15M'

# Build engine (creates DataFetcher, default Strategy)
engine = Engine(
    series_ticker=SERIES,
    bankroll=BANKROLL,
    risk_pct=0.015,
    num_markets=NUM_MARKETS,
    volume_fill_pct=VOL_FILL_PCT,
    tp_fill_rate=TP_FILL_RATE,
    refresh_markets=(REFRESH == 1),
)

# Swap in AdaptiveLimitFadeStrategy before running
engine.strategy = AdaptiveLimitFadeStrategy(
    bankroll=BANKROLL,
    risk_pct=0.015,
    vol_threshold=VOL_THRESHOLD,
    entry_price=0.10,
    tp1_price=0.33,
    tp2_price=0.50,
    entry_expiry_minutes=6,
    vol_window_minutes=3,
    stop_minutes=7,
    active_hours=(13, 20),
    skip_weekends=True,
    volume_fill_pct=VOL_FILL_PCT,
    tp_fill_rate=TP_FILL_RATE,
)

s = engine.strategy
print(f"Strategy: AdaptiveLimitFadeStrategy")
print(f"  Vol threshold: {VOL_THRESHOLD:.4f} ({VOL_THRESHOLD*100:.2f}%)")
print(f"  Entry: {s.entry_price}  |  TP1: {s.tp1_price}  |  TP2: {s.tp2_price}")
print(f"  Entry window: 3-6 min  |  Vol window: 0-3 min  |  Time stop: {s.stop_seconds//60} min")
print(f"  Active hours (UTC): {s.active_hours[0]:02d}:00-{s.active_hours[1]:02d}:00  |  Skip weekends: {s.skip_weekends}")
print()

results = engine.run()
strat = engine.strategy

# ── Fade-specific summary ──
print()
print("=" * 60)
print("  ADAPTIVE LIMIT FADE — STRATEGY-SPECIFIC STATS")
print("=" * 60)
print()
print("  FILTERS")
print(f"    Weekend markets skipped:  {strat.weekend_skips}")
print(f"    Off-hour markets skipped: {strat.hour_skips}")
total_considered = strat.vol_gate_triggered + strat.vol_gate_skipped
gate_pct = (strat.vol_gate_triggered / total_considered * 100
            if total_considered > 0 else 0)
print(f"    Volatility gate triggered:{strat.vol_gate_triggered:>5}  ({gate_pct:.1f}% of eligible)")
print(f"    Volatility gate skipped:  {strat.vol_gate_skipped:>5}")
print()
print("  EXITS")
print(f"    TP1 fills (33¢, 50%):     {strat.tp1_fills}")
print(f"    TP2 fills (45¢, 50%):     {strat.tp2_fills}")
print(f"    Total TP fills:           {strat.tp1_fills + strat.tp2_fills}")
print(f"    Time stops (min 10):      {strat.time_stops}")

df = results['df']
if not df.empty:
    settles = df[df['action'] == 'settlement']
    time_stops = df[df['action'] == 'time_stop']
    settle_wins = settles[settles.get('won', False) == True] if 'won' in settles.columns else settles[settles['pnl'] > 0]
    settle_losses = settles[settles['pnl'] <= 0] if 'won' not in settles.columns else settles[settles.get('won', True) == False]
    print(f"    Settlement wins:          {len(settle_wins)}")
    print(f"    Settlement losses:        {len(settle_losses)}")
    print()

    print("  P&L BREAKDOWN")
    tp_pnl = df[df['action'] == 'sell_fill']['pnl'].sum()
    ts_pnl = time_stops['pnl'].sum() if len(time_stops) > 0 else 0
    st_pnl = settles['pnl'].sum() if len(settles) > 0 else 0
    print(f"    From TP exits:            ${tp_pnl:+,.2f}")
    print(f"    From time stops:          ${ts_pnl:+,.2f}")
    print(f"    From settlements:         ${st_pnl:+,.2f}")
    print()

    exits = df[df['action'].isin(['sell_fill', 'settlement', 'time_stop'])]
    avg_pnl = exits['pnl'].mean() if len(exits) > 0 else 0
    total_pnl = exits['pnl'].sum() if len(exits) > 0 else 0
    ts_avg = time_stops['pnl'].mean() if len(time_stops) > 0 else 0
    print(f"  Avg P&L per exit:           ${avg_pnl:+,.2f}")
    print(f"  Avg time-stop P&L:          ${ts_avg:+,.2f}")
    print(f"  Total P&L:                  ${total_pnl:+,.2f}")
print("=" * 60)

# Save trade log CSV
if not df.empty:
    csv_path = os.path.join(_dir, f"trades_{SERIES}_fade.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTrade log saved to {csv_path}")
else:
    print("\nNo trades generated.")
