#!/usr/bin/env python3
"""
Compare setups: event-day vs normal-day PnL.
Run: python compare_event_vs_normal.py [num_markets]
"""
import sys
sys.path.insert(0, '.')
from engine import Engine

EVENT_DATES = {'26JAN28','26JAN29','26FEB18','26FEB19','26FEB20',
               '26FEB24','26MAR01','26MAR02','26MAR03'}

def ticker_date(t):
    parts = t.split('-')
    return parts[1][:7] if len(parts) >= 2 else ''

combos = [
    (0.10, 0.33, '10/33 (current)'),
    (0.10, 0.50, '10/50'),
    (0.11, 0.33, '11/33'),
    (0.11, 0.53, '11/53'),
    (0.09, 0.45, ' 9/45'),
    (0.12, 0.40, '12/40'),
    (0.08, 0.50, ' 8/50'),
    (0.15, 0.40, '15/40'),
    (0.20, 0.40, '20/40'),
    (0.20, 0.50, '20/50'),
]

num_markets = int(sys.argv[1]) if len(sys.argv) > 1 else 7300
print(f"Comparing setups on {num_markets} markets...")
print()

print(f"{'Setup':>16}  {'Return':>8}  {'Fills':>6}  {'TP%':>6}  {'DD%':>6}  {'Evt PnL':>10}  {'Norm PnL':>10}  {'Norm/mkt':>10}")
print(f"{'-'*16}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

for buy_p, sell_p, label in combos:
    eng = Engine(
        series_ticker='KXBTC15M', bankroll=1000.0, risk_pct=0.03,
        num_markets=num_markets, volume_fill_pct=0.10,
        buy_price=buy_p, sell_price=sell_p,
    )
    r = eng.run(silent=True)
    df = r['df']
    if df.empty:
        continue

    closes = df[df['action'].isin(['sell_fill','settlement'])]
    closes_evt = closes[closes['market'].apply(lambda m: ticker_date(m) in EVENT_DATES)]
    closes_norm = closes[closes['market'].apply(lambda m: ticker_date(m) not in EVENT_DATES)]

    evt_pnl = closes_evt['pnl'].sum()
    norm_pnl = closes_norm['pnl'].sum()
    norm_markets = closes_norm['market'].nunique()
    pnl_per_norm = norm_pnl / norm_markets if norm_markets else 0

    print(f"{label:>16}  {r['total_return_pct']:>+7.1f}%  {r['markets_filled']:>6}  "
          f"{r['tp_pct']:>5.1f}%  {r['max_drawdown_pct']:>5.1f}%  "
          f"${evt_pnl:>+8,.0f}  ${norm_pnl:>+8,.0f}  ${pnl_per_norm:>+8.2f}")

print()
print("Norm/mkt = avg PnL per market on NON-event days (less negative = less bleed)")
