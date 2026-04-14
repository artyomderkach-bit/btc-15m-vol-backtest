#!/usr/bin/env python3
"""Analyze live Kalshi trades vs backtest expectations."""
import os, sys, json, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_fetcher import DataFetcher

fetcher = DataFetcher('KXBTC15M')

fills = []
cursor = None
for page in range(50):
    params = {'limit': 100}
    if cursor:
        params['cursor'] = cursor
    resp = fetcher._api_get('/portfolio/fills', params=params)
    batch = resp.get('fills', [])
    fills.extend(batch)
    cursor = resp.get('cursor')
    if not batch or not cursor:
        break
    time.sleep(0.5)

print(f"Total fills fetched: {len(fills)}")

market_data = defaultdict(lambda: {
    'buy_qty': 0, 'sell_qty': 0, 'buy_cost': 0.0, 'sell_rev': 0.0,
    'side': '', 'fees': 0.0, 'sell_side': ''
})

for f in fills:
    ticker = f['ticker']
    action = f['action']
    count = f['count']
    side = f['side']
    fee = float(f.get('fee_cost', '0'))
    no_price = f.get('no_price', 0) / 100.0
    yes_price = f.get('yes_price', 0) / 100.0

    md = market_data[ticker]
    md['fees'] += fee

    if action == 'buy':
        cost_per = no_price if side == 'no' else yes_price
        md['buy_cost'] += count * cost_per + fee
        md['side'] = side
        md['buy_qty'] += count
    elif action == 'sell':
        rev_per = no_price if side == 'no' else yes_price
        md['sell_rev'] += count * rev_per - fee
        md['sell_qty'] += count
        md['sell_side'] = side

# Show all sell fills
print("\n=== ALL SELL FILLS ===")
sell_fills = [f for f in fills if f['action'] == 'sell']
for f in sell_fills[:25]:
    t = f['ticker']
    s = f['side']
    q = f['count']
    np = f.get('no_price', 0)
    yp = f.get('yes_price', 0)
    fee = f.get('fee_cost', '0')
    print(f"  {t}  side={s}  qty={q}  no_price={np}c  yes_price={yp}c  fee={fee}")

# Check: for sells, what side is the ORIGINAL buy?
print("\n=== SELL SIDE vs BUY SIDE MISMATCH CHECK ===")
mismatches = 0
for ticker, md in market_data.items():
    if md['sell_qty'] > 0 and md['side'] and md['sell_side']:
        if md['side'] != md['sell_side']:
            mismatches += 1
            if mismatches <= 5:
                print(f"  {ticker}: bought {md['side']}, sold {md['sell_side']}")
if mismatches == 0:
    print("  No mismatches (sells always on same side as buys? checking...)")
    for ticker, md in market_data.items():
        if md['sell_qty'] > 0:
            print(f"  {ticker}: buy_side={md['side']}  sell_side={md['sell_side']}  buy_qty={md['buy_qty']}  sell_qty={md['sell_qty']}")
            break
else:
    print(f"  Total mismatches: {mismatches}")

# Summary
total_markets = len(market_data)
sold_markets = sum(1 for m in market_data.values() if m['sell_qty'] > 0)
unsold = total_markets - sold_markets
total_buy = sum(m['buy_cost'] for m in market_data.values())
total_sell = sum(m['sell_rev'] for m in market_data.values())
total_fees = sum(m['fees'] for m in market_data.values())

print(f"\n{'='*60}")
print(f"  LIVE TRADING ANALYSIS")
print(f"{'='*60}")
print(f"  Markets entered:    {total_markets}")
print(f"  Markets with TP:    {sold_markets} ({sold_markets/total_markets*100:.1f}%)")
print(f"  Markets to settle:  {unsold} ({unsold/total_markets*100:.1f}%)")
print(f"  Total buy cost:     ${total_buy:.2f}")
print(f"  Total sell revenue: ${total_sell:.2f}")
print(f"  Total fees:         ${total_fees:.2f}")
print(f"  Net (fills only):   ${total_sell - total_buy:+.2f}")
print(f"  Settlement value:   $0 (all settled positions lost)")
print()
print(f"  LIVE TP rate:       {sold_markets/total_markets*100:.1f}%")
print(f"  BACKTEST TP rate:   ~43%")
print(f"  DIFFERENCE:         {sold_markets/total_markets*100 - 43:.1f} percentage points")

# Expected P&L table
print(f"\n{'='*60}")
print(f"  EXPECTED P&L AT DIFFERENT TP RATES ($7/market)")
print(f"{'='*60}")
for tp in [29.5, 35, 40, 43, 50]:
    per_mkt = (tp/100) * 16.10 + (1 - tp/100) * (-7.00)
    total = per_mkt * total_markets
    print(f"  TP={tp:.1f}%: ${per_mkt:+.2f}/market = ${total:+.2f} over {total_markets} markets")

# Balance
try:
    bal = fetcher._api_get('/portfolio/balance')
    print(f"\n  Current balance: ${bal.get('balance', 0)/100:.2f}")
except:
    pass

fetcher.close()
