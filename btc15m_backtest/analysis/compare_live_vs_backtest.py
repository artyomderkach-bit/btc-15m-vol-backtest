#!/usr/bin/env python3
"""
Compare backtest trades to live Kalshi fills. Find specific mismatches.
"""
import os
import sys
import csv
import time
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from data_fetcher import DataFetcher

SERIES = 'KXBTC15M'
TRADES_CSV = os.path.join(_ROOT, 'trades_KXBTC15M.csv')


def load_backtest_trades():
    """Parse backtest CSV. Return dict: market -> {side, outcome, qty, pnl}."""
    bt = {}
    with open(TRADES_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            m = row.get('market', '').strip()
            if not m or not m.startswith(SERIES):
                continue
            action = row.get('action', '')
            side = row.get('side', '')
            qty = int(float(row.get('qty', 0) or 0))
            pnl = float(row.get('pnl', 0) or 0)
            if m not in bt:
                bt[m] = {'side': side, 'outcome': None, 'qty': 0, 'pnl': 0}
            bt[m]['qty'] = max(bt[m]['qty'], qty)
            if action == 'buy_fill':
                bt[m]['side'] = side
            elif action == 'sell_fill':
                bt[m]['outcome'] = 'TP'
                bt[m]['pnl'] = pnl
            elif action == 'settlement':
                bt[m]['outcome'] = 'settlement'
                bt[m]['pnl'] = pnl
    for m, d in bt.items():
        if d['outcome'] is None:
            d['outcome'] = 'settlement'  # buy_fill only, no exit row
    return bt


def fetch_live_fills(fetcher):
    """Fetch all KXBTC15M fills from Kalshi API."""
    fills = []
    cursor = None
    for _ in range(100):
        params = {'limit': 200}
        if cursor:
            params['cursor'] = cursor
        resp = fetcher._api_get('/portfolio/fills', params=params)
        batch = resp.get('fills', [])
        fills.extend([f for f in batch if (f.get('ticker') or f.get('market_ticker') or '').startswith(SERIES)])
        cursor = resp.get('cursor')
        if not batch or not cursor:
            break
        time.sleep(0.4)
    return fills


def build_live_by_market(fills):
    """Aggregate fills by market. Return dict: market -> {side, outcome, buy_qty, sell_qty}."""
    live = defaultdict(lambda: {'side': '', 'buy_qty': 0, 'sell_qty': 0})
    for f in fills:
        t = f.get('ticker') or f.get('market_ticker', '')
        if not t.startswith(SERIES):
            continue
        action = (f.get('action') or '').lower()
        side = (f.get('side') or '').lower()
        count = int(float(f.get('count', 0) or f.get('count_fp', 0)))
        if action == 'buy':
            live[t]['side'] = side
            live[t]['buy_qty'] += count
        elif action == 'sell':
            live[t]['sell_qty'] += count
    for t, d in live.items():
        d['outcome'] = 'TP' if d['sell_qty'] > 0 else 'settlement'
    return dict(live)


def get_bt_markets_chronological(bt):
    """Return list of market tickers in chronological order (oldest first)."""
    return sorted(bt.keys(), key=lambda m: m)  # ticker format sorts chronologically


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare backtest vs live trades')
    parser.add_argument('--last', '-n', type=int, default=0, help='Compare only last N backtest markets')
    args = parser.parse_args()

    print("Loading backtest trades...")
    bt = load_backtest_trades()
    bt_markets_all = set(bt.keys())

    if args.last > 0:
        chrono = get_bt_markets_chronological(bt)
        last_n = chrono[-args.last:]
        bt_markets = set(last_n)
        print(f"  Backtest (last {args.last} markets): {len(bt_markets)} markets")
    else:
        bt_markets = bt_markets_all
        print(f"  Backtest: {len(bt_markets)} markets")
    print(f"  Backtest: {len(bt_markets)} markets")

    print("Fetching live fills from Kalshi API...")
    fetcher = DataFetcher(SERIES)
    fills = fetch_live_fills(fetcher)
    live = build_live_by_market(fills)
    live_markets = set(live.keys())
    print(f"  Live: {len(live_markets)} markets")
    fetcher.close()

    overlap = bt_markets & live_markets
    print(f"\nOverlap (markets in BOTH): {len(overlap)}")
    if args.last > 0:
        print(f"  (of {len(bt_markets)} backtest markets in window)")

    # Compare
    matches = []
    mismatches = []
    bt_only = []
    live_only = []

    for m in overlap:
        b = bt[m]
        l = live[m]
        if b['outcome'] == l['outcome']:
            matches.append((m, b, l))
        else:
            mismatches.append((m, b, l))

    for m in bt_markets - live_markets:
        bt_only.append((m, bt[m]))
    for m in live_markets - bt_markets:
        live_only.append((m, live[m]))

    # Report
    print("\n" + "=" * 70)
    print("  COMPARISON: BACKTEST vs LIVE")
    print("=" * 70)

    print(f"\n  MATCHES (same outcome): {len(matches)}")
    for m, b, l in matches[:10]:
        print(f"    {m}: BT={b['outcome']} Live={l['outcome']} (qty BT={b['qty']} Live_buy={l['buy_qty']})")
    if len(matches) > 10:
        print(f"    ... and {len(matches)-10} more")

    print(f"\n  MISMATCHES (different outcome): {len(mismatches)}")
    for m, b, l in mismatches:
        print(f"    {m}")
        print(f"      Backtest: {b['outcome']} (side={b['side']}, qty={b['qty']}, pnl=${b['pnl']:.2f})")
        print(f"      Live:     {l['outcome']} (buy_qty={l['buy_qty']}, sell_qty={l['sell_qty']})")
        if b['outcome'] == 'TP' and l['outcome'] == 'settlement':
            print(f"      *** Backtest says TP but LIVE LOST (no sell fill) ***")
        elif b['outcome'] == 'settlement' and l['outcome'] == 'TP':
            print(f"      *** Backtest says settlement but LIVE HIT TP ***")

    if bt_only:
        print(f"\n  BACKTEST ONLY (not in live): {len(bt_only)} markets")
        for m, b in bt_only[:5]:
            print(f"    {m}: {b['outcome']}")
        if len(bt_only) > 5:
            print(f"    ... and {len(bt_only)-5} more")

    if live_only:
        print(f"\n  LIVE ONLY (not in backtest): {len(live_only)} markets")
        for m, l in live_only[:10]:
            print(f"    {m}: {l['outcome']} (buy={l['buy_qty']} sell={l['sell_qty']})")
        if len(live_only) > 10:
            print(f"    ... and {len(live_only)-10} more")

    # Summary stats (filter to bt_markets for --last mode)
    bt_subset = {m: bt[m] for m in bt_markets}
    bt_tp = sum(1 for b in bt_subset.values() if b['outcome'] == 'TP')
    live_overlap = {m: live[m] for m in overlap}
    live_tp_overlap = sum(1 for l in live_overlap.values() if l['outcome'] == 'TP')
    bt_filled = len(bt_subset)
    live_filled = sum(1 for l in live.values() if l['buy_qty'] > 0)
    print(f"\n  SUMMARY")
    print(f"    Backtest TP rate: {bt_tp}/{bt_filled} = {100*bt_tp/bt_filled:.1f}%" if bt_filled else "    N/A")
    print(f"    Live TP rate (all): {live_tp_overlap}/{len(live_overlap)} = {100*live_tp_overlap/len(live_overlap):.1f}%" if live_overlap else "    N/A")
    if overlap:
        match_pct = 100 * len(matches) / len(overlap)
        print(f"    Match rate (overlap): {len(matches)}/{len(overlap)} = {match_pct:.1f}%")
    print(f"    Critical: {len([x for x in mismatches if x[1]['outcome']=='TP' and x[2]['outcome']=='settlement'])} markets where BT=TP but Live=loss")
    print("=" * 70)


if __name__ == '__main__':
    main()
