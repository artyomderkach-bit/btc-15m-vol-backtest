#!/usr/bin/env python3
"""Unified bot status: account, positions, resting orders, fill rates, P&L, and trade history."""

import os
import sys
import base64
import datetime
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore', module='urllib3')

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

API_KEY_ID = os.environ.get('KALSHI_API_KEY_ID', 'YOUR_API_KEY_ID')
PRIVATE_KEY_PATH = os.environ.get('KALSHI_PRIVATE_KEY_PATH', 'kalshi_private_key')
_script_dir = os.path.dirname(os.path.abspath(__file__))
PRIVATE_KEY_PATH = os.path.join(_script_dir, PRIVATE_KEY_PATH) if not os.path.isabs(PRIVATE_KEY_PATH) else PRIVATE_KEY_PATH
BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'
SERIES_TICKER = 'KXBTC15M'


def _load_private_key():
    pem_env = os.environ.get('KALSHI_PRIVATE_KEY_PEM')
    if pem_env:
        key_data = pem_env.encode() if isinstance(pem_env, str) else pem_env
        if key_data.strip():
            return serialization.load_pem_private_key(key_data, password=None, backend=default_backend())
    if not os.path.exists(PRIVATE_KEY_PATH):
        print("ERROR: Private key not found:", PRIVATE_KEY_PATH, file=sys.stderr)
        sys.exit(1)
    with open(PRIVATE_KEY_PATH, 'rb') as f:
        key_data = f.read()
    if not key_data.strip():
        print("ERROR: Private key file is empty", file=sys.stderr)
        sys.exit(1)
    return serialization.load_pem_private_key(key_data, password=None, backend=default_backend())


def _sign(private_key, timestamp, method, path):
    path_no_query = path.split('?')[0]
    message = f"{timestamp}{method}{path_no_query}".encode('utf-8')
    sig = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode('utf-8')


def _api_get(path, params=None):
    private_key = _load_private_key()
    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
    full_path = f"/trade-api/v2{path}" if not path.startswith('/trade-api') else path
    sign_path = full_path.split('?')[0]
    signature = _sign(private_key, timestamp, 'GET', sign_path)
    headers = {
        'KALSHI-ACCESS-KEY': API_KEY_ID,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp,
    }
    r = requests.get(BASE_URL + path, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _print_account_status():
    bal_resp = _api_get('/portfolio/balance')
    cash_cents = bal_resp.get('balance', 0)
    cash_dol = cash_cents / 100.0

    port_resp = _api_get('/portfolio/positions', params={'limit': 100})
    positions = port_resp.get('market_positions', [])
    all_btc_positions = [p for p in positions if (p.get('ticker') or '').startswith(SERIES_TICKER)]
    btc_positions = [p for p in all_btc_positions if int(p.get('position', 0) or 0) != 0]
    port_value_cents = sum(int(p.get('market_exposure', 0) or 0) for p in all_btc_positions)
    port_dol = port_value_cents / 100.0
    total_cents = cash_cents + port_value_cents
    total_dol = total_cents / 100.0

    print("=" * 70)
    print("BOT STATUS -- KXBTC15M")
    print("=" * 70)
    print("\nACCOUNT")
    print(f"  Cash:              ${cash_dol:.2f}")
    print(f"  Positions (KXBTC): ${port_dol:.2f}")
    print(f"  Total:             ${total_dol:.2f}")

    print("\nPOSITIONS")
    if not btc_positions:
        print("  (none)")
    else:
        for p in btc_positions:
            ticker = p.get('ticker', '?')
            pos_val = int(p.get('position', 0))
            cost_cents = int(p.get('market_exposure', 0) or 0)
            cost = cost_cents / 100.0
            rpnl = int(p.get('realized_pnl', 0) or 0) / 100.0
            side = "YES" if pos_val > 0 else "NO"
            qty = abs(pos_val)
            print(f"  {ticker}: {side} {qty} contracts | cost ${cost:.2f} | realized ${rpnl:+.2f}")

    print("\nRESTING ORDERS")
    orders_resp = _api_get('/portfolio/orders', params={'status': 'resting', 'limit': 50})
    resting = [o for o in orders_resp.get('orders', []) if (o.get('ticker') or '').startswith(SERIES_TICKER)]
    if not resting:
        print("  (none)")
    else:
        for o in resting:
            ticker = o.get('ticker', '?')
            side = (o.get('side') or '?').upper()
            action = (o.get('action') or '?').lower()
            remaining = o.get('remaining_count', 0)
            yes_price = o.get('yes_price')
            no_price = o.get('no_price')
            price_c = yes_price if side == 'YES' else no_price
            price_str = f"{price_c}c" if price_c is not None else "?"
            print(f"  {ticker}: {side} {action} {remaining} @ {price_str}")
    print()


def _fetch_all_fills(min_ts=None, max_ts=None):
    all_fills = []
    cursor = None
    params = {'limit': 200}
    if min_ts is not None:
        params['min_ts'] = min_ts
    if max_ts is not None:
        params['max_ts'] = max_ts
    while True:
        if cursor:
            params['cursor'] = cursor
        resp = _api_get('/portfolio/fills', params=params)
        fills = resp.get('fills', [])
        all_fills.extend(fills)
        cursor = resp.get('cursor')
        if not cursor or not fills:
            break
    return all_fills


def _fetch_all_orders(min_ts=None):
    all_orders = []
    n_portfolio = 0
    n_historical = 0
    for status in ('executed', 'canceled'):
        cursor = None
        while True:
            params = {'limit': 200, 'status': status}
            if cursor:
                params['cursor'] = cursor
            try:
                resp = _api_get('/portfolio/orders', params=params)
            except Exception:
                break
            orders = resp.get('orders', [])
            all_orders.extend(orders)
            n_portfolio += len(orders)
            cursor = resp.get('cursor')
            if not cursor or not orders:
                break
    cursor = None
    while True:
        params = {'limit': 200}
        if cursor:
            params['cursor'] = cursor
        try:
            resp = _api_get('/historical/orders', params=params)
        except Exception:
            break
        orders = resp.get('orders', [])
        all_orders.extend(orders)
        n_historical += len(orders)
        cursor = resp.get('cursor')
        if not cursor or not orders:
            break
    return all_orders, n_portfolio, n_historical


def _order_ts(o):
    for key in ('created_time', 'last_update_time'):
        val = o.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return int(val) // 1000 if val > 1e12 else int(val)
        try:
            s = str(val).replace('Z', '+00:00')
            if '.' in s:
                idx = s.find('+') if '+' in s else s.find('-', 1)
                if idx > 0:
                    base, frac = s[:idx].split('.', 1)
                    s = base + '.' + frac[:3] + s[idx:]
            return int(datetime.datetime.fromisoformat(s).timestamp())
        except Exception:
            pass
    return 0


def _fill_price_cents(f):
    side = (f.get('side') or '?').upper()
    yes_price = f.get('yes_price')
    no_price = f.get('no_price')
    price_c = yes_price if side == 'YES' else no_price
    if price_c is None:
        price_c = (100 - yes_price) if yes_price is not None else f.get('price')
    price_c = price_c if price_c is not None else 0
    if (f.get('action') or '').lower() == 'sell' and price_c > 50:
        price_c = 100 - price_c
    return price_c


def main():
    _print_account_status()

    print("Fetching fills and orders...")
    fills = _fetch_all_fills()
    btc_fills = [f for f in fills if (f.get('ticker') or f.get('market_ticker') or '').startswith(SERIES_TICKER)]

    orders, n_portfolio, n_historical = _fetch_all_orders()
    btc_orders = [o for o in orders if (o.get('ticker') or '').startswith(SERIES_TICKER)]

    buy_10c_orders = [o for o in btc_orders if (o.get('action') or '').lower() == 'buy'
                      and (o.get('yes_price') == 10 or o.get('no_price') == 10)]
    sell_33c_orders = [o for o in btc_orders if (o.get('action') or '').lower() == 'sell'
                      and (o.get('yes_price') == 33 or o.get('no_price') == 33)]

    markets_attempted_10c = set(o.get('ticker') for o in buy_10c_orders if o.get('ticker'))
    markets_filled_10c = set(o.get('ticker') for o in buy_10c_orders if o.get('ticker') and (o.get('fill_count') or 0) > 0)
    markets_placed_33c = set(o.get('ticker') for o in sell_33c_orders if o.get('ticker'))
    markets_filled_33c = set(o.get('ticker') for o in sell_33c_orders if o.get('ticker') and (o.get('fill_count') or 0) > 0)

    markets_bought = set(f.get('ticker') or f.get('market_ticker') for f in btc_fills
                        if (f.get('action') or '').lower() == 'buy')
    markets_sold = set(f.get('ticker') or f.get('market_ticker') for f in btc_fills
                      if (f.get('action') or '').lower() == 'sell')

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: SELL ORDER PLACEMENT")
    print(f"{'='*70}")
    print(f"  10c buy orders placed:     {len(markets_attempted_10c)} markets")
    print(f"  10c buy orders FILLED:     {len(markets_filled_10c)} markets")
    print(f"  33c sell orders PLACED:    {len(markets_placed_33c)} markets")
    print(f"  33c sell orders FILLED:    {len(markets_filled_33c)} markets")
    print()

    if markets_filled_10c:
        gap = markets_filled_10c - markets_placed_33c
        print(f"  Markets with 10c fill but NO 33c sell order: {len(gap)}/{len(markets_filled_10c)}")
        if gap:
            print(f"  *** BOT IS NOT PLACING SELL ORDERS FOR THESE MARKETS ***")
            for t in sorted(gap)[:20]:
                print(f"    {t}")
            if len(gap) > 20:
                print(f"    ... and {len(gap)-20} more")
        else:
            print(f"  All markets with 10c fill have a 33c sell order placed. Good.")

    if markets_placed_33c:
        not_filled = markets_placed_33c - markets_filled_33c
        print(f"\n  33c sell placed but NOT filled: {len(not_filled)}/{len(markets_placed_33c)}")
        print(f"  33c sell placed AND filled:     {len(markets_filled_33c)}/{len(markets_placed_33c)}")

    # Check sell order details
    print(f"\n{'='*70}")
    print(f"  SELL ORDER DETAILS (checking for YES/NO issues)")
    print(f"{'='*70}")
    for o in sell_33c_orders[:15]:
        ticker = o.get('ticker', '?')
        side = (o.get('side') or '?').upper()
        action = (o.get('action') or '?').lower()
        yes_p = o.get('yes_price')
        no_p = o.get('no_price')
        fill_count = o.get('fill_count', 0)
        remaining = o.get('remaining_count', 0)
        status = o.get('status', '?')
        print(f"  {ticker}  {side} {action}  yes={yes_p}c no={no_p}c  filled={fill_count} remaining={remaining}  status={status}")

    # Check which side the buy was on for these same markets
    print(f"\n  CROSS-CHECK: buy side vs sell side")
    buy_side_by_market = {}
    for o in buy_10c_orders:
        t = o.get('ticker')
        if t:
            buy_side_by_market[t] = (o.get('side') or '?').upper()

    mismatches = 0
    for o in sell_33c_orders[:15]:
        ticker = o.get('ticker', '?')
        sell_side = (o.get('side') or '?').upper()
        buy_side = buy_side_by_market.get(ticker, '?')
        match = 'OK' if buy_side == sell_side else 'MISMATCH'
        if buy_side != sell_side and buy_side != '?':
            mismatches += 1
        print(f"  {ticker}: buy={buy_side} sell={sell_side} {match}")

    if mismatches:
        print(f"\n  *** {mismatches} SELL ORDERS ON WRONG SIDE ***")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
