#!/usr/bin/env python3
"""
Check how many limit orders (contracts) are sitting at 10¢ on KXBTC15M markets.
Uses Kalshi orderbook API to see if the 10¢ level is crowded (strategy may be well-known).

Run from repo root: python btc15m_backtest/scripts/check_10c_depth.py
"""
import os
import sys
import base64
import datetime
import time

# .../btc15m_backtest/scripts/this_file.py -> repo root = three dirname levels
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, '.env'), override=False)
except ImportError:
    pass

import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

API_KEY_ID = os.environ.get('KALSHI_API_KEY_ID', '')
_PRIVATE_KEY_PATH = os.environ.get('KALSHI_PRIVATE_KEY_PATH', 'kalshi_private_key')
_PRIVATE_KEY_PATH = (
    os.path.join(_PROJECT_ROOT, _PRIVATE_KEY_PATH)
    if not os.path.isabs(_PRIVATE_KEY_PATH)
    else _PRIVATE_KEY_PATH
)
BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'
SERIES_TICKER = 'KXBTC15M'
TARGET_PRICE = 0.10  # 10¢


def _load_private_key():
    pem_env = os.environ.get('KALSHI_PRIVATE_KEY_PEM')
    if pem_env:
        key_data = pem_env.encode() if isinstance(pem_env, str) else pem_env
        if key_data.strip():
            return serialization.load_pem_private_key(key_data, password=None, backend=default_backend())
    with open(_PRIVATE_KEY_PATH, 'rb') as f:
        key_data = f.read()
    return serialization.load_pem_private_key(key_data, password=None, backend=default_backend())


def _sign(private_key, timestamp: str, method: str, path: str) -> str:
    path_no_query = path.split('?')[0]
    message = f"{timestamp}{method}{path_no_query}".encode('utf-8')
    sig = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode('utf-8')


def _api_get(path: str, params=None):
    private_key = _load_private_key()
    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
    full_path = f"/trade-api/v2{path}" if not path.startswith('/trade-api') else path
    signature = _sign(private_key, timestamp, 'GET', full_path.split('?')[0])
    headers = {
        'KALSHI-ACCESS-KEY': API_KEY_ID,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp,
    }
    r = requests.get(BASE_URL + path, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _qty_at_price(levels, target_price_dollars):
    """Sum contract quantity at target price. levels can be yes_dollars/no_dollars or yes/no."""
    if not levels:
        return 0
    total = 0
    for level in levels:
        if not level or len(level) < 2:
            continue
        price_str = str(level[0])
        qty_str = str(level[1])
        try:
            price = float(price_str)
            qty = float(qty_str)
        except (ValueError, TypeError):
            continue
        is_10c = abs(price - 0.10) < 0.001 or abs(price - 10) < 0.01
        if is_10c:
            total += qty
    return int(total)


def main():
    print("=" * 60)
    print("10¢ ORDERBOOK DEPTH — KXBTC15M")
    print("(How many contracts are bidding at 10¢ — strategy crowding check)")
    print("=" * 60)

    try:
        resp = _api_get('/markets', params={'status': 'open', 'series_ticker': SERIES_TICKER, 'limit': 20})
        markets = resp.get('markets', [])
    except Exception as e:
        print(f"Error fetching markets: {e}")
        sys.exit(1)

    if not markets:
        print("No open KXBTC15M markets right now.")
        print("Run during :00–:06, :15–:21, :30–:36, or :45–:51 UTC for active markets.")
        return

    print(f"\nFound {len(markets)} open market(s). Fetching orderbooks...\n")

    total_yes_10c = 0
    total_no_10c = 0
    results = []

    for m in markets:
        ticker = m.get('ticker', '?')
        try:
            ob = _api_get(f'/markets/{ticker}/orderbook', params={'depth': 0})
            time.sleep(0.4)
        except Exception as e:
            print(f"  {ticker}: orderbook error — {e}")
            continue

        ob_data = ob.get('orderbook_fp') or ob.get('orderbook') or {}
        yes_levels = ob_data.get('yes_dollars') or ob_data.get('yes') or []
        no_levels = ob_data.get('no_dollars') or ob_data.get('no') or []

        yes_10c = _qty_at_price(yes_levels, TARGET_PRICE)
        no_10c = _qty_at_price(no_levels, TARGET_PRICE)

        total_yes_10c += yes_10c
        total_no_10c += no_10c
        results.append((ticker, yes_10c, no_10c))

    for ticker, y, n in results:
        print(f"  {ticker}")
        print(f"    YES @ 10¢: {y:,} contracts")
        print(f"    NO  @ 10¢: {n:,} contracts")
        print()

    print("-" * 60)
    print(f"  TOTAL at 10¢ across all open markets:")
    print(f"    YES: {total_yes_10c:,} contracts")
    print(f"    NO:  {total_no_10c:,} contracts")
    print(f"    Combined: {total_yes_10c + total_no_10c:,} contracts")
    print("-" * 60)

    combined = total_yes_10c + total_no_10c
    if combined > 5000:
        print("\n  HEAVY crowding at 10¢ — many bots/strategies likely at this level.")
        print("     Queue position matters; partial fills more likely.")
    elif combined > 2000:
        print("\n  Moderate crowding at 10¢. Expect some competition for fills.")
    elif combined > 500:
        print("\n  Light crowding. 10¢ level has some activity but not saturated.")
    else:
        print("\n  Low crowding. 10¢ level is relatively uncrowded.")

    print("=" * 60)


if __name__ == '__main__':
    main()
