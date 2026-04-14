#!/usr/bin/env python3
"""
Fetch last 100 markets from Kalshi API, verify last one closed at 9:15 EST,
and match them to your real live fills. Also compare backtest vs real trades.

Usage:
    python match_api_markets_to_live.py
"""
import os
import sys
import csv
import time
from collections import defaultdict
from datetime import datetime

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "btc15m_backtest"))
from data_fetcher import DataFetcher

SERIES = "KXBTC15M"
TRADES_CSV = os.path.join(_ROOT, "btc15m_backtest", "trades_KXBTC15M.csv")

# 9:15 AM EST = 14:15 UTC (EST = UTC-5)
EXPECTED_LAST_CLOSE_UTC = (14, 15)  # hour, minute


def _close_ts(m):
    """Extract close_time as Unix timestamp."""
    ct = m.get("close_time")
    if ct is None:
        return 0
    if isinstance(ct, (int, float)):
        ts = int(ct) if ct < 1e12 else int(ct) // 1000
        return ts
    try:
        from datetime import datetime as dt
        return int(dt.fromisoformat(str(ct).replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0


def fetch_last_n_markets(fetcher, n=100):
    """Fetch last N settled markets from API (refresh). Return list oldest-first."""
    print(f"  Refreshing markets from API for {SERIES}...")
    cursor = None
    fetched = 0
    for _ in range(50):
        params = {"status": "settled", "limit": 500, "series_ticker": SERIES}
        if cursor:
            params["cursor"] = cursor
        resp = fetcher._api_get("/markets", params=params)
        batch = resp.get("markets", [])
        for m in batch:
            fetcher._cache_market(m)
        fetched += len(batch)
        cursor = resp.get("cursor")
        if not batch or not cursor:
            break
        time.sleep(0.4)
    fetcher._conn.commit()

    cached = fetcher._get_cached_markets()
    all_markets = list(cached.values())
    all_markets.sort(key=_close_ts, reverse=True)  # most recent first
    last_n = all_markets[:n]
    last_n.reverse()  # oldest first for chronological order
    print(f"  Fetched {fetched} markets. Using last {len(last_n)} (most recent).")
    return last_n


def parse_close_time_from_ticker(ticker):
    """Parse close time from ticker like KXBTC15M-26MAR111415-15. Returns (hour_utc, min_utc) or None."""
    # Format: KXBTC15M-YYMMMDDHHMM-M  e.g. 26MAR112115 = 2026 Mar 11, 21:15 UTC (11 chars)
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    date_part = parts[1]  # e.g. 26MAR112115
    if len(date_part) < 11:
        return None
    try:
        # YY(2) MMM(3) DD(2) HH(2) MM(2) -> indices 7:9=HH, 9:11=MM
        hh = int(date_part[7:9])
        mm = int(date_part[9:11])
        return (hh, mm)
    except (ValueError, IndexError):
        return None


def fetch_live_fills(fetcher):
    """Fetch all KXBTC15M fills from Kalshi API."""
    fills = []
    cursor = None
    for _ in range(100):
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = fetcher._api_get("/portfolio/fills", params=params)
        batch = resp.get("fills", [])
        fills.extend([
            f for f in batch
            if (f.get("ticker") or f.get("market_ticker") or "").startswith(SERIES)
        ])
        cursor = resp.get("cursor")
        if not batch or not cursor:
            break
        time.sleep(0.4)
    return fills


def load_backtest_trades():
    """Parse backtest CSV. Return dict: market -> {side, outcome, qty, pnl}."""
    if not os.path.exists(TRADES_CSV):
        return {}
    bt = {}
    with open(TRADES_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            m = row.get("market", "").strip()
            if not m or not m.startswith(SERIES):
                continue
            action = row.get("action", "")
            side = row.get("side", "")
            qty = int(float(row.get("qty", 0) or 0))
            pnl = float(row.get("pnl", 0) or 0)
            if m not in bt:
                bt[m] = {"side": side, "outcome": None, "qty": 0, "pnl": 0}
            bt[m]["qty"] = max(bt[m]["qty"], qty)
            if action == "buy_fill":
                bt[m]["side"] = side
            elif action == "sell_fill":
                bt[m]["outcome"] = "TP"
                bt[m]["pnl"] = pnl
            elif action == "settlement":
                bt[m]["outcome"] = "settlement"
                bt[m]["pnl"] = pnl
    for m, d in bt.items():
        if d["outcome"] is None:
            d["outcome"] = "settlement"
    return bt


def build_live_by_market(fills):
    """Aggregate fills by market. Return dict: market -> {side, outcome, buy_qty, sell_qty}."""
    live = defaultdict(lambda: {"side": "", "buy_qty": 0, "sell_qty": 0})
    for f in fills:
        t = f.get("ticker") or f.get("market_ticker", "")
        if not t.startswith(SERIES):
            continue
        action = (f.get("action") or "").lower()
        side = (f.get("side") or "").lower()
        count = int(float(f.get("count", 0) or f.get("count_fp", 0)))
        if action == "buy":
            live[t]["side"] = side
            live[t]["buy_qty"] += count
        elif action == "sell":
            live[t]["sell_qty"] += count
    for t, d in live.items():
        d["outcome"] = "TP" if d["sell_qty"] > 0 else "settlement"
    return dict(live)


def main():
    print("=" * 70)
    print("  MATCH: Last 100 API Markets vs Your Real Trades")
    print("=" * 70)

    fetcher = DataFetcher(SERIES)

    # 1. Fetch last 100 markets
    print("\n1. Fetching last 100 markets from Kalshi API...")
    api_markets = fetch_last_n_markets(fetcher, n=100)
    api_tickers = [m.get("ticker", "") for m in api_markets if m.get("ticker")]

    if not api_tickers:
        print("  ERROR: No markets returned.")
        fetcher.close()
        sys.exit(1)

    last_ticker = api_tickers[-1]
    close_time = parse_close_time_from_ticker(last_ticker)
    if close_time:
        h, m = close_time
        expected_h, expected_m = EXPECTED_LAST_CLOSE_UTC
        if (h, m) == (expected_h, expected_m):
            print(f"  OK: Last market {last_ticker} closed at {h:02d}:{m:02d} UTC (9:15 EST)")
        else:
            print(f"  NOTE: Last market {last_ticker} closed at {h:02d}:{m:02d} UTC (expected 14:15 UTC = 9:15 EST)")

    # 2. Load backtest trades
    print("\n2. Loading backtest trades...")
    backtest = load_backtest_trades()
    bt_in_api = {t for t in api_tickers if t in backtest}
    print(f"  Backtest markets in last 100: {len(bt_in_api)}")

    # 3. Fetch live fills (real trades)
    print("\n3. Fetching your real trades (live fills) from Kalshi API...")
    fills = fetch_live_fills(fetcher)
    live = build_live_by_market(fills)
    live_filled = {t for t, d in live.items() if d["buy_qty"] > 0}
    fetcher.close()

    print(f"  API markets (last 100): {len(api_tickers)}")
    print(f"  Real trades (live):     {len(live_filled)}")

    # 4. Match API vs real trades
    api_set = set(api_tickers)
    overlap = api_set & live_filled
    api_only = api_set - live_filled
    live_only = live_filled - api_set

    print("\n" + "=" * 70)
    print("  MATCH RESULTS")
    print("=" * 70)

    print(f"\n  Markets in API (last 100):     {len(api_set)}")
    print(f"  Markets you filled (live):    {len(live_filled)}")
    print(f"  Overlap (in BOTH):             {len(overlap)}")
    print(f"  In API but NOT filled:        {len(api_only)}")
    print(f"  Filled but NOT in last 100:   {len(live_only)}")

    if overlap:
        match_pct = 100 * len(overlap) / len(api_set)
        print(f"\n  Of the last 100 markets, you filled {len(overlap)} ({match_pct:.1f}%)")

    # Outcome breakdown for overlap (real trades)
    if overlap:
        tp_count = sum(1 for t in overlap if live[t]["outcome"] == "TP")
        settle_count = len(overlap) - tp_count
        print(f"\n  Real trades outcome (overlap):")
        print(f"    TP (sold at 33c):     {tp_count}")
        print(f"    Settlement:           {settle_count}")
        if overlap:
            print(f"    TP rate:              {100 * tp_count / len(overlap):.1f}%")

    # Backtest vs real trades (for overlapping markets)
    bt_vs_live_overlap = overlap & set(backtest.keys())
    if bt_vs_live_overlap:
        matches = sum(1 for t in bt_vs_live_overlap if backtest[t]["outcome"] == live[t]["outcome"])
        mismatches = [(t, backtest[t], live[t]) for t in bt_vs_live_overlap if backtest[t]["outcome"] != live[t]["outcome"]]
        match_pct = 100 * matches / len(bt_vs_live_overlap)
        print(f"\n  BACKTEST vs REAL TRADES (last 100 overlap):")
        print(f"    Markets in both:       {len(bt_vs_live_overlap)}")
        print(f"    Match (same outcome):  {matches}/{len(bt_vs_live_overlap)} = {match_pct:.1f}%")
        if mismatches:
            print(f"    Mismatches:            {len(mismatches)}")
            for t, bt, lv in mismatches[:5]:
                print(f"      {t}")
                print(f"        Backtest: {bt['outcome']} (qty={bt['qty']})")
                print(f"        Real:     {lv['outcome']} (buy={lv['buy_qty']} sell={lv['sell_qty']})")
            if len(mismatches) > 5:
                print(f"      ... and {len(mismatches) - 5} more")

    # List first/last few
    print(f"\n  Last 5 API markets (most recent):")
    for t in api_tickers[-5:]:
        filled = "FILLED" if t in overlap else "no fill"
        outcome = live[t]["outcome"] if t in overlap else "-"
        print(f"    {t}  {filled}  {outcome}")

    if live_only:
        print(f"\n  Filled but outside last 100 (older): {len(live_only)} markets")
        for t in sorted(live_only)[-5:]:
            print(f"    {t}  {live[t]['outcome']} (buy={live[t]['buy_qty']} sell={live[t]['sell_qty']})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
