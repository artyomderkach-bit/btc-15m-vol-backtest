#!/usr/bin/env python3
"""
Run the NYCClimatologyFadeStrategy backtest.

Uses the existing DataFetcher and candle parser but implements a custom market
loop to handle weather market durations (not hardcoded to 15-minute windows).

The Engine's run() is BTC-specific (open_ts = close_ts - 15*60).  Weather
markets have longer trading windows, so this runner computes open_ts from
the market's actual open_time field and feeds candles for the full duration.

Usage (from btc15m_backtest):
    python future_exploration/run_backtest_nyc_climate.py                                # defaults
    python future_exploration/run_backtest_nyc_climate.py KXHIGHNY 500                   # 500 markets
    python future_exploration/run_backtest_nyc_climate.py KXHIGHNY 500 0.10 1000 0.08    # custom edge
"""
import sys
import os
import re
import math
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict

_FE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_FE)
sys.path[:0] = [_ROOT, _FE]

from data_fetcher import DataFetcher
from nyc_climatology_fade_strategy import NYCClimatologyFadeStrategy
from weather_candles import parse_candle_weather

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


# ── CLI arguments ──
# Usage: SERIES NUM_MARKETS VOL_FILL_PCT BANKROLL MIN_EDGE TP_FILL_RATE REFRESH
SERIES_TICKER = sys.argv[1] if len(sys.argv) > 1 else 'KXHIGHNY'
NUM_MARKETS   = _parse_int(sys.argv[2], 500) if len(sys.argv) > 2 else 500
VOL_FILL_PCT  = _parse_float(sys.argv[3], 0.10) if len(sys.argv) > 3 else 0.10
BANKROLL      = _parse_float(sys.argv[4], 1000.0) if len(sys.argv) > 4 else 1000.0
MIN_EDGE      = _parse_float(sys.argv[5], 0.08) if len(sys.argv) > 5 else 0.08
TP_FILL_RATE  = _parse_float(sys.argv[6], 1.0) if len(sys.argv) > 6 else 1.0
REFRESH       = _parse_int(sys.argv[7], 0) if len(sys.argv) > 7 else 0

strategy = NYCClimatologyFadeStrategy(
    bankroll=BANKROLL,
    risk_pct=0.02,
    min_edge=MIN_EDGE,
    entry_spread=0.02,
    tp_profit=0.20,
    volume_fill_pct=VOL_FILL_PCT,
    tp_fill_rate=TP_FILL_RATE,
)

print(f"Strategy: NYCClimatologyFadeStrategy (nyc_climatology_fade)")
print(f"  Series: {SERIES_TICKER}")
print(f"  Min edge: ±{MIN_EDGE:.2f}  |  Entry spread: 0.02  |  TP profit: 0.20")
print(f"  Markets: {NUM_MARKETS}  |  Vol fill: {VOL_FILL_PCT*100:.0f}%"
      f"  |  Bankroll: ${BANKROLL:.0f}")
print()


# ── Helpers ──

def _extract_ts(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val) if val < 1e12 else int(val) // 1000
    try:
        return int(pd.Timestamp(str(val)).timestamp())
    except Exception:
        return None


def _extract_threshold_from_market(market):
    """Extract temperature threshold from market data.

    Uses floor_strike (for greater-than markets) or cap_strike (for less-than)
    from the API response directly — more reliable than regex parsing.
    Falls back to regex on title/ticker if those fields are absent (older markets).
    """
    strike_type = market.get('strike_type', '')

    # Primary: use structured fields from API response
    if strike_type == 'greater':
        v = market.get('floor_strike')
        if v is not None:
            try:
                val = int(float(v))
                if -20 <= val <= 130:
                    return val
            except (ValueError, TypeError):
                pass
    elif strike_type == 'less':
        v = market.get('cap_strike')
        if v is not None:
            try:
                val = int(float(v))
                if -20 <= val <= 130:
                    return val
            except (ValueError, TypeError):
                pass

    # Fallback: regex on title / ticker for older cached markets
    title = market.get('title', '') or ''
    subtitle = market.get('subtitle', '') or ''
    text = f"{title} {subtitle}"

    m = re.search(r'[>]\s*=?\s*(\d{1,3})\s*[°]?', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'[<]\s*=?\s*(\d{1,3})\s*[°]?', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'above\s+(\d{1,3})', text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    ticker = market.get('ticker', '')
    m = re.search(r'[-_]T(\d{1,3})(?=[^:\d]|$)', ticker)
    if m:
        val = int(m.group(1))
        if -20 <= val <= 130:
            return val
    return None


def _is_tradeable_market(market):
    """Only trade simple greater-than or less-than markets. Skip between-range (B prefix)."""
    strike_type = market.get('strike_type', '')
    if strike_type == 'between':
        return False
    # Also filter by ticker: skip B-prefix markets that lack a strike_type field
    ticker = market.get('ticker', '')
    if re.search(r'[-_]B\d', ticker):
        return False
    return True


def _is_nyc_market(market):
    ticker = market.get('ticker', '') or ''
    # Direct ticker match — KXHIGHNY-* and legacy HIGHNY-* are both NYC high temp
    if re.search(r'\bKXHIGHNY\b|\bHIGHNY\b', ticker, re.IGNORECASE):
        return True
    text = f"{market.get('title', '')} {market.get('subtitle', '')}"
    return bool(re.search(r'\bNYC\b|\bNew\s*York\b', text, re.IGNORECASE))


# ── Fetch markets ──

fetcher = DataFetcher(SERIES_TICKER)
markets = fetcher.fetch_markets(limit=NUM_MARKETS, refresh=(REFRESH == 1))

markets = sorted(markets, key=lambda m: _extract_ts(m.get('close_time')) or 0)

total = len(markets)
traded = 0
filled_markets = 0
tp_markets = 0
nyc_count = 0
threshold_parsed = 0

hour_stats = defaultdict(lambda: {
    'traded': 0, 'filled': 0, 'tp': 0, 'settled': 0, 'pnl': 0.0,
})
day_stats = defaultdict(lambda: {
    'traded': 0, 'filled': 0, 'tp': 0, 'settled': 0, 'pnl': 0.0,
})

for i, market in enumerate(markets):
    if strategy.halted:
        print(f"  Halted at market {i+1}/{total}")
        break

    ticker = market.get('ticker', '')
    result = market.get('result', '')
    close_time = market.get('close_time')

    if close_time is None or result not in ('yes', 'no'):
        continue

    if not _is_nyc_market(market):
        continue
    if not _is_tradeable_market(market):
        continue
    nyc_count += 1

    threshold_temp = _extract_threshold_from_market(market)
    strike_type    = market.get('strike_type', 'greater')
    if threshold_temp is not None:
        threshold_parsed += 1
        strategy.inject_market_info(ticker, threshold_temp, strike_type)

    close_ts = _extract_ts(close_time)
    if not close_ts:
        continue

    open_time = market.get('open_time')
    open_ts = _extract_ts(open_time)
    if not open_ts:
        open_ts = close_ts - 24 * 3600

    dt = datetime.datetime.utcfromtimestamp(open_ts)
    hour_utc = dt.hour
    day_utc = dt.weekday()
    if (i + 1) % 50 == 0 or i == 0:
        thr_str = f"  threshold={threshold_temp}°F" if threshold_temp else ""
        print(f"  Processing {i+1}/{total}: {ticker}{thr_str}"
              f"  (bankroll: ${strategy.bankroll:.2f})")

    try:
        raw_candles = fetcher.fetch_candles(ticker, open_ts, close_ts)
    except Exception as e:
        print(f"  Error fetching candles for {ticker}: {e}")
        continue

    bars = []
    for rc in raw_candles:
        bar = parse_candle_weather(rc)
        if bar and open_ts < bar.ts <= close_ts:
            bars.append(bar)
    bars.sort(key=lambda b: b.ts)

    if len(bars) < 2:
        continue

    traded += 1
    pre_log_start = len(strategy.trade_log)

    strategy.on_market_open(ticker, open_ts)

    had_fill = False
    had_tp = False

    for bar in bars:
        pre_log_len = len(strategy.trade_log)
        strategy.on_candle(ticker, bar, open_ts, close_ts)
        for entry in strategy.trade_log[pre_log_len:]:
            if entry.get('action') == 'buy_fill':
                had_fill = True
            if entry.get('action') == 'sell_fill':
                had_tp = True

    pre_settle_len = len(strategy.trade_log)
    strategy.on_market_settle(ticker, result)
    held_to_settle = False
    for entry in strategy.trade_log[pre_settle_len:]:
        if entry.get('action') == 'settlement':
            held_to_settle = True

    market_pnl = sum(
        e.get('pnl', 0) for e in strategy.trade_log[pre_log_start:]
        if e.get('action') in ('sell_fill', 'settlement')
    )

    hs = hour_stats[hour_utc]
    ds = day_stats[day_utc]
    hs['traded'] += 1
    ds['traded'] += 1
    if had_fill:
        filled_markets += 1
        hs['filled'] += 1
        ds['filled'] += 1
    if had_tp:
        tp_markets += 1
        hs['tp'] += 1
        ds['tp'] += 1
    if held_to_settle:
        hs['settled'] += 1
        ds['settled'] += 1
    hs['pnl'] += market_pnl
    ds['pnl'] += market_pnl

fetcher.close()


# ── Report ──

df = pd.DataFrame(strategy.trade_log) if strategy.trade_log else pd.DataFrame()
net_pnl = strategy.bankroll - strategy.initial_bankroll
total_return = (net_pnl / strategy.initial_bankroll) * 100
dd_pct = (strategy.max_drawdown / strategy.peak_bankroll * 100
          if strategy.peak_bankroll else 0)

avg_win = avg_loss = 0
if not df.empty and 'pnl' in df.columns:
    exits = df[df['action'].isin(['sell_fill', 'settlement'])]
    wins = exits[exits['pnl'] > 0]
    losses = exits[exits['pnl'] < 0]
    avg_win = wins['pnl'].mean() if len(wins) else 0
    avg_loss = losses['pnl'].mean() if len(losses) else 0

sharpe = 0
if not df.empty and 'bankroll' in df.columns:
    df_s = df.copy()
    df_s['bankroll'] = pd.to_numeric(df_s['bankroll'], errors='coerce')
    equity = np.concatenate([
        [strategy.initial_bankroll],
        df_s.dropna(subset=['bankroll'])['bankroll'].values,
    ])
    ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], np.nan)
    ret = ret[~np.isnan(ret) & np.isfinite(ret)]
    if len(ret) > 1 and np.std(ret) > 0:
        sharpe = (np.mean(ret) / np.std(ret)) * math.sqrt(252)

print()
print("=" * 70)
print("  NYC CLIMATOLOGY FADE — BACKTEST REPORT")
print("=" * 70)
print()
print("  MARKET FILTERING")
print(f"    Total markets in series:  {total}")
print(f"    NYC simple (T) markets:   {nyc_count}  (B/between skipped)")
print(f"    Thresholds parsed:        {threshold_parsed}")
print(f"    Markets with candle data: {traded}")
print()
print("  SIGNAL GENERATION")
print(f"    Signals fired (edge ok):  {strategy.signals_fired}")
print(f"    No edge (inside min):     {strategy.no_edge_count}")
print(f"    No threshold parsed:      {strategy.markets_no_threshold}")
print()
print("  FILLS & EXITS")
print(f"    Entry fills:              {strategy.entry_fills}")
print(f"    TP fills:                 {strategy.tp_fills}")
print(f"    Settlement wins:          {strategy.settlement_wins}")
print(f"    Settlement losses:        {strategy.settlement_losses}")
print()
print("  PERFORMANCE")
print(f"    Initial Bankroll:         ${strategy.initial_bankroll:,.2f}")
print(f"    Final Bankroll:           ${strategy.bankroll:,.2f}")
print(f"    Net P/L:                  ${net_pnl:+,.2f}")
print(f"    Total Return:             {total_return:+.2f}%")
print(f"    Max Drawdown:             ${strategy.max_drawdown:,.2f}  ({dd_pct:.1f}%)")
print(f"    Max Consecutive Losses:   {strategy.max_consecutive_losses}")
print(f"    Sharpe Ratio (ann.):      {sharpe:.2f}")
print()
print("  AVERAGES")
print(f"    Avg profit (winners):     ${avg_win:+,.2f}")
print(f"    Avg loss (losers):        ${avg_loss:+,.2f}")
if strategy.entry_fills > 0 and not df.empty:
    exit_pnl = df[df['action'].isin(['sell_fill', 'settlement'])]['pnl'].sum()
    avg_pnl = exit_pnl / strategy.entry_fills
    print(f"    Avg P&L per trade:        ${avg_pnl:+,.2f}")
    print(f"    Total P&L:                ${exit_pnl:+,.2f}")
print("=" * 70)

if hour_stats:
    print()
    print("=" * 70)
    print("  PERFORMANCE BY HOUR (UTC)")
    print("=" * 70)
    print(f"  {'Hour':>6}  {'Traded':>7}  {'Filled':>7}  {'TP':>5}"
          f"  {'Settle':>7}  {'PnL':>12}")
    print(f"  {'----':>6}  {'------':>7}  {'------':>7}  {'--':>5}"
          f"  {'------':>7}  {'---':>12}")
    for h in range(24):
        hs = hour_stats.get(h)
        if not hs or hs['traded'] == 0:
            continue
        print(f"  {h:02d}:00   {hs['traded']:>7}  {hs['filled']:>7}"
              f"  {hs['tp']:>5}  {hs['settled']:>7}"
              f"  ${hs['pnl']:>+10,.2f}")
    print("=" * 70)

if day_stats:
    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print()
    print("=" * 70)
    print("  PERFORMANCE BY DAY OF WEEK (UTC)")
    print("=" * 70)
    print(f"  {'Day':>6}  {'Traded':>7}  {'Filled':>7}  {'TP':>5}"
          f"  {'Settle':>7}  {'PnL':>12}")
    print(f"  {'---':>6}  {'------':>7}  {'------':>7}  {'--':>5}"
          f"  {'------':>7}  {'---':>12}")
    for d in range(7):
        ds = day_stats.get(d)
        if not ds or ds['traded'] == 0:
            continue
        print(f"  {DAY_NAMES[d]:>6}  {ds['traded']:>7}  {ds['filled']:>7}"
              f"  {ds['tp']:>5}  {ds['settled']:>7}"
              f"  ${ds['pnl']:>+10,.2f}")
    print("=" * 70)

# ── Save trade log ──

if not df.empty:
    csv_path = os.path.join(_dir, f"trades_{SERIES_TICKER}_nyc_climate.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTrade log saved to {csv_path}")
else:
    print("\nNo trades generated.")
