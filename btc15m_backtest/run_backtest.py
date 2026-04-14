#!/usr/bin/env python3
"""
Run the Kalshi prediction market backtest.

Usage (use ./run.sh or .venv/bin/python - 'python' may not be in PATH):
    ./run.sh                                    # BTC 15-min, 1000 markets, 10% fill, $1000
    ./run.sh KXBTC15M 7000 0.10 300             # 7000 markets, $300 start
    ./run.sh KXBTC15M 1000 0.10 1000 0.85      # Conservative: 85% TP fill rate (queue model)
    .venv/bin/python run_backtest.py KXBTC15M 7000 0.10 300 0.85

Args: [SERIES] [MARKETS] [VOL_FILL_PCT] [BANKROLL] [TP_FILL_RATE] [REFRESH] [OUTPUT]
  TP_FILL_RATE: When price touches the TP limit (default 40¢), assume fill this fraction (0-1).
    Default 1.0 = optimistic. 0.85 = conservative (models queue position).
  REFRESH: 1 = fetch fresh markets from API (most recent). 0 = use cache.
  OUTPUT: "recent" = write to trades_*_recent.csv (for small backtests, leaves main CSV untouched).
"""
import sys
import re
import random
from engine import Engine

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

SERIES_TICKER = sys.argv[1] if len(sys.argv) > 1 else 'KXBTC15M'
NUM_MARKETS = _parse_int(sys.argv[2], 1000) if len(sys.argv) > 2 else 1000
VOLUME_FILL_PCT = _parse_float(sys.argv[3], 0.10) if len(sys.argv) > 3 else 0.10
BANKROLL = _parse_float(sys.argv[4], 1000.0) if len(sys.argv) > 4 else 1000.0
TP_FILL_RATE = _parse_float(sys.argv[5], 1.0) if len(sys.argv) > 5 else 1.0
REFRESH = _parse_int(sys.argv[6], 0) if len(sys.argv) > 6 else 0
OUTPUT_RECENT = (sys.argv[7] == 'recent') if len(sys.argv) > 7 else False

if TP_FILL_RATE < 1.0:
    random.seed(42)  # Reproducible when modeling queue position

engine = Engine(
    series_ticker=SERIES_TICKER,
    bankroll=BANKROLL,
    risk_pct=0.01,
    num_markets=NUM_MARKETS,
    volume_fill_pct=VOLUME_FILL_PCT,
    sell_price=0.40,
    tp_fill_rate=TP_FILL_RATE,
    refresh_markets=(REFRESH == 1),
)

results = engine.run()

import os
import json
from datetime import datetime

_dir = os.path.dirname(os.path.abspath(__file__))

# Save trade log CSV
df = results['df']
if not df.empty:
    base = f"trades_{SERIES_TICKER}"
    csv_path = os.path.join(_dir, f"{base}_recent.csv" if OUTPUT_RECENT else f"{base}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTrade log saved to {csv_path}")

# Save bot-ready config: copy backtest_results.py into your bot
def _r(x):
    return round(x, 2) if isinstance(x, (int, float)) else x

bot_config = {
    'series_ticker': SERIES_TICKER,
    'buy_price': 0.10,
    'sell_price': 0.40,
    'risk_pct': 0.01,
    'entry_cutoff_min': 4,
    'static_risk_dol': 7.0,
    'risk_threshold_gain': 100,
    'tp_fill_rate': TP_FILL_RATE,
    'backtest': {
        'markets_analyzed': results['markets_total'],
        'markets_filled': results['markets_filled'],
        'tp_pct': _r(results['tp_pct']),
        'settle_pct': _r(results['settle_pct']),
        'settle_win_rate': _r(results['settle_win_rate']),
        'total_return_pct': _r(results['total_return_pct']),
        'max_drawdown_pct': _r(results['max_drawdown_pct']),
        'avg_win': _r(results['avg_win']),
        'avg_loss': _r(results['avg_loss']),
        'sharpe': _r(results['sharpe']),
        'run_at': datetime.now().isoformat(),
    }
}
config_path = os.path.join(_dir, 'backtest_results.py')
with open(config_path, 'w') as f:
    f.write('"""Backtest results - copy into your bot.\n')
    f.write(f'Run: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    f.write('"""\n\n')
    f.write('BACKTEST_CONFIG = ')
    f.write(json.dumps(bot_config, indent=2))
    f.write('\n')
print(f"Bot config saved to {config_path}")
