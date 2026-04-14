#!/usr/bin/env python3
"""
Find optimal entry (buy) and exit (sell) prices by grid search over backtest data.

Usage:
    python optimize_prices.py KXBTC15M 2000     # BTC, 2000 markets
    python optimize_prices.py KXETH15M 1000    # ETH, 1000 markets
"""
import sys
import os
from datetime import datetime
from engine import Engine

SERIES = sys.argv[1] if len(sys.argv) > 1 else 'KXBTC15M'
NUM_MARKETS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
VOL_FILL = 0.10

# Grid: buy prices (cents) and sell prices (cents)
# Must have buy < sell and reasonable spread
# Granular: 2¢ steps for buy (6–20¢), 2¢ steps for sell (24–54¢)
BUY_CENTS = list(range(6, 21, 2))   # 6, 8, 10, 12, 14, 16, 18, 20
SELL_CENTS = list(range(24, 56, 2))  # 24, 26, 28, ..., 52, 54

# Capture output for saving
_output_lines = []

def _log(s=""):
    print(s)
    _output_lines.append(s)

_log(f"Optimizing entry/exit prices: {SERIES}, {NUM_MARKETS} markets")
_log(f"  Buy grid:  {BUY_CENTS}c")
_log(f"  Sell grid: {SELL_CENTS}c")
_log()

results = []
for buy_c in BUY_CENTS:
    for sell_c in SELL_CENTS:
        if buy_c >= sell_c:
            continue
        buy_p = buy_c / 100.0
        sell_p = sell_c / 100.0

        engine = Engine(
            series_ticker=SERIES,
            bankroll=1000.0,
            risk_pct=0.03,
            num_markets=NUM_MARKETS,
            volume_fill_pct=VOL_FILL,
            buy_price=buy_p,
            sell_price=sell_p,
        )
        r = engine.run(silent=True)

        results.append({
            'buy': buy_c,
            'sell': sell_c,
            'return_pct': r['total_return_pct'],
            'final_br': r['final_bankroll'],
            'filled': r['markets_filled'],
            'tp_pct': r['tp_pct'],
            'settle_pct': r['settle_pct'],
            'sharpe': r['sharpe'],
            'max_dd_pct': r['max_drawdown_pct'],
        })
        _log(f"  {buy_c}c / {sell_c}c  ->  {r['total_return_pct']:+.1f}%  (filled: {r['markets_filled']}, TP: {r['tp_pct']:.1f}%)")

# Sort by return
results.sort(key=lambda x: x['return_pct'], reverse=True)

_log()
_log("=" * 70)
_log("  TOP 10 BY TOTAL RETURN")
_log("=" * 70)
for i, r in enumerate(results[:10], 1):
    _log(f"  {i}. Buy {r['buy']}c / Sell {r['sell']}c  ->  {r['return_pct']:+.1f}%  |  Final: ${r['final_br']:,.0f}  |  Fills: {r['filled']}  |  TP: {r['tp_pct']:.1f}%  |  DD: {r['max_dd_pct']:.1f}%")

best = results[0]
_log()
_log("  OPTIMAL (by return):")
_log(f"    BUY_PRICE  = {best['buy']/100:.2f}  ({best['buy']}c)")
_log(f"    SELL_PRICE = {best['sell']/100:.2f}  ({best['sell']}c)")
_log(f"    Expected return: {best['return_pct']:+.1f}%")
_log()

# Save results to file
_dir = os.path.dirname(os.path.abspath(__file__))
report_path = os.path.join(_dir, f"price_optimizer_results_{SERIES}.txt")
header = f"Price Optimizer: {SERIES}  |  {NUM_MARKETS} markets  |  Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
with open(report_path, 'w') as f:
    f.write(header + "\n")
    f.write("\n".join(_output_lines))
    f.write("\n")
print(f"Results saved to {report_path}")
