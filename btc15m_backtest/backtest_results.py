"""Backtest results - copy into your bot.
Run: 2026-04-06 13:35
"""

BACKTEST_CONFIG = {
  "series_ticker": "KXBTC15M",
  "buy_price": 0.1,
  "sell_price": 0.4,
  "risk_pct": 0.01,
  "entry_cutoff_min": 4,
  "static_risk_dol": 7.0,
  "risk_threshold_gain": 100,
  "tp_fill_rate": 1.0,
  "backtest": {
    "markets_analyzed": 10681,
    "markets_filled": 1389,
    "tp_pct": 46.51,
    "settle_pct": 55.36,
    "settle_win_rate": 0.26,
    "total_return_pct": 1849.39,
    "max_drawdown_pct": 27.98,
    "avg_win": 79.25,
    "avg_loss": -65.39,
    "sharpe": 1.44,
    "run_at": "2026-04-06T13:35:32.324391"
  }
}
