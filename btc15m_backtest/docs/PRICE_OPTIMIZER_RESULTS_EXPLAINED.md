# Price Optimizer Results Explained

## What the Price Optimizer Does

`optimize_prices.py` runs a **grid search** over buy/sell price combinations. For each pair (e.g. 6¢ buy / 54¢ sell), it backtests the strategy and records total return, fills, TP%, and drawdown.

- **Buy grid:** 6, 8, 10, 12, 14, 16, 18, 20¢ (2¢ steps)
- **Sell grid:** 24, 26, 28, …, 52, 54¢ (2¢ steps)
- **Run:** `python optimize_prices.py KXBTC15M 2000`

---

## How to Read the Output

| Column | Meaning |
|--------|---------|
| **Buy X¢ / Sell Y¢** | Entry price (buy) and exit price (sell) |
| **Return** | Total return % on $1,000 bankroll |
| **Filled** | Number of markets where at least one order filled |
| **TP%** | % of filled markets that hit take-profit (vs. held to settlement) |
| **DD** | Max drawdown % |

---

## Trade-offs

- **Lower buy (6¢):** Fewer fills (harder to get filled), but when filled, bigger profit per trade. Higher return if TP hits.
- **Higher buy (18–20¢):** More fills, but smaller profit per trade and more settlement bleed.
- **Higher sell (50–54¢):** Bigger profit per TP, but lower TP% (harder to reach).
- **Lower sell (24–30¢):** Easier to hit TP, but smaller profit per trade.

---

## Your Latest Results (KXBTC15M, 2000 markets)

**Best by return:** 6¢ buy / 54¢ sell → +1894.6% (Final: $19,946, Fills: 272, TP: 26.5%, DD: 16.4%)

**Top 10:**
1. 6¢/54¢  +1894.6%
2. 6¢/52¢  +1808.8%
3. 6¢/50¢  +1609.2%
4. 6¢/44¢  +1464.3%
5. 6¢/48¢  +1450.4%
6. 6¢/42¢  +1396.6%
7. 6¢/46¢  +1322.0%
8. 6¢/40¢  +1309.6%
9. 6¢/38¢  +1181.6%
10. 8¢/54¢ +1146.3%

**Current default (10¢/33¢):** ~+385% on 7300 markets — fewer fills, more conservative.

**Caveat:** 6¢ fills are rare in live markets. Best backtest return may not translate to live. Use `compare_event_vs_normal.py` to see event-day vs normal-day PnL.
