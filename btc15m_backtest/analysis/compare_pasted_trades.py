#!/usr/bin/env python3
"""
Compare pasted real trades to backtest. Paste your trade log, parse it, and match.
"""
import os
import sys
import csv
import re
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES_CSV = os.path.join(_ROOT, "trades_KXBTC15M.csv")
SERIES = "KXBTC15M"

# Pasted trades from user (2026-03-12)
PASTED_TRADES = """
2026-03-12 04:04:05 UTC  BUY  NO    5 @  10¢  KXBTC15M-26MAR120015-15  (fee $0.00)
2026-03-12 04:03:49 UTC  BUY  NO   43 @  10¢  KXBTC15M-26MAR120015-15  (fee $0.00)
2026-03-12 04:03:49 UTC  BUY  NO   22 @  10¢  KXBTC15M-26MAR120015-15  (fee $0.00)
2026-03-12 03:47:04 UTC  BUY  YES  48 @  10¢  KXBTC15M-26MAR120000-00  (fee $0.00)
2026-03-12 03:47:02 UTC  BUY  YES  22 @  10¢  KXBTC15M-26MAR120000-00  (fee $0.00)
2026-03-12 03:33:18 UTC  BUY  NO   67 @  10¢  KXBTC15M-26MAR112345-45  (fee $0.00)
2026-03-12 03:33:16 UTC  BUY  NO    3 @  10¢  KXBTC15M-26MAR112345-45  (fee $0.00)
2026-03-12 03:20:56 UTC  BUY  NO   41 @  10¢  KXBTC15M-26MAR112330-30  (fee $0.00)
2026-03-12 03:20:55 UTC  BUY  NO   29 @  10¢  KXBTC15M-26MAR112330-30  (fee $0.00)
2026-03-12 02:19:45 UTC  BUY  YES  50 @  10¢  KXBTC15M-26MAR112230-30  (fee $0.00)
2026-03-12 02:19:44 UTC  BUY  YES   1 @  10¢  KXBTC15M-26MAR112230-30  (fee $0.00)
2026-03-12 02:19:43 UTC  BUY  YES   9 @  10¢  KXBTC15M-26MAR112230-30  (fee $0.00)
2026-03-12 02:19:43 UTC  BUY  YES   1 @  10¢  KXBTC15M-26MAR112230-30  (fee $0.00)
2026-03-12 02:19:43 UTC  BUY  YES   8 @  10¢  KXBTC15M-26MAR112230-30  (fee $0.00)
2026-03-12 02:19:42 UTC  BUY  YES   1 @  10¢  KXBTC15M-26MAR112230-30  (fee $0.00)
2026-03-12 01:50:14 UTC  BUY  YES  69 @  10¢  KXBTC15M-26MAR112200-00  (fee $0.00)
2026-03-12 01:50:12 UTC  BUY  YES   1 @  10¢  KXBTC15M-26MAR112200-00  (fee $0.00)
2026-03-12 01:42:06 UTC  SELL NO   70 @  33¢  KXBTC15M-26MAR112145-45  (fee $0.00)
2026-03-12 01:35:07 UTC  BUY  YES  55 @  10¢  KXBTC15M-26MAR112145-45  (fee $0.00)
2026-03-12 01:34:57 UTC  BUY  YES   1 @  10¢  KXBTC15M-26MAR112145-45  (fee $0.00)
2026-03-12 01:34:57 UTC  BUY  YES  14 @  10¢  KXBTC15M-26MAR112145-45  (fee $0.00)
2026-03-12 01:03:26 UTC  BUY  YES  70 @  10¢  KXBTC15M-26MAR112115-15  (fee $0.00)
2026-03-12 00:35:53 UTC  SELL NO    7 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:35:53 UTC  SELL NO    5 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:35:53 UTC  SELL NO    1 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:35:53 UTC  SELL NO    2 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:35:52 UTC  SELL NO   14 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:35:51 UTC  SELL NO    5 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:35:51 UTC  SELL NO   14 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:35:51 UTC  SELL NO   14 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:35:50 UTC  SELL NO    3 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:35:49 UTC  SELL NO    5 @  33¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-12 00:34:27 UTC  BUY  YES  70 @  10¢  KXBTC15M-26MAR112045-45  (fee $0.00)
2026-03-11 23:00:20 UTC  BUY  NO   70 @  10¢  KXBTC15M-26MAR111915-15  (fee $0.00)
2026-03-11 22:35:28 UTC  BUY  YES  66 @  10¢  KXBTC15M-26MAR111845-45  (fee $0.00)
2026-03-11 22:35:27 UTC  BUY  YES   3 @  10¢  KXBTC15M-26MAR111845-45  (fee $0.00)
2026-03-11 22:35:26 UTC  BUY  YES   1 @  10¢  KXBTC15M-26MAR111845-45  (fee $0.00)
2026-03-11 22:20:36 UTC  BUY  YES   6 @  10¢  KXBTC15M-26MAR111830-30  (fee $0.00)
2026-03-11 22:20:35 UTC  BUY  YES  30 @  10¢  KXBTC15M-26MAR111830-30  (fee $0.00)
2026-03-11 22:20:32 UTC  BUY  YES  32 @  10¢  KXBTC15M-26MAR111830-30  (fee $0.00)
2026-03-11 22:20:24 UTC  BUY  YES   2 @  10¢  KXBTC15M-26MAR111830-30  (fee $0.00)
2026-03-11 21:35:26 UTC  BUY  NO   28 @  10¢  KXBTC15M-26MAR111745-45  (fee $0.00)
2026-03-11 21:35:25 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR111745-45  (fee $0.00)
2026-03-11 21:35:25 UTC  BUY  NO    5 @  10¢  KXBTC15M-26MAR111745-45  (fee $0.00)
2026-03-11 21:35:24 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR111745-45  (fee $0.00)
2026-03-11 21:35:24 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR111745-45  (fee $0.00)
2026-03-11 21:35:24 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111745-45  (fee $0.00)
2026-03-11 21:35:23 UTC  BUY  NO    6 @  10¢  KXBTC15M-26MAR111745-45  (fee $0.00)
2026-03-11 21:35:23 UTC  BUY  NO   18 @  10¢  KXBTC15M-26MAR111745-45  (fee $0.00)
2026-03-11 21:35:22 UTC  BUY  NO    8 @  10¢  KXBTC15M-26MAR111745-45  (fee $0.00)
2026-03-11 19:35:08 UTC  BUY  NO   56 @  10¢  KXBTC15M-26MAR111545-45  (fee $0.00)
2026-03-11 19:35:07 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR111545-45  (fee $0.00)
2026-03-11 19:35:07 UTC  BUY  NO   11 @  10¢  KXBTC15M-26MAR111545-45  (fee $0.00)
2026-03-11 19:35:05 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR111545-45  (fee $0.00)
2026-03-11 19:35:04 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR111545-45  (fee $0.00)
2026-03-11 17:34:30 UTC  BUY  YES  69 @  10¢  KXBTC15M-26MAR111345-45  (fee $0.00)
2026-03-11 17:34:29 UTC  BUY  YES   1 @  10¢  KXBTC15M-26MAR111345-45  (fee $0.00)
2026-03-11 17:03:10 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR111315-15  (fee $0.00)
2026-03-11 17:03:10 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR111315-15  (fee $0.00)
2026-03-11 17:03:10 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111315-15  (fee $0.00)
2026-03-11 17:03:10 UTC  BUY  NO   11 @  10¢  KXBTC15M-26MAR111315-15  (fee $0.00)
2026-03-11 17:03:09 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111315-15  (fee $0.00)
2026-03-11 17:03:09 UTC  BUY  NO   53 @  10¢  KXBTC15M-26MAR111315-15  (fee $0.00)
2026-03-11 16:34:26 UTC  BUY  NO   61 @  10¢  KXBTC15M-26MAR111245-45  (fee $0.00)
2026-03-11 16:34:26 UTC  BUY  NO    7 @  10¢  KXBTC15M-26MAR111245-45  (fee $0.00)
2026-03-11 16:34:24 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111245-45  (fee $0.00)
2026-03-11 15:38:48 UTC  SELL NO   70 @  33¢  KXBTC15M-26MAR111145-45  (fee $0.00)
2026-03-11 15:31:02 UTC  BUY  YES  70 @  10¢  KXBTC15M-26MAR111145-45  (fee $0.00)
2026-03-11 15:03:21 UTC  BUY  YES  70 @  10¢  KXBTC15M-26MAR111115-15  (fee $0.00)
2026-03-11 14:20:01 UTC  BUY  NO   51 @  10¢  KXBTC15M-26MAR111030-30  (fee $0.00)
2026-03-11 14:20:01 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR111030-30  (fee $0.00)
2026-03-11 14:19:57 UTC  BUY  NO    4 @  10¢  KXBTC15M-26MAR111030-30  (fee $0.00)
2026-03-11 14:19:45 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111030-30  (fee $0.00)
2026-03-11 14:19:44 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111030-30  (fee $0.00)
2026-03-11 14:19:44 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111030-30  (fee $0.00)
2026-03-11 14:19:43 UTC  BUY  NO    8 @  10¢  KXBTC15M-26MAR111030-30  (fee $0.00)
2026-03-11 14:08:26 UTC  SELL YES  62 @  33¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 14:08:21 UTC  SELL YES   1 @  33¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 14:08:21 UTC  SELL YES   7 @  33¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 14:02:39 UTC  BUY  NO   35 @  10¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 14:02:39 UTC  BUY  NO   11 @  10¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 14:02:39 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 14:02:39 UTC  BUY  NO    8 @  10¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 14:02:38 UTC  BUY  NO   10 @  10¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 14:02:37 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 14:02:36 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR111015-15  (fee $0.00)
2026-03-11 13:49:04 UTC  BUY  YES  70 @  10¢  KXBTC15M-26MAR111000-00  (fee $0.00)
2026-03-11 13:33:16 UTC  BUY  NO   62 @  10¢  KXBTC15M-26MAR110945-45  (fee $0.00)
2026-03-11 13:33:16 UTC  BUY  NO    8 @  10¢  KXBTC15M-26MAR110945-45  (fee $0.00)
2026-03-11 13:20:13 UTC  BUY  NO   64 @  10¢  KXBTC15M-26MAR110930-30  (fee $0.00)
2026-03-11 13:20:13 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR110930-30  (fee $0.00)
2026-03-11 13:20:12 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR110930-30  (fee $0.00)
2026-03-11 13:20:11 UTC  BUY  NO    2 @  10¢  KXBTC15M-26MAR110930-30  (fee $0.00)
2026-03-11 09:52:36 UTC  SELL YES  70 @  33¢  KXBTC15M-26MAR110600-00  (fee $0.00)
2026-03-11 09:48:15 UTC  BUY  NO   70 @  10¢  KXBTC15M-26MAR110600-00  (fee $0.00)
2026-03-11 08:06:44 UTC  SELL YES  70 @  33¢  KXBTC15M-26MAR110415-15  (fee $0.00)
2026-03-11 08:05:59 UTC  BUY  NO   11 @  10¢  KXBTC15M-26MAR110415-15  (fee $0.00)
2026-03-11 08:05:58 UTC  BUY  NO    3 @  10¢  KXBTC15M-26MAR110415-15  (fee $0.00)
2026-03-11 08:05:52 UTC  BUY  NO   11 @  10¢  KXBTC15M-26MAR110415-15  (fee $0.00)
2026-03-11 08:05:52 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR110415-15  (fee $0.00)
2026-03-11 08:05:50 UTC  BUY  NO   25 @  10¢  KXBTC15M-26MAR110415-15  (fee $0.00)
2026-03-11 08:05:46 UTC  BUY  NO   11 @  10¢  KXBTC15M-26MAR110415-15  (fee $0.00)
2026-03-11 08:05:45 UTC  BUY  NO    8 @  10¢  KXBTC15M-26MAR110415-15  (fee $0.00)
2026-03-11 07:50:25 UTC  BUY  NO   20 @  10¢  KXBTC15M-26MAR110400-00  (fee $0.00)
2026-03-11 07:50:24 UTC  BUY  NO    5 @  10¢  KXBTC15M-26MAR110400-00  (fee $0.00)
2026-03-11 07:50:24 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR110400-00  (fee $0.00)
2026-03-11 07:50:23 UTC  BUY  NO    5 @  10¢  KXBTC15M-26MAR110400-00  (fee $0.00)
2026-03-11 07:50:22 UTC  BUY  NO    5 @  10¢  KXBTC15M-26MAR110400-00  (fee $0.00)
2026-03-11 07:50:22 UTC  BUY  NO   13 @  10¢  KXBTC15M-26MAR110400-00  (fee $0.00)
2026-03-11 07:50:22 UTC  BUY  NO   21 @  10¢  KXBTC15M-26MAR110400-00  (fee $0.00)
2026-03-11 07:34:44 UTC  BUY  YES  70 @  10¢  KXBTC15M-26MAR110345-45  (fee $0.00)
2026-03-11 06:34:07 UTC  BUY  NO   68 @  10¢  KXBTC15M-26MAR110245-45  (fee $0.00)
2026-03-11 06:34:07 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR110245-45  (fee $0.00)
2026-03-11 06:34:04 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR110245-45  (fee $0.00)
2026-03-11 06:04:57 UTC  BUY  NO    1 @  10¢  KXBTC15M-26MAR110215-15  (fee $0.00)
2026-03-11 06:04:57 UTC  BUY  NO   10 @  10¢  KXBTC15M-26MAR110215-15  (fee $0.00)
2026-03-11 04:20:38 UTC  BUY  NO   70 @  10¢  KXBTC15M-26MAR110030-30  (fee $0.00)
"""


def parse_pasted_trades(text):
    """Parse pasted trade log. Return dict: market -> {side, outcome, buy_qty, sell_qty}."""
    live = defaultdict(lambda: {"side": "", "buy_qty": 0, "sell_qty": 0})
    # Pattern: BUY/SELL YES/NO N @ 10¢/33¢ TICKER
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or not SERIES in line:
            continue
        m = re.search(r"(BUY|SELL)\s+(YES|NO)\s+(\d+)\s+@\s+[^\s]+\s+(" + SERIES + r"[\w-]+)", line)
        if not m:
            continue
        action, side, count, ticker = m.groups()
        count = int(count)
        if action == "BUY":
            live[ticker]["side"] = side.lower()
            live[ticker]["buy_qty"] += count
        elif action == "SELL":
            live[ticker]["sell_qty"] += count
    for t, d in live.items():
        d["outcome"] = "TP" if d["sell_qty"] > 0 else "settlement"
    return dict(live)


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


def main():
    print("=" * 70)
    print("  COMPARE: Your Pasted Trades vs Backtest")
    print("=" * 70)

    real = parse_pasted_trades(PASTED_TRADES)
    backtest = load_backtest_trades()

    real_markets = set(real.keys())
    overlap = real_markets & set(backtest.keys())

    print(f"\n  Your pasted trades:  {len(real_markets)} markets")
    print(f"  Backtest has:        {len(overlap)} of those markets")

    if not overlap:
        print("\n  No overlap - backtest may not include these markets yet.")
        print("  Run: ./backtest 7000  (or more) to include recent markets.")
        return

    matches = []
    mismatches = []
    bt_only = []
    real_only = []

    for m in overlap:
        r, b = real[m], backtest[m]
        if r["outcome"] == b["outcome"]:
            matches.append((m, r, b))
        else:
            mismatches.append((m, r, b))

    for m in real_markets - set(backtest.keys()):
        real_only.append((m, real[m]))
    for m in set(backtest.keys()) - real_markets:
        if m in [x[0] for x in real_only]:
            pass
        # bt_only = markets in backtest but not in pasted - skip for brevity

    print(f"\n  OVERLAP: {len(overlap)} markets in both")
    print(f"  MATCH (same outcome): {len(matches)}/{len(overlap)} = {100*len(matches)/len(overlap):.1f}%")

    if mismatches:
        print(f"\n  MISMATCHES ({len(mismatches)}):")
        for m, r, b in mismatches:
            print(f"    {m}")
            print(f"      Your trades:  {r['outcome']} (buy={r['buy_qty']} sell={r['sell_qty']})")
            print(f"      Backtest:    {b['outcome']} (qty={b['qty']})")
    else:
        print("\n  All overlapping markets MATCH.")

    if real_only:
        print(f"\n  In your trades but NOT in backtest ({len(real_only)}):")
        for m, r in sorted(real_only, key=lambda x: x[0])[:10]:
            print(f"    {m}  {r['outcome']}")
        if len(real_only) > 10:
            print(f"    ... and {len(real_only)-10} more")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
