#!/usr/bin/env python3
"""
VIX-filtered backtest: filter trades to only those on days when VIX >= threshold.
Does NOT modify run_backtest.py, engine.py, or strategy.py.
Post-processes the existing trade log CSV.

Usage:
    python backtest_vix_filter.py [trades_csv] [--vix-min 18]
    python backtest_vix_filter.py trades_KXBTC15M.csv --vix-min 20

Compares filtered results to baseline.
"""
import argparse
import os
import re
from datetime import datetime

import pandas as pd
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Filter backtest by VIX threshold")
    p.add_argument("trades_csv", nargs="?", default=os.path.join(_dir, "trades_KXBTC15M.csv"),
                   help="Trade log CSV")
    p.add_argument("--vix-min", type=float, default=18.0,
                   help="Only trade when VIX >= this (default: 18)")
    p.add_argument("--vix-csv", default=os.path.join(_dir, "overlay_vix.csv"),
                   help="VIX overlay CSV (date, value)")
    return p.parse_args()


MONTH = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
         "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def market_ticker_to_date(ticker, default_year=2026):
    """Parse KXBTC15M-26FEB070900-00 -> 2026-02-26."""
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    s = parts[1]  # e.g. 26FEB070900
    m = re.match(r"(\d{2})([A-Z]{3})", s)
    if not m:
        return None
    day = int(m.group(1))
    mon = MONTH.get(m.group(2).upper())
    if mon is None:
        return None
    try:
        return datetime(default_year, mon, day).date()
    except ValueError:
        return None


def load_vix_by_date(vix_csv):
    """Load VIX as {date: value}. Also returns sorted (date, value) for alignment."""
    if not os.path.exists(vix_csv):
        return {}, []
    df = pd.read_csv(vix_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    by_date = dict(zip(df["date"].dt.date, df["value"]))
    sorted_pairs = list(zip(df["date"].dt.date, df["value"]))
    return by_date, sorted_pairs


def compute_metrics(df, initial=1000.0):
    """Compute return, drawdown, sharpe from trade log. Uses bankroll column."""
    if df.empty:
        return {"total_return_pct": 0, "max_drawdown_pct": 0, "sharpe": 0,
                "trades": 0, "final_bankroll": initial}
    df = df.copy()
    df["bankroll"] = pd.to_numeric(df["bankroll"], errors="coerce")
    market_max_ts = df.groupby("market")["ts"].transform("max")
    df["_pts"] = df["ts"].fillna(market_max_ts + 60)
    df = df.sort_values("_pts").dropna(subset=["bankroll"])
    equity = df["bankroll"].values
    # Prepend initial for first point
    equity = np.concatenate([[initial], equity])
    final = equity[-1]
    total_return_pct = (final - initial) / initial * 100
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.where(peak > 0, peak, np.nan)
    max_dd = np.nanmin(drawdown) * 100 if np.any(drawdown < 0) else 0
    daily_ret = np.diff(equity) / equity[:-1]
    daily_ret = daily_ret[~np.isnan(daily_ret) & np.isfinite(daily_ret)]
    sharpe = (np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) if len(daily_ret) > 1 and np.std(daily_ret) > 0 else 0
    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "trades": len(df),
        "markets": df["market"].nunique(),
        "final_bankroll": final,
    }


def main():
    args = parse_args()
    trades_path = args.trades_csv
    vix_min = args.vix_min
    vix_csv = args.vix_csv

    if not os.path.exists(trades_path):
        print(f"File not found: {trades_path}")
        return 1

    df = pd.read_csv(trades_path)
    vix_by_date, vix_sorted = load_vix_by_date(vix_csv)

    if not vix_by_date and not vix_sorted:
        print(f"VIX data not found: {vix_csv}")
        print("Run overlay_analysis.py first to fetch VIX data.")
        return 1

    # Build market -> date (chronological order)
    market_max_ts = df.groupby("market")["ts"].transform("max")
    df["_plot_ts"] = df["ts"].fillna(market_max_ts + 60)
    markets_ordered = df.groupby("market")["_plot_ts"].min().sort_values().index.tolist()

    # Market -> VIX: direct date lookup, or align by position when no overlap
    market_to_vix = {}
    for i, m in enumerate(markets_ordered):
        d = market_ticker_to_date(m)
        vix = vix_by_date.get(d) if d else None
        if vix is None and vix_sorted:
            # Align by position: market i gets VIX value at position -n+i
            n = len(markets_ordered)
            if len(vix_sorted) >= n:
                vix = vix_sorted[-(n - i)][1]
            else:
                vix = vix_sorted[-1][1]
        if vix is not None:
            market_to_vix[m] = vix

    markets_keep = set(m for m, v in market_to_vix.items() if v >= vix_min)

    df_baseline = df
    df_filtered = df[df["market"].isin(markets_keep)]

    baseline = compute_metrics(df_baseline)
    filtered = compute_metrics(df_filtered)

    print("=" * 60)
    print("VIX-Filtered Backtest Comparison")
    print("=" * 60)
    print(f"VIX threshold: >= {vix_min}")
    print(f"Baseline markets: {df_baseline['market'].nunique()}")
    print(f"Filtered markets (VIX>={vix_min}): {len(markets_keep)}")
    print()
    print(f"{'Metric':<22}  {'Baseline':>12}  {'Filtered':>12}  {'Diff':>12}")
    print("-" * 60)
    print(f"{'Total return %':<22}  {baseline['total_return_pct']:>+11.1f}%  {filtered['total_return_pct']:>+11.1f}%  {filtered['total_return_pct']-baseline['total_return_pct']:>+11.1f}%")
    print(f"{'Max drawdown %':<22}  {baseline['max_drawdown_pct']:>11.1f}%  {filtered['max_drawdown_pct']:>11.1f}%  {filtered['max_drawdown_pct']-baseline['max_drawdown_pct']:>+11.1f}%")
    print(f"{'Sharpe':<22}  {baseline['sharpe']:>12.2f}  {filtered['sharpe']:>12.2f}  {filtered['sharpe']-baseline['sharpe']:>+12.2f}")
    print(f"{'Final bankroll':<22}  ${baseline['final_bankroll']:>11,.0f}  ${filtered['final_bankroll']:>11,.0f}  ${filtered['final_bankroll']-baseline['final_bankroll']:>+11,.0f}")
    print()

    if filtered["total_return_pct"] > baseline["total_return_pct"]:
        print("VIX filter IMPROVES return (on this sample).")
    elif filtered["total_return_pct"] < baseline["total_return_pct"]:
        print("VIX filter REDUCES return (on this sample).")
    else:
        print("VIX filter has no effect on return.")

    if filtered["sharpe"] > baseline["sharpe"]:
        print("VIX filter IMPROVES risk-adjusted return (Sharpe).")
    elif filtered["sharpe"] < baseline["sharpe"]:
        print("VIX filter REDUCES risk-adjusted return (Sharpe).")

    # Save filtered trades for inspection (drop temp columns)
    out_path = trades_path.replace(".csv", f"_vix_filtered_{int(vix_min)}.csv")
    out_df = df_filtered.drop(columns=["_plot_ts"], errors="ignore")
    out_df.to_csv(out_path, index=False)
    print(f"\nFiltered trades saved: {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
