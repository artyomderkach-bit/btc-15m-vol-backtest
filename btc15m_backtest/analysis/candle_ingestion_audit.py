#!/usr/bin/env python3
"""
Audit 1-minute candle ingestion for the N most recent settled markets in a series.

Uses DataFetcher.fetch_candles (live + historical fallback, 404-safe).

Kalshi returns **sparse** candle rows (periods with book/trade updates), not a dense
minute grid. The backtest aligns with ``engine.expand_sparse_candles_to_minute_grid``
(forward-fill), same idea as trade-aggregated minute bars. This audit records both
raw API counts and the expanded minute-grid bar count.

For each market we record:
- viable: settled with yes/no result, valid open/close, duration > 0
- n_candles: raw API row count (sparse)
- n_bars_sparse: bars after parse + window filter (still sparse)
- n_bars_minute_grid: one bar per contract minute after forward-fill (matches default INX research)
- coverage_ratio: n_candles / dur_min (sparse; often << 1 on illiquid series)
- coverage_minute_grid: n_bars_minute_grid / dur_min (often ~1.0 when data exists)

Usage:
  python3 candle_ingestion_audit.py KXINXU 1000
  python3 candle_ingestion_audit.py KXINXU 1000 --no-fill-minute-gaps   # skip grid metrics
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import DataFetcher
from engine import expand_sparse_candles_to_minute_grid, parse_candle


def _to_ts(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv // 1000 if iv > 1_000_000_000_000 else iv
    try:
        import pandas as pd
        return int(pd.Timestamp(v).timestamp())
    except Exception:
        return None


def main():
    argv = [a for a in sys.argv[1:] if a]
    fill_minute_grid = "--no-fill-minute-gaps" not in argv
    argv = [a for a in argv if a != "--no-fill-minute-gaps"]
    series = argv[0] if len(argv) > 0 else "KXINXU"
    n = int(argv[1]) if len(argv) > 1 else 1000

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"candle_audit_{series}_{n}.csv")
    summary_path = os.path.join(out_dir, f"candle_audit_{series}_{n}_summary.json")

    f = DataFetcher(series)
    cutoff = f._market_settled_cutoff_ts()
    print(f"Series: {series}  |  market_settled_ts cutoff (unix): {cutoff}")
    print(f"Minute-grid metrics (n_bars_minute_grid): {'on' if fill_minute_grid else 'off'}")
    markets = f.fetch_markets(limit=n, refresh=False)
    markets = sorted(markets, key=lambda m: f._close_ts(m), reverse=True)[:n]
    print(f"Markets to audit: {len(markets)}")

    rows = []
    viable_with_data = 0
    viable_no_data = []

    for i, m in enumerate(markets):
        ticker = m.get("ticker")
        result = m.get("result")
        ots = _to_ts(m.get("open_time"))
        cts = _to_ts(m.get("close_time"))

        viable = bool(
            ticker
            and result in ("yes", "no")
            and ots
            and cts
            and cts > ots
        )

        if not viable:
            rows.append(
                {
                    "ticker": ticker or "",
                    "result": result or "",
                    "viable": "no",
                    "skip_reason": "not_settled_or_bad_times"
                    if not (ticker and ots and cts and cts > ots)
                    else "missing_result",
                    "n_candles": 0,
                    "n_bars_sparse": "",
                    "n_bars_minute_grid": "",
                    "dur_min": "",
                    "coverage_ratio": "",
                    "coverage_minute_grid": "",
                    "prefer_historical": "",
                    "close_ts": cts or "",
                }
            )
            continue

        dur_min = max(1, (cts - ots) // 60)
        prefer_hist = bool(cutoff and int(cts) < int(cutoff))
        try:
            candles = f.fetch_candles(ticker, ots, cts, period_interval=1)
        except Exception as e:
            rows.append(
                {
                    "ticker": ticker,
                    "result": result,
                    "viable": "yes",
                    "skip_reason": f"api_error:{e!s}",
                    "n_candles": 0,
                    "n_bars_sparse": 0,
                    "n_bars_minute_grid": 0,
                    "dur_min": dur_min,
                    "coverage_ratio": 0.0,
                    "coverage_minute_grid": 0.0,
                    "prefer_historical": prefer_hist,
                    "close_ts": cts,
                }
            )
            viable_no_data.append(ticker)
            continue

        nc = len(candles)
        sparse = []
        for rc in candles:
            b = parse_candle(rc)
            if b and ots < b.ts <= cts:
                sparse.append(b)
        n_sparse = len(sparse)
        n_grid = len(expand_sparse_candles_to_minute_grid(candles, ots, cts)) if fill_minute_grid else 0
        cov = round(nc / dur_min, 4) if dur_min else 0.0
        cov_grid = round(n_grid / dur_min, 4) if dur_min and fill_minute_grid else ""
        if nc > 0:
            viable_with_data += 1
        else:
            viable_no_data.append(ticker)

        rows.append(
            {
                "ticker": ticker,
                "result": result,
                "viable": "yes",
                "skip_reason": "no_candles" if nc == 0 else "",
                "n_candles": nc,
                "n_bars_sparse": n_sparse,
                "n_bars_minute_grid": n_grid if fill_minute_grid else "",
                "dur_min": dur_min,
                "coverage_ratio": cov,
                "coverage_minute_grid": cov_grid,
                "prefer_historical": prefer_hist,
                "close_ts": cts,
            }
        )

        if (i + 1) % 100 == 0:
            print(
                f"  ... {i + 1}/{len(markets)}  viable_with_data: {viable_with_data}  viable_zero_candles: {len(viable_no_data)}",
                flush=True,
            )

    f.close()

    viable_rows = [r for r in rows if r.get("viable") == "yes"]
    n_viable = len(viable_rows)
    covs = [float(r["coverage_ratio"]) for r in viable_rows if r.get("n_candles", 0) > 0 and r.get("coverage_ratio") != ""]
    covs_grid = [
        float(r["coverage_minute_grid"])
        for r in viable_rows
        if r.get("coverage_minute_grid") not in ("", None)
    ]

    summary = {
        "series": series,
        "markets_requested": n,
        "viable_settled_with_valid_window": n_viable,
        "viable_with_at_least_one_candle": viable_with_data,
        "viable_with_zero_candles": len(viable_no_data),
        "non_viable": len(rows) - n_viable,
        "coverage_ratio_median_sparse_api": statistics.median(covs) if covs else None,
        "coverage_minute_grid_median": statistics.median(covs_grid) if covs_grid else None,
        "minute_grid_metrics_enabled": fill_minute_grid,
        "tickers_viable_but_no_candles": viable_no_data[:50],
        "note": "Viable = settled yes/no + open/close + duration>0. n_candles is sparse API count; coverage_minute_grid matches expand_sparse_candles_to_minute_grid (backtest default).",
    }

    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["ticker"])
        w.writeheader()
        w.writerows(rows)

    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print()
    print("=== Summary ===")
    print(f"Viable markets (settled + valid times): {n_viable}")
    print(f"Viable with >0 candles: {viable_with_data}")
    print(f"Viable with 0 candles (data gap): {len(viable_no_data)}")
    if covs:
        print(f"Median coverage sparse API (n_candles/dur_min): {statistics.median(covs):.4f}")
    if covs_grid:
        print(f"Median coverage minute grid (forward-filled): {statistics.median(covs_grid):.4f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")
    if viable_no_data:
        print(f"First tickers with 0 candles: {', '.join(viable_no_data[:10])}")


if __name__ == "__main__":
    main()
