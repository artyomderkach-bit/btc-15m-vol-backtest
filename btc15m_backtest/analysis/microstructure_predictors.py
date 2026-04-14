#!/usr/bin/env python3
"""
Kalshi-level microstructure features → per-market P&L prediction analysis.

For every market in the backtest trade log, extracts features from the
first 4 candles (minutes 1–4, BEFORE the 6-min entry window closes) and
tests whether they correlate with / predict per-market P&L.

Features extracted (from bars 1–4 of the 15-min market):
  - spread_mean        : mean(ask_close − bid_close)          tighter = more liquid
  - spread_min         : min(ask_low − bid_high)              tightest spread seen
  - vol_early          : sum of volume in first 4 bars        early activity
  - vol_bar1           : volume of the very first bar         opening burst
  - oi_bar4            : open_interest at bar 4               depth proxy
  - rvol_4bar          : std of 1-min log-returns (bars 1–4)  realized micro-vol
  - price_mid_bar4     : midpoint (ask+bid)/2 at bar 4        skew / distance from 50¢
  - imbalance_bar4     : (bid − (1−ask)) / spread at bar 4    order-flow imbalance
  - range_early        : max(price_high) − min(price_low)     early price range
  - price_level_abs50  : |mid − 0.50| at bar 4                how lopsided the market is

Outputs (in charts/btc_main/):
  microstructure_features.csv        full feature + P&L table
  microstructure_corr.csv            feature vs P&L correlations
  microstructure_bucket_summary.csv  P&L by quantile buckets of key features
  microstructure_summary.txt         human-readable report
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT)                      # btc15m_backtest/
_PROJECT = os.path.dirname(_ROOT)                      # workspace root
TRADELOG = os.path.join(_ROOT, "charts", "btc_main", "current_strat_full_tradelog.csv")
DB_PATH = os.path.join(_PROJECT, "cache.db")
OUT_DIR = os.path.join(_ROOT, "charts", "btc_main")

EARLY_BARS = 4  # first 4 minutes of the 15-min market


def get_open_close_ts(market_data: dict) -> Optional[tuple]:
    """Derive (open_ts, close_ts) from the market metadata's close_time field."""
    ct = market_data.get("close_time")
    if ct is None:
        return None
    if isinstance(ct, str):
        close_ts = int(pd.Timestamp(ct).timestamp())
    else:
        close_ts = int(ct)
    open_ts = close_ts - 15 * 60
    return open_ts, close_ts


def _dollar_or_cent(obj, key):
    dk = key + "_dollars"
    if dk in obj and obj[dk] is not None:
        try:
            return float(obj[dk])
        except (TypeError, ValueError):
            pass
    v = obj.get(key)
    if v is None:
        return None
    try:
        return int(v) / 100.0
    except (TypeError, ValueError):
        return None


def extract_features(candles_json: list, open_ts: int) -> Optional[dict]:
    """Extract microstructure features from first EARLY_BARS candle bars."""
    close_ts = open_ts + 15 * 60
    bars = []
    for c in candles_json:
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        ts = int(ts)
        if not (open_ts < ts <= close_ts):
            continue

        price = c.get("price") or {}
        ask = c.get("yes_ask") or {}
        bid = c.get("yes_bid") or {}

        vol_raw = c.get("volume_fp") or c.get("volume")
        try:
            vol = int(float(vol_raw)) if vol_raw is not None else 0
        except (TypeError, ValueError):
            vol = 0

        oi_raw = c.get("open_interest_fp") or c.get("open_interest")
        try:
            oi = int(float(oi_raw)) if oi_raw is not None else 0
        except (TypeError, ValueError):
            oi = 0

        bars.append({
            "ts": ts,
            "p_open": _dollar_or_cent(price, "open"),
            "p_high": _dollar_or_cent(price, "high"),
            "p_low": _dollar_or_cent(price, "low"),
            "p_close": _dollar_or_cent(price, "close"),
            "ask_open": _dollar_or_cent(ask, "open"),
            "ask_high": _dollar_or_cent(ask, "high"),
            "ask_low": _dollar_or_cent(ask, "low"),
            "ask_close": _dollar_or_cent(ask, "close"),
            "bid_open": _dollar_or_cent(bid, "open"),
            "bid_high": _dollar_or_cent(bid, "high"),
            "bid_low": _dollar_or_cent(bid, "low"),
            "bid_close": _dollar_or_cent(bid, "close"),
            "volume": vol,
            "oi": oi,
        })

    bars.sort(key=lambda b: b["ts"])
    if len(bars) < EARLY_BARS:
        return None

    early = bars[:EARLY_BARS]

    spreads_close = []
    for b in early:
        ac, bc = b["ask_close"], b["bid_close"]
        if ac is not None and bc is not None and ac > bc:
            spreads_close.append(ac - bc)

    spreads_tight = []
    for b in early:
        al, bh = b["ask_low"], b["bid_high"]
        if al is not None and bh is not None and al >= bh:
            spreads_tight.append(al - bh)

    prices_close = [b["p_close"] for b in early if b["p_close"] is not None]
    log_rets = []
    for i in range(1, len(prices_close)):
        if prices_close[i - 1] > 0 and prices_close[i] > 0:
            log_rets.append(math.log(prices_close[i] / prices_close[i - 1]))

    vol_early = sum(b["volume"] for b in early)
    vol_bar1 = early[0]["volume"]
    oi_bar4 = early[EARLY_BARS - 1]["oi"]

    last = early[-1]
    ask_c = last["ask_close"]
    bid_c = last["bid_close"]
    mid_bar4 = (ask_c + bid_c) / 2 if (ask_c is not None and bid_c is not None) else None
    spread_bar4 = (ask_c - bid_c) if (ask_c is not None and bid_c is not None and ask_c > bid_c) else None

    imbalance = None
    if spread_bar4 and spread_bar4 > 0 and bid_c is not None and ask_c is not None:
        imbalance = (bid_c - (1.0 - ask_c)) / spread_bar4

    highs = [b["p_high"] for b in early if b["p_high"] is not None]
    lows = [b["p_low"] for b in early if b["p_low"] is not None]
    range_early = (max(highs) - min(lows)) if highs and lows else None

    # Features from all 15 bars (full market)
    all_vols = sum(b["volume"] for b in bars)
    all_prices = [b["p_close"] for b in bars if b["p_close"] is not None]
    all_log_rets = []
    for i in range(1, len(all_prices)):
        if all_prices[i - 1] > 0 and all_prices[i] > 0:
            all_log_rets.append(math.log(all_prices[i] / all_prices[i - 1]))

    return {
        "spread_mean": np.mean(spreads_close) if spreads_close else None,
        "spread_min": min(spreads_tight) if spreads_tight else None,
        "vol_early": vol_early,
        "vol_bar1": vol_bar1,
        "oi_bar4": oi_bar4,
        "rvol_4bar": np.std(log_rets) if len(log_rets) >= 2 else None,
        "price_mid_bar4": mid_bar4,
        "imbalance_bar4": imbalance,
        "range_early": range_early,
        "price_level_abs50": abs(mid_bar4 - 0.50) if mid_bar4 is not None else None,
        "vol_total": all_vols,
        "rvol_full": np.std(all_log_rets) if len(all_log_rets) >= 4 else None,
        "n_bars": len(bars),
    }


def main():
    if not os.path.exists(TRADELOG):
        print(f"Missing trade log: {TRADELOG}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(DB_PATH):
        print(f"Missing candle cache: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(TRADELOG)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")

    exit_mask = df["action"].isin(("sell_fill", "settlement", "stop_loss"))
    pnl_by_mkt = df.loc[exit_mask].groupby("market")["pnl"].sum()

    buy_fill_mkts = set(df.loc[df["action"] == "buy_fill", "market"])

    conn = sqlite3.connect(DB_PATH)

    # Pre-load market metadata (close_time etc) for all BTC markets
    mkt_meta = {}
    for ticker, data_json in conn.execute(
        "SELECT ticker, data FROM markets WHERE series_ticker='KXBTC15M'"
    ).fetchall():
        mkt_meta[ticker] = json.loads(data_json)

    rows = []
    missing = 0
    for mkt, pnl in pnl_by_mkt.items():
        if mkt not in buy_fill_mkts:
            continue

        meta = mkt_meta.get(mkt)
        if meta is None:
            missing += 1
            continue
        ts_pair = get_open_close_ts(meta)
        if ts_pair is None:
            missing += 1
            continue
        open_ts, close_ts = ts_pair

        row = conn.execute(
            "SELECT data FROM candlesticks WHERE market_ticker=?", (mkt,)
        ).fetchone()
        if row is None:
            missing += 1
            continue
        candles = json.loads(row[0])
        feats = extract_features(candles, open_ts)
        if feats is None:
            missing += 1
            continue
        feats["market"] = mkt
        feats["pnl"] = float(pnl)
        feats["won"] = 1 if pnl > 0 else 0
        rows.append(feats)

    conn.close()
    print(f"Extracted features for {len(rows)} markets ({missing} skipped / missing candles)")

    if not rows:
        print("No features extracted — nothing to analyze.", file=sys.stderr)
        sys.exit(1)

    F = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    F.to_csv(os.path.join(OUT_DIR, "microstructure_features.csv"), index=False)

    # --- Correlations ---
    feat_cols = [
        "spread_mean", "spread_min", "vol_early", "vol_bar1", "oi_bar4",
        "rvol_4bar", "price_mid_bar4", "imbalance_bar4", "range_early",
        "price_level_abs50", "vol_total", "rvol_full",
    ]
    corr_rows = []
    for c in feat_cols:
        sub = F[["pnl", "won", c]].dropna()
        if len(sub) < 20:
            continue
        corr_rows.append({
            "feature": c,
            "n": len(sub),
            "corr_pnl": sub["pnl"].corr(sub[c]),
            "corr_win": sub["won"].corr(sub[c]),
            "mean_val": sub[c].mean(),
            "std_val": sub[c].std(),
        })
    C = pd.DataFrame(corr_rows).sort_values("corr_pnl", key=abs, ascending=False)
    C.to_csv(os.path.join(OUT_DIR, "microstructure_corr.csv"), index=False)

    # --- Bucket analysis for top features ---
    bucket_rows = []
    top_feats = list(C.head(6)["feature"])
    for feat in top_feats:
        sub = F[["pnl", "won", feat]].dropna()
        if len(sub) < 40:
            continue
        try:
            sub["q"] = pd.qcut(sub[feat].rank(method="first"), 4,
                               labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        except ValueError:
            continue
        for q, g in sub.groupby("q", observed=True):
            bucket_rows.append({
                "feature": feat,
                "quartile": q,
                "n": len(g),
                "mean_pnl": g["pnl"].mean(),
                "median_pnl": g["pnl"].median(),
                "win_rate": g["won"].mean(),
                "sum_pnl": g["pnl"].sum(),
                "mean_feat_val": g[feat].mean(),
            })
    B = pd.DataFrame(bucket_rows)
    B.to_csv(os.path.join(OUT_DIR, "microstructure_bucket_summary.csv"), index=False)

    # --- Text report ---
    lines = []
    lines.append("MICROSTRUCTURE PREDICTOR REPORT")
    lines.append(f"Markets with features: {len(F)}  |  Feature window: first {EARLY_BARS} bars (~{EARLY_BARS} min)")
    lines.append("")
    lines.append("=== Feature → P&L correlations (sorted by |corr_pnl|) ===")
    lines.append(C.to_string(index=False))
    lines.append("")
    lines.append("=== Quartile bucket analysis (top 6 features) ===")
    for feat in top_feats:
        fb = B[B["feature"] == feat]
        if fb.empty:
            continue
        lines.append(f"\n--- {feat} ---")
        lines.append(fb[["quartile", "n", "mean_feat_val", "mean_pnl",
                         "median_pnl", "win_rate", "sum_pnl"]].to_string(index=False))

    lines.append("")
    lines.append("=== INTERPRETATION ===")
    lines.append("- These features are measured from the FIRST 4 minutes of each market,")
    lines.append("  which is BEFORE or concurrent with the entry fill window (6 min cutoff).")
    lines.append("- Strongest correlations point to microstructure conditions that your")
    lines.append("  10¢ buy / TP sell strategy benefits from.")
    lines.append("- Quartile splits show monotonicity: does P&L improve consistently")
    lines.append("  as the feature moves from Q1 to Q4, or is it nonlinear?")
    lines.append("- CAUTION: in-sample on one strategy backtest; validate out-of-sample.")
    lines.append("- Even modest r ~ 0.10-0.15 on microstructure features can be actionable")
    lines.append("  because they match the TIME SCALE of your edge (minutes, not days).")

    txt = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "microstructure_summary.txt"), "w") as f:
        f.write(txt)

    print()
    print(txt)
    print(f"\nWrote: {OUT_DIR}/microstructure_features.csv")
    print(f"Wrote: {OUT_DIR}/microstructure_corr.csv")
    print(f"Wrote: {OUT_DIR}/microstructure_bucket_summary.csv")
    print(f"Wrote: {OUT_DIR}/microstructure_summary.txt")


if __name__ == "__main__":
    main()
