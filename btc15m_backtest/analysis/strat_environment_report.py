#!/usr/bin/env python3
"""
Analyze *environments* where the current strategy P&L concentrates, and
how much simple macro / calendar features explain (honest caveats included).

Inputs:
  charts/btc_main/current_strat_full_tradelog.csv

Outputs:
  charts/btc_main/strat_environment_summary.txt
  charts/btc_main/strat_environment_by_hour.csv
  charts/btc_main/strat_environment_daily_predictors.csv
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT)
TRADELOG = os.path.join(_ROOT, "charts", "btc_main", "current_strat_full_tradelog.csv")
OUT_DIR = os.path.join(_ROOT, "charts", "btc_main")

# KXBTC15M-26MAR010345-45  -> 2026-03-01 03:45 UTC (interval start)
_TICKER_RE = re.compile(
    r"^KXBTC15M-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})-",
    re.I,
)
_MONTH = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_open_utc(ticker: str) -> Optional[pd.Timestamp]:
    m = _TICKER_RE.match(ticker.strip())
    if not m:
        return None
    yy, mon, dd, hh, mm = m.groups()
    try:
        y = 2000 + int(yy)
        month = _MONTH[mon.upper()]
        d = int(dd)
        h = int(hh)
        minute = int(mm)
        return pd.Timestamp(year=y, month=month, day=d, hour=h, minute=minute, tz="UTC")
    except (KeyError, ValueError):
        return None


def yf_close_series(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    import yfinance as yf
    h = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    if h is None or h.empty:
        return None
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    h = h.reset_index()
    dcol = "Date" if "Date" in h.columns else h.columns[0]
    ccol = "Close" if "Close" in h.columns else h.columns[-1]
    s = pd.Series(
        pd.to_numeric(h[ccol], errors="coerce").values.ravel(),
        index=pd.to_datetime(h[dcol]).dt.normalize(),
    )
    return s.dropna().sort_index()


def main():
    if not os.path.exists(TRADELOG):
        print(f"Missing {TRADELOG}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(TRADELOG)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")

    exit_mask = df["action"].isin(("sell_fill", "settlement", "stop_loss"))
    pnl_by_mkt = df.loc[exit_mask].groupby("market")["pnl"].sum().rename("pnl_mkt")

    filled = df.loc[df["action"] == "buy_fill", "market"].drop_duplicates()
    filled_set = set(filled)

    rows = []
    for mkt, pnl in pnl_by_mkt.items():
        op = parse_open_utc(mkt)
        if op is None:
            continue
        day = op.tz_convert(None).normalize()
        rows.append({
            "market": mkt,
            "pnl_mkt": float(pnl),
            "open_utc": op,
            "open_day": day,
            "hour_utc": op.hour,
            "dow": op.dayofweek,
            "had_fill": mkt in filled_set,
        })

    M = pd.DataFrame(rows)
    if M.empty:
        print("No per-market PnL parsed", file=sys.stderr)
        sys.exit(1)

    d0 = M["open_day"].min().strftime("%Y-%m-%d")
    d1 = (M["open_day"].max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    vix = yf_close_series("^VIX", d0, d1)
    hyg = yf_close_series("HYG", d0, d1)
    btc = yf_close_series("BTC-USD", d0, d1)

    vix_df = vix.rename("vix").reset_index() if vix is not None else None
    if vix_df is not None:
        vix_df.columns = ["open_day", "vix"]
        M = M.merge(vix_df, on="open_day", how="left")

    if hyg is not None:
        h = hyg.rename("hyg_c").reset_index()
        h.columns = ["open_day", "hyg_c"]
        h["hyg_ret"] = h["hyg_c"].pct_change()
        M = M.merge(h[["open_day", "hyg_ret"]], on="open_day", how="left")

    if btc is not None:
        b = btc.rename("btc_c").reset_index()
        b.columns = ["open_day", "btc_c"]
        b["btc_ret"] = b["btc_c"].pct_change()
        M = M.merge(b[["open_day", "btc_ret"]], on="open_day", how="left")

    # VIX terciles on sample (trade days only)
    M["vix_terc"] = pd.qcut(M["vix"].rank(method="first"), 3, labels=["low_vix", "mid_vix", "high_vix"])

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- By hour UTC ---
    by_h = M.groupby("hour_utc").agg(
        n=("market", "count"),
        mean_pnl=("pnl_mkt", "mean"),
        sum_pnl=("pnl_mkt", "sum"),
        median_pnl=("pnl_mkt", "median"),
        win=("pnl_mkt", lambda s: (s > 0).mean()),
    ).reset_index()
    by_h = by_h.sort_values("mean_pnl", ascending=False)
    by_h.to_csv(os.path.join(OUT_DIR, "strat_environment_by_hour.csv"), index=False)

    # --- By dow ---
    dow_n = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_d = M.groupby("dow").agg(
        n=("market", "count"),
        mean_pnl=("pnl_mkt", "mean"),
        sum_pnl=("pnl_mkt", "sum"),
        win=("pnl_mkt", lambda s: (s > 0).mean()),
    ).reset_index()
    by_d["dow_name"] = by_d["dow"].map(dict(enumerate(dow_n)))

    # --- By VIX tercile ---
    by_v = M.groupby("vix_terc", observed=True).agg(
        n=("market", "count"),
        mean_vix=("vix", "mean"),
        mean_pnl=("pnl_mkt", "mean"),
        sum_pnl=("pnl_mkt", "sum"),
        win=("pnl_mkt", lambda s: (s > 0).mean()),
    ).reset_index()

    # --- Daily PnL (markets grouped by open_day UTC) ---
    daily = M.groupby("open_day").agg(
        pnl_day=("pnl_mkt", "sum"),
        n_mkt=("market", "count"),
        vix=("vix", "first"),
    ).reset_index().sort_values("open_day")

    if hyg is not None:
        h2 = hyg.rename("hyg_c").reset_index()
        h2.columns = ["open_day", "hyg_c"]
        h2["hyg_ret"] = h2["hyg_c"].pct_change()
        daily = daily.merge(h2[["open_day", "hyg_ret"]], on="open_day", how="left")

    if btc is not None:
        b2 = btc.rename("btc_c").reset_index()
        b2.columns = ["open_day", "btc_c"]
        b2["btc_ret"] = b2["btc_c"].pct_change()
        daily = daily.merge(b2[["open_day", "btc_ret"]], on="open_day", how="left")

    daily["vix_chg"] = daily["vix"].diff()
    daily["pnl_next"] = daily["pnl_day"].shift(-1)

    # A) Same calendar day macro (close of D) vs NEXT day's total PnL (markets opening D+1).
    #    Caveat: some 15m markets open before US equity close — coarse regime signal only.
    pred_fwd = daily.dropna(subset=["pnl_next"]).copy()
    same_day_cols = ["vix", "vix_chg"]
    if "hyg_ret" in pred_fwd.columns:
        same_day_cols.append("hyg_ret")
    if "btc_ret" in pred_fwd.columns:
        same_day_cols.append("btc_ret")

    corr_rows = []
    for c in same_day_cols:
        if c in pred_fwd.columns:
            sub = pred_fwd[["pnl_next", c]].dropna()
            if len(sub) > 10:
                corr_rows.append((f"{c} (day t) vs pnl day t+1", sub["pnl_next"].corr(sub[c])))

    # B) Strict lag: only prior calendar day info vs TODAY's PnL (no same-day macro).
    pred_strict = daily.copy()
    for name, col in [("vix_lag1", "vix"), ("vix_chg_lag1", "vix_chg")]:
        if col in pred_strict.columns:
            pred_strict[name] = pred_strict[col].shift(1)
    if "hyg_ret" in pred_strict.columns:
        pred_strict["hyg_ret_lag1"] = pred_strict["hyg_ret"].shift(1)
    if "btc_ret" in pred_strict.columns:
        pred_strict["btc_ret_lag1"] = pred_strict["btc_ret"].shift(1)

    strict_feats = [c for c in pred_strict.columns if c.endswith("_lag1")]
    for c in strict_feats:
        sub = pred_strict[["pnl_day", c]].dropna()
        if len(sub) > 10:
            corr_rows.append((f"{c} vs pnl same day t", sub["pnl_day"].corr(sub[c])))

    pred_strict.to_csv(os.path.join(OUT_DIR, "strat_environment_daily_predictors.csv"), index=False)

    # --- Text report ---
    lines = []
    lines.append("STRATEGY ENVIRONMENT REPORT (filled markets only, P&L from exits)")
    lines.append(f"Markets in sample: {len(M)}  |  Date range (open_day UTC): {M['open_day'].min().date()} .. {M['open_day'].max().date()}")
    lines.append("")
    lines.append("=== Best / worst UTC hours (by mean P&L per market) ===")
    lines.append(by_h.head(6).to_string(index=False))
    lines.append("...")
    lines.append(by_h.tail(4).to_string(index=False))
    lines.append("")
    lines.append("=== By weekday (UTC) ===")
    lines.append(by_d.sort_values("mean_pnl", ascending=False).to_string(index=False))
    lines.append("")
    lines.append("=== By same-day VIX tercile (among trade days only) ===")
    lines.append(by_v.to_string(index=False))
    lines.append("")
    lines.append("=== 'Prediction' checks (daily total PnL; UTC calendar days) ===")
    lines.append("A) Same-day macro (t) vs NEXT day PnL (t+1): coarse; equity close may lag some opens.")
    lines.append("B) Prior-day macro (t-1) vs SAME day PnL (t): no same-day lookahead.")
    for c, r in corr_rows:
        lines.append(f"  corr: {c}: {r:+.3f}")
    lines.append("")
    lines.append("INTERPRETATION / CAVEATS")
    lines.append("- These are IN-SAMPLE patterns on one backtest; easy to overfit. Validate with walk-forward or new data.")
    lines.append("- Macro (VIX, HYG, BTC) is DAILY while your edge is 15m microstructure — weak linkage is expected.")
    lines.append("- Strong hours/days may reflect LIQUIDITY / participant mix, not 'VIX predicts Kalshi'.")
    lines.append("- Fill model + compounding size dominate P&L path; regime labels are only coarse filters.")

    txt = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "strat_environment_summary.txt"), "w") as f:
        f.write(txt)

    print(txt)
    print(f"\nWrote: {OUT_DIR}/strat_environment_summary.txt")
    print(f"Wrote: {OUT_DIR}/strat_environment_by_hour.csv")
    print(f"Wrote: {OUT_DIR}/strat_environment_daily_predictors.csv")


if __name__ == "__main__":
    main()
