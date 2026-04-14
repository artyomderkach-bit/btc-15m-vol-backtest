#!/usr/bin/env python3
"""
Deep dive: rolling 200-market windows with less compounding distortion, macro
context for the recent window, and sensitivity to "first-second" fills.

1) Per-market contribution = end_bankroll - start_bankroll (global chronological
   order). Rolling window return = (br_after_window - br_before) / br_before.

2) Same using sum(pnl_mkt) / br_before_window (approx. additive check).

3) Last-200 window: calendar span, VIX/HYG vs full trade-day sample.

4) First-second filter: markets where first buy_fill ts <= open_ts + 1 (from
   cache.db market close_time). Rebuild adjusted bankroll path with those
   markets contributing 0 (as if trades removed).

Outputs:
  charts/btc_main/rolling_window_deep_dive_summary.txt
  charts/btc_main/rolling_window_first_second.csv   (per-market flags)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT)
_PROJECT = os.path.dirname(_ROOT)
TRADELOG = os.path.join(_ROOT, "charts", "btc_main", "current_strat_full_tradelog.csv")
DB_PATH = os.path.join(_PROJECT, "cache.db")
OUT_DIR = os.path.join(_ROOT, "charts", "btc_main")
OUT_SUMMARY = os.path.join(OUT_DIR, "rolling_window_deep_dive_summary.txt")
OUT_PER_MKT = os.path.join(OUT_DIR, "rolling_window_first_second.csv")

WINDOW = 200


def load_open_ts_map() -> Dict[str, int]:
    if not os.path.exists(DB_PATH):
        return {}
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ticker, data FROM markets WHERE series_ticker='KXBTC15M'"
    ).fetchall()
    conn.close()
    out: Dict[str, int] = {}
    for ticker, raw in rows:
        try:
            m = json.loads(raw)
            ct = m.get("close_time")
            if ct is None:
                continue
            if isinstance(ct, str):
                close_ts = int(pd.Timestamp(ct).timestamp())
            else:
                close_ts = int(ct)
            out[ticker] = close_ts - 15 * 60
        except Exception:
            continue
    return out


def fetch_daily_closes(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    h = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    if h is None or h.empty:
        return pd.Series(dtype=float)
    if isinstance(h.columns, pd.MultiIndex):
        h = h.copy()
        h.columns = h.columns.get_level_values(0)
    h = h.reset_index()
    dcol = "Date" if "Date" in h.columns else h.columns[0]
    ccol = "Close" if "Close" in h.columns else h.columns[-1]
    s = pd.to_numeric(h[ccol], errors="coerce")
    idx = pd.to_datetime(h[dcol], utc=True).dt.tz_localize(None).dt.normalize()
    return pd.Series(s.values, index=idx).dropna().sort_index()


def build_market_table(df: pd.DataFrame, open_ts_map: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["bankroll"] = pd.to_numeric(df["bankroll"], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    mm = df.groupby("market")["ts"].transform("max")
    df["plot_ts"] = df["ts"].fillna(mm + 60)
    df = df.sort_values("plot_ts").reset_index(drop=True)

    # Process markets in chronological order of first log row
    order = df.groupby("market")["plot_ts"].min().sort_values().index.tolist()

    rows = []
    for mkt in order:
        g = df.loc[df["market"] == mkt].sort_values("plot_ts")
        pos0 = int(g.index[0])
        start_br = float(df.loc[pos0 - 1, "bankroll"]) if pos0 > 0 else 1000.0
        last = g.iloc[-1]
        end_br = float(last["bankroll"])
        contrib = end_br - start_br

        exit_mask = g["action"].isin(("sell_fill", "settlement", "stop_loss"))
        pnl_exit = float(g.loc[exit_mask, "pnl"].sum())

        buys = g[g["action"] == "buy_fill"]
        first_buy_ts = float(buys["ts"].min()) if len(buys) else np.nan

        ots = open_ts_map.get(mkt)
        # Sub-second timing is not in 1-min candle data; first bar ends at open+60.
        first_sec = False
        fill_first_minute = False
        if ots is not None and np.isfinite(first_buy_ts):
            first_sec = first_buy_ts <= (ots + 1)  # almost never True with 1m bars
            fill_first_minute = first_buy_ts <= (ots + 60)

        if ots:
            tsu = pd.Timestamp.utcfromtimestamp(ots)
            open_day = tsu.tz_convert(None).normalize() if tsu.tzinfo else tsu.normalize()
        else:
            open_day = pd.NaT
        hour_utc = pd.Timestamp.utcfromtimestamp(ots).hour if ots else -1

        rows.append({
            "market": mkt,
            "first_plot_ts": float(g["plot_ts"].iloc[0]),
            "last_plot_ts": float(g["plot_ts"].iloc[-1]),
            "open_ts": int(ots) if ots is not None else None,
            "open_day": open_day,
            "hour_utc": hour_utc,
            "start_br": start_br,
            "end_br": end_br,
            "contrib": contrib,
            "pnl_exit": pnl_exit,
            "first_buy_ts": first_buy_ts,
            "first_second_fill": first_sec,
            "fill_first_minute": fill_first_minute,
            "has_open_ts": ots is not None,
        })

    return pd.DataFrame(rows)


def rolling_window_metrics(M: pd.DataFrame, w: int = WINDOW):
    """Uses actual bankroll bridge: return over window of w markets."""
    n = len(M)
    if n < w:
        return pd.DataFrame(), None

    br_before = np.empty(n)
    br_after = np.empty(n)
    cum = 1000.0
    br_before[0] = 1000.0
    for i in range(n):
        br_before[i] = cum
        cum += M["contrib"].iloc[i]
        br_after[i] = cum

    records = []
    for i in range(0, n - w + 1):
        b0 = br_before[i]
        b1 = br_after[i + w - 1]
        ret = (b1 - b0) / b0 if b0 > 0 else np.nan
        sum_pnl = M["pnl_exit"].iloc[i : i + w].sum()
        approx = sum_pnl / b0 if b0 > 0 else np.nan
        records.append({
            "i0": i,
            "i1": i + w - 1,
            "br_before": b0,
            "br_after": b1,
            "window_ret": ret,
            "sum_pnl_exit": sum_pnl,
            "sum_pnl_over_br0": approx,
            "mean_pnl": M["pnl_exit"].iloc[i : i + w].mean(),
            "win_rate": (M["pnl_exit"].iloc[i : i + w] > 0).mean(),
        })
    R = pd.DataFrame(records)
    last = R.iloc[-1]
    rank_ret = (R["window_ret"] <= last["window_ret"]).sum()
    return R, last, rank_ret


def rebuild_adjusted_path(M: pd.DataFrame, filter_col: str = "fill_first_minute") -> pd.DataFrame:
    """Zero contribution for filtered markets; recompute synthetic end_br."""
    adj = M.copy()
    if filter_col in adj.columns:
        mask = adj[filter_col].fillna(False).to_numpy(dtype=bool)
    else:
        mask = np.zeros(len(adj), dtype=bool)
    adj["contrib_adj"] = np.where(mask, 0.0, adj["contrib"])
    br = 1000.0
    end_adj = []
    for c in adj["contrib_adj"]:
        br += c
        end_adj.append(br)
    adj["end_br_adj"] = end_adj
    adj["cum_contrib_adj"] = adj["contrib_adj"].cumsum()
    return adj


def rolling_on_adjusted(M_adj: pd.DataFrame, w: int = WINDOW):
    n = len(M_adj)
    br_before = np.empty(n)
    cum = 1000.0
    for i in range(n):
        br_before[i] = cum
        cum += M_adj["contrib_adj"].iloc[i]
    records = []
    for i in range(0, n - w + 1):
        b0 = br_before[i]
        b1 = b0 + M_adj["contrib_adj"].iloc[i : i + w].sum()
        ret = (b1 - b0) / b0 if b0 > 0 else np.nan
        records.append({
            "i0": i,
            "window_ret_adj": ret,
            "sum_pnl_exit_window": M_adj["pnl_exit"].iloc[i : i + w].sum(),
        })
    return pd.DataFrame(records)


def main():
    if not os.path.exists(TRADELOG):
        print(f"Missing {TRADELOG}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(TRADELOG)
    open_map = load_open_ts_map()
    M = build_market_table(df, open_map)

    os.makedirs(OUT_DIR, exist_ok=True)
    M.to_csv(OUT_PER_MKT, index=False)

    lines = []
    lines.append("ROLLING WINDOW DEEP DIVE")
    lines.append(f"Markets (chronological by first event): {len(M)}")
    lines.append(f"Window size: {WINDOW}")
    lines.append("")

    # First-second stats
    fs = M["first_second_fill"].fillna(False)
    n_fs = int(fs.sum())
    lines.append("=== Timing filters on first buy_fill (unix ts vs API open_ts) ===")
    lines.append("Note: candles are 1-minute; fills use bar end_ts — 'first second' is rarely observable.")
    lines.append("Practical proxy: fill in first MINUTE after open = first_buy_ts <= open_ts + 60.")
    lines.append(f"Markets with known open_ts: {int(M['has_open_ts'].sum())}")
    lines.append(f"first_second (<= open+1s): {n_fs} ({100*n_fs/len(M):.1f}%)")
    fm = M["fill_first_minute"].fillna(False)
    n_fm = int(fm.sum())
    lines.append(f"fill_first_minute (<= open+60s): {n_fm} ({100*n_fm/len(M):.1f}%)")
    lines.append(f"Exit PnL sum on first-minute fills: ${M.loc[fm, 'pnl_exit'].sum():,.2f}")
    lines.append(f"Exit PnL sum on OTHER markets: ${M.loc[~fm, 'pnl_exit'].sum():,.2f}")
    lines.append(f"Contrib sum (path $ bridge) first-minute: ${M.loc[fm, 'contrib'].sum():,.2f}  |  other: ${M.loc[~fm, 'contrib'].sum():,.2f}")
    last200 = M.iloc[-WINDOW:]
    fm200 = last200["fill_first_minute"].fillna(False)
    lines.append(f"Last-200: first-minute markets = {int(fm200.sum())}, their exit PnL = ${last200.loc[fm200, 'pnl_exit'].sum():,.2f}")
    lines.append(f"Last-200: exit PnL if those removed (sum rest) = ${last200.loc[~fm200, 'pnl_exit'].sum():,.2f}")
    lines.append("")
    lines.append("Note: 'contrib' is end_br−start_br on the realized path; first-minute markets can")
    lines.append("carry large positive $contrib while others sum negative (still totals +$10,205 to final).")
    lines.append("Zeroing contrib for a subset is NOT the same as 'fixing data errors' — it rewires compounding.")
    lines.append("")

    R, last, rank_ret = rolling_window_metrics(M, WINDOW)
    lines.append("=== Rolling WINDOW — bankroll-based return (compounding-aware) ===")
    lines.append("window_ret = (br_after_last_in_window - br_before_first) / br_before_first")
    lines.append(f"Last window: markets index [{int(last['i0'])} .. {int(last['i1'])}]")
    lines.append(f"  window_ret = {last['window_ret']*100:+.2f}%")
    lines.append(f"  sum_pnl_exit / br_before = {last['sum_pnl_over_br0']*100:+.2f}% (approx additive)")
    lines.append(f"  mean pnl/market = ${last['mean_pnl']:+.2f}, win_rate = {last['win_rate']*100:.1f}%")
    lines.append(f"  Rank of last window_ret (1=worst): {rank_ret} / {len(R)}")
    z = (last["window_ret"] - R["window_ret"].mean()) / R["window_ret"].std(ddof=0)
    lines.append(f"  z-score vs all rolling windows: {float(z):+.2f}")
    lines.append("")

    M_adj = rebuild_adjusted_path(M, "fill_first_minute")
    R_adj = rolling_on_adjusted(M_adj, WINDOW)
    if len(R_adj):
        la = R_adj.iloc[-1]
        lines.append("=== Synthetic path: first-MINUTE markets get contrib=0 (illustrative only) ===")
        lines.append("Expect nonsense if those markets were net + on the path (here they are, full sample).")
        lines.append(f"Last window window_ret_adj = {la['window_ret_adj']*100:+.2f}%")
        ra = (R_adj["window_ret_adj"] <= la["window_ret_adj"]).sum()
        lines.append(f"Rank of last window (1=worst): {ra} / {len(R_adj)}")
        lines.append(f"Final synthetic bankroll: ${M_adj['end_br_adj'].iloc[-1]:,.2f}  |  actual: ${M['end_br'].iloc[-1]:,.2f}")
    lines.append("")

    # Last 200 macro
    last_mkts = M.iloc[-WINDOW:]
    d0 = last_mkts["open_day"].min()
    d1 = last_mkts["open_day"].max()
    lines.append("=== Last-200 window — calendar / session ===")
    lines.append(f"Open-day UTC range: {d0} .. {d1}")
    lines.append("Hour UTC (market open) distribution (last 200):")
    hc = last_mkts["hour_utc"].value_counts().sort_index()
    for h, c in hc.items():
        if h < 0:
            continue
        lines.append(f"  {int(h):02d}:00  n={int(c)}")
    lines.append("")

    # Macro vs full (align naive dates to Yahoo index)
    def _naive_days(series) -> pd.DatetimeIndex:
        d = pd.DatetimeIndex(pd.to_datetime(series.dropna(), utc=True))
        return d.tz_localize(None).normalize().unique()

    all_days = _naive_days(M["open_day"])
    last_days = _naive_days(last_mkts["open_day"])
    if len(all_days) and pd.notna(d0):
        start = (pd.Timestamp(all_days.min()) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(all_days.max()) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        vix = fetch_daily_closes("^VIX", start, end)
        hyg = fetch_daily_closes("HYG", start, end)

        def day_stats(days: pd.DatetimeIndex, name: str):
            days = pd.DatetimeIndex(days).sort_values()
            vx = vix.reindex(days).dropna()
            hg_on = hyg.reindex(days).dropna()
            lines.append(f"{name}: n_calendar_days={len(days)}  VIX_obs={len(vx)}  HYG_obs={len(hg_on)}")
            if len(vx):
                lines.append(f"  VIX close mean (on those days) = {vx.mean():.2f}  (median {vx.median():.2f})")
            if len(hg_on):
                lines.append(f"  HYG close mean (on those days) = {hg_on.mean():.2f}")
            d0_, d1_ = days.min(), days.max()
            hspan = hyg.loc[(hyg.index >= d0_) & (hyg.index <= d1_)]
            if len(hspan) >= 2:
                chg = hspan.sort_index().pct_change().dropna()
                lines.append(
                    f"  HYG mean daily pct chg over continuous span {d0_.date()}..{d1_.date()} = "
                    f"{chg.mean()*100:+.4f}%  (std {chg.std()*100:.3f}%, trading days={len(chg)})"
                )

        lines.append("=== Macro on market OPEN days (UTC calendar day) ===")
        day_stats(all_days, "ALL trade days")
        day_stats(last_days, "LAST-200 window open days")
        lines.append("(Compare VIX/HYG lines across rows; last window is a subset of dates.)")

    txt = "\n".join(lines)
    with open(OUT_SUMMARY, "w") as f:
        f.write(txt)
    print(txt)
    print(f"\nWrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_PER_MKT}")


if __name__ == "__main__":
    main()
