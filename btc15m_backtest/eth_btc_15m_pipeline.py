"""
Kalshi 15-minute ETH (KXETH15M) vs BTC (KXBTC15M) binary up/down pipeline.

Matched on exact (open_ts, close_ts). Builds a 2-second synchronized panel from
tick trades with trade-derived aggressive buy/sell traces (NOT true bid/ask).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_fetcher import DataFetcher
from wti_brent_pipeline import (
    EventAgg,
    aggregate_events,
    flatten_markets_to_csv,
    to_unix_ts,
    trades_to_records,
    trim_trades_records,
    write_trades_parquet,
    _align_index_unix,
)

STEP_2S = 2
EPS = 1e-3


def window_dirname(open_ts: int) -> str:
    dt = datetime.fromtimestamp(open_ts, tz=timezone.utc)
    return f"{dt.year:04d}_{dt.month:02d}_{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}"


def _ffill_after_first(values: List[float]) -> List[float]:
    """Forward-fill only after first non-NaN (use math.nan sentinel)."""
    out = list(values)
    first: Optional[int] = None
    for i, v in enumerate(out):
        if not (isinstance(v, float) and math.isnan(v)):
            first = i
            break
    if first is None:
        return out
    last = math.nan
    for i in range(len(out)):
        v = out[i]
        if not (isinstance(v, float) and math.isnan(v)):
            last = v
            out[i] = v
        elif i >= first:
            out[i] = last
    return out


def derive_aggressive_traces(
    records: Sequence[dict],
    t_lo: int,
    t_hi: int,
    step: int = STEP_2S,
) -> pd.DataFrame:
    """
    Per bin [t, t+step): trade-derived execution traces (NOT order book bid/ask).

    - yes_last: last trade YES price in bin
    - yes_aggressive_buy: max YES price among trades with taker_side == 'yes' (aggressive YES buy)
    - yes_aggressive_sell: min YES price among trades with taker_side == 'no' (aggressive YES sell)
    - vol: sum of contract count in bin
    """
    index_ts = _align_index_unix(t_lo, t_hi, step)
    if not index_ts:
        return pd.DataFrame()
    n = len(index_ts)
    bin_index = {ts: i for i, ts in enumerate(index_ts)}

    last_px = [math.nan] * n
    agg_buy = [math.nan] * n
    agg_sell = [math.nan] * n
    vol = [0.0] * n

    for r in sorted(records, key=lambda x: x["ts"]):
        ts = int(r["ts"])
        if ts < t_lo or ts > t_hi:
            continue
        bstart = (ts // step) * step
        i = bin_index.get(bstart)
        if i is None:
            continue
        yp = float(r["yes_price"])
        last_px[i] = yp
        side = str(r.get("taker_side") or "").lower()
        if side == "yes":
            agg_buy[i] = yp if math.isnan(agg_buy[i]) else max(agg_buy[i], yp)
        elif side == "no":
            agg_sell[i] = yp if math.isnan(agg_sell[i]) else min(agg_sell[i], yp)
        vol[i] += float(r.get("count") or 0.0)

    last_px = _ffill_after_first(last_px)
    agg_buy = _ffill_after_first(agg_buy)
    agg_sell = _ffill_after_first(agg_sell)

    for i in range(n):
        b = agg_buy[i]
        s = agg_sell[i]
        if not math.isnan(b) and not math.isnan(s) and s > b:
            agg_sell[i] = min(s, b)
            agg_buy[i] = max(b, s)

    idx = pd.to_datetime([datetime.fromtimestamp(ts, tz=timezone.utc) for ts in index_ts])
    return pd.DataFrame(
        {
            "yes_last": last_px,
            "yes_aggressive_buy": agg_buy,
            "yes_aggressive_sell": agg_sell,
            "vol": vol,
        },
        index=idx,
    )


def _logit_clipped(p: float) -> float:
    p = max(EPS, min(1.0 - EPS, float(p)))
    return math.log(p / (1.0 - p))


def build_aligned_2s_panel(
    eth_recs: List[dict],
    btc_recs: List[dict],
    t_lo: int,
    t_hi: int,
) -> pd.DataFrame:
    eth_df = derive_aggressive_traces(eth_recs, t_lo, t_hi, STEP_2S)
    btc_df = derive_aggressive_traces(btc_recs, t_lo, t_hi, STEP_2S)
    if eth_df.empty or btc_df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=eth_df.index)
    out["eth_yes_last"] = eth_df["yes_last"].values
    out["eth_yes_aggressive_buy"] = eth_df["yes_aggressive_buy"].values
    out["eth_yes_aggressive_sell"] = eth_df["yes_aggressive_sell"].values
    out["eth_vol"] = eth_df["vol"].values
    out["btc_yes_last"] = btc_df["yes_last"].values
    out["btc_yes_aggressive_buy"] = btc_df["yes_aggressive_buy"].values
    out["btc_yes_aggressive_sell"] = btc_df["yes_aggressive_sell"].values
    out["btc_vol"] = btc_df["vol"].values

    eb = out["eth_yes_last"]
    bb = out["btc_yes_last"]
    out["spread_btcYes_minus_ethYes"] = bb - eb
    out["spread_ethYes_minus_btcYes"] = eb - bb
    out["spread_btcYes_minus_ethNo"] = bb - (1.0 - eb)

    div = []
    for i in range(len(out)):
        e = eb.iloc[i]
        b = bb.iloc[i]
        if pd.isna(e) or pd.isna(b):
            div.append(math.nan)
        else:
            try:
                div.append(_logit_clipped(b) - _logit_clipped(e))
            except (ValueError, ZeroDivisionError):
                div.append(math.nan)
    out["divergence_logit"] = div

    out.index.name = "ts_utc"
    return out


def compute_spread_stats(panel: pd.DataFrame, eth_n_trades: int = 0, btc_n_trades: int = 0) -> dict:
    if panel.empty or "divergence_logit" not in panel.columns:
        return {
            "mean_div": math.nan,
            "std_div": math.nan,
            "max_abs_div": math.nan,
            "max_pos_spread_btcYes_minus_ethYes": math.nan,
            "max_pos_spread_ethYes_minus_btcYes": math.nan,
            "frac_div_above_0_05": math.nan,
            "eth_n_trades": eth_n_trades,
            "btc_n_trades": btc_n_trades,
        }
    d = pd.to_numeric(panel["divergence_logit"], errors="coerce")
    s1 = pd.to_numeric(panel["spread_btcYes_minus_ethYes"], errors="coerce")
    s2 = pd.to_numeric(panel["spread_ethYes_minus_btcYes"], errors="coerce")
    return {
        "mean_div": float(d.mean(skipna=True)) if d.notna().any() else math.nan,
        "std_div": float(d.std(skipna=True)) if d.notna().sum() > 1 else math.nan,
        "max_abs_div": float(d.abs().max(skipna=True)) if d.notna().any() else math.nan,
        "max_pos_spread_btcYes_minus_ethYes": float(s1.max(skipna=True)) if s1.notna().any() else math.nan,
        "max_pos_spread_ethYes_minus_btcYes": float(s2.max(skipna=True)) if s2.notna().any() else math.nan,
        "frac_div_above_0_05": float((d.abs() > 0.05).mean()) if d.notna().any() else math.nan,
        "eth_n_trades": eth_n_trades,
        "btc_n_trades": btc_n_trades,
    }


def overlap_stats_for_window(
    window_utc: str,
    panel: pd.DataFrame,
    t_lo: int,
    t_hi: int,
    min_both_pct: float,
    min_overlap_bins: int,
) -> dict:
    overlap_bins = len(panel) if not panel.empty else max(0, (t_hi - t_lo) // STEP_2S + 1)
    if panel.empty:
        return {
            "window_utc": window_utc,
            "overlap_bins": overlap_bins,
            "eth_bins_with_print": 0,
            "btc_bins_with_print": 0,
            "both_sides_bins_with_print": 0,
            "both_sides_pct": 0.0,
            "ready_for_cointegration": False,
        }
    eth_print = (pd.to_numeric(panel["eth_vol"], errors="coerce").fillna(0) > 0) | panel["eth_yes_last"].notna()
    btc_print = (pd.to_numeric(panel["btc_vol"], errors="coerce").fillna(0) > 0) | panel["btc_yes_last"].notna()
    both = eth_print & btc_print
    n = len(panel)
    both_pct = float(both.sum() / n) if n else 0.0
    ready = both_pct >= min_both_pct and overlap_bins >= min_overlap_bins
    return {
        "window_utc": window_utc,
        "overlap_bins": overlap_bins,
        "eth_bins_with_print": int(eth_print.sum()),
        "btc_bins_with_print": int(btc_print.sum()),
        "both_sides_bins_with_print": int(both.sum()),
        "both_sides_pct": round(both_pct, 6),
        "ready_for_cointegration": bool(ready),
    }


@dataclass
class Matched15m:
    open_ts: int
    close_ts: int
    eth: EventAgg
    btc: EventAgg
    eth_ticker: str
    btc_ticker: str
    match_status: str = "matched"
    reject_reason: str = ""


def match_events_by_window(
    eth_events: Dict[str, EventAgg],
    btc_events: Dict[str, EventAgg],
    max_duration_skew: int = 60,
) -> Tuple[List[Matched15m], List[dict]]:
    def key(ev: EventAgg) -> Tuple[int, int]:
        return (ev.open_ts, ev.close_ts)

    eth_by: Dict[Tuple[int, int], EventAgg] = {}
    for ev in eth_events.values():
        if ev.open_ts and ev.close_ts:
            eth_by[key(ev)] = ev
    btc_by: Dict[Tuple[int, int], EventAgg] = {}
    for ev in btc_events.values():
        if ev.open_ts and ev.close_ts:
            btc_by[key(ev)] = ev

    matched: List[Matched15m] = []
    diag: List[dict] = []

    for k, e_ev in eth_by.items():
        o, c = k
        dur = c - o
        if abs(dur - 900) > max_duration_skew:
            diag.append(
                {
                    "window_utc": datetime.fromtimestamp(o, tz=timezone.utc).isoformat(),
                    "match_status": "rejected_duration_eth",
                    "eth_event": e_ev.event_ticker,
                    "btc_event": "",
                    "open_ts": o,
                    "close_ts": c,
                    "reject_reason": f"duration_{dur}s_not_900s_tol_{max_duration_skew}s",
                }
            )
            continue
        b_ev = btc_by.get(k)
        if b_ev is None:
            diag.append(
                {
                    "window_utc": datetime.fromtimestamp(o, tz=timezone.utc).isoformat(),
                    "match_status": "unmatched_eth",
                    "eth_event": e_ev.event_ticker,
                    "btc_event": "",
                    "open_ts": o,
                    "close_ts": c,
                    "reject_reason": "no_btc_event_same_open_close",
                }
            )
            continue
        bdur = b_ev.close_ts - b_ev.open_ts
        if abs(bdur - 900) > max_duration_skew:
            diag.append(
                {
                    "window_utc": datetime.fromtimestamp(o, tz=timezone.utc).isoformat(),
                    "match_status": "rejected_duration_btc",
                    "eth_event": e_ev.event_ticker,
                    "btc_event": b_ev.event_ticker,
                    "open_ts": o,
                    "close_ts": c,
                    "reject_reason": f"btc_duration_{bdur}s",
                }
            )
            continue
        eth_m = e_ev.markets[0] if e_ev.markets else {}
        btc_m = b_ev.markets[0] if b_ev.markets else {}
        eth_tk = str(eth_m.get("ticker") or "")
        btc_tk = str(btc_m.get("ticker") or "")
        matched.append(
            Matched15m(
                open_ts=o,
                close_ts=c,
                eth=e_ev,
                btc=b_ev,
                eth_ticker=eth_tk,
                btc_ticker=btc_tk,
            )
        )

    for k, b_ev in btc_by.items():
        if k not in eth_by:
            o, c = k
            diag.append(
                {
                    "window_utc": datetime.fromtimestamp(o, tz=timezone.utc).isoformat(),
                    "match_status": "unmatched_btc",
                    "eth_event": "",
                    "btc_event": b_ev.event_ticker,
                    "open_ts": o,
                    "close_ts": c,
                    "reject_reason": "no_eth_event_same_open_close",
                }
            )

    matched.sort(key=lambda m: m.close_ts)
    return matched, diag


def render_recent_grid(
    windows: Sequence[Matched15m],
    panels_root: str,
    charts_dir: str,
    n: int = 9,
) -> List[str]:
    os.makedirs(charts_dir, exist_ok=True)
    n = max(1, int(n))
    wins = list(windows)[-n:] if len(windows) > n else list(windows)
    if not wins:
        return []

    spread_df: Optional[pd.DataFrame] = None
    spread_path = os.path.join(os.path.dirname(panels_root), "diagnostics", "spread_stats.csv")
    if os.path.exists(spread_path):
        spread_df = pd.read_csv(spread_path)

    # Price-only view to fit more windows per page.
    ncols = 4
    page_size = 16
    paths: List[str] = []
    for page_start in range(0, len(wins), page_size):
        chunk = wins[page_start : page_start + page_size]
        ncell = len(chunk)
        nrows = int(math.ceil(ncell / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.9 * nrows), squeeze=False)
        for idx, m in enumerate(chunk):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            pdir = os.path.join(panels_root, window_dirname(m.open_ts))
            fp = os.path.join(pdir, "aligned_2s.parquet")
            fc = os.path.join(pdir, "aligned_2s.csv")
            if os.path.exists(fp):
                df = pd.read_parquet(fp)
            elif os.path.exists(fc):
                df = pd.read_csv(fc, parse_dates=["ts_utc"])
            else:
                ax.set_title(f"{window_dirname(m.open_ts)} (no panel file)")
                ax.axis("off")
                continue
            if "ts_utc" in df.columns:
                df = df.set_index("ts_utc")
            df.index = pd.to_datetime(df.index, utc=True)

            ax.plot(df.index, df["btc_yes_last"], label="BTC yes_last", color="C0", linewidth=0.9)
            ax.plot(df.index, df["eth_yes_last"], label="ETH yes_last", color="C1", linewidth=0.9)
            if "btc_yes_aggressive_buy" in df.columns:
                ax.plot(
                    df.index,
                    df["btc_yes_aggressive_buy"],
                    color="C0",
                    alpha=0.35,
                    linewidth=0.45,
                    label="BTC aggr buy (trade-derived)",
                )
                ax.plot(
                    df.index,
                    df["btc_yes_aggressive_sell"],
                    color="C0",
                    alpha=0.35,
                    linewidth=0.45,
                    linestyle="--",
                    label="BTC aggr sell (trade-derived)",
                )
            if "eth_yes_aggressive_buy" in df.columns:
                ax.plot(
                    df.index,
                    df["eth_yes_aggressive_buy"],
                    color="C1",
                    alpha=0.35,
                    linewidth=0.45,
                    label="ETH aggr buy (trade-derived)",
                )
                ax.plot(
                    df.index,
                    df["eth_yes_aggressive_sell"],
                    color="C1",
                    alpha=0.35,
                    linewidth=0.45,
                    linestyle="--",
                    label="ETH aggr sell (trade-derived)",
                )
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=4.5, loc="upper left")

            wlab = window_dirname(m.open_ts)
            wiso = datetime.fromtimestamp(m.open_ts, tz=timezone.utc).isoformat()
            mean_div = math.nan
            btc_n = 0
            eth_n = 0
            if spread_df is not None and not spread_df.empty and "window_utc" in spread_df.columns:
                sub = spread_df[spread_df["window_utc"] == wiso]
                if not sub.empty:
                    mean_div = float(sub["mean_div"].iloc[0]) if pd.notna(sub["mean_div"].iloc[0]) else math.nan
                    if "btc_n_trades" in sub.columns:
                        btc_n = int(sub["btc_n_trades"].iloc[0])
                    if "eth_n_trades" in sub.columns:
                        eth_n = int(sub["eth_n_trades"].iloc[0])
            if not math.isnan(mean_div):
                ax.set_title(
                    f"{wlab} UTC | btc_n={btc_n} | eth_n={eth_n} | mean_div={mean_div:+.3f}",
                    fontsize=6.5,
                )
            else:
                ax.set_title(f"{wlab} UTC | btc_n={btc_n} | eth_n={eth_n}", fontsize=6.5)

        for j in range(ncell, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")
        plt.suptitle(
            "Kalshi KXBTC15M vs KXETH15M (2s grid): price-only view",
            fontsize=10,
            y=1.01,
        )
        plt.tight_layout()
        page = page_start // page_size + 1
        outp = os.path.join(charts_dir, f"recent_grid_page{page}.png")
        fig.savefig(outp, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(outp)
    return paths


def run_pipeline(
    out_dir: str,
    eth_series: str = "KXETH15M",
    btc_series: str = "KXBTC15M",
    refresh_markets: bool = True,
    market_limit: int = 20000,
    max_matched_events: Optional[int] = None,
    render_chart_n: int = 9,
    min_both_pct: float = 0.5,
    min_overlap_bins: int = 60,
    log: Optional[Callable[[str], None]] = print,
) -> dict:
    _log = log or (lambda *_: None)
    markets_dir = os.path.join(out_dir, "markets")
    trades_dir = os.path.join(out_dir, "trades", "raw")
    panels_root = os.path.join(out_dir, "panels")
    charts_dir = os.path.join(out_dir, "charts")
    diag_dir = os.path.join(out_dir, "diagnostics")
    for d in (markets_dir, trades_dir, panels_root, charts_dir, diag_dir):
        os.makedirs(d, exist_ok=True)

    eth_client = DataFetcher(eth_series)
    btc_client = DataFetcher(btc_series)
    try:
        _log(f"Fetching markets {eth_series} refresh={refresh_markets}...")
        eth_markets = eth_client.fetch_markets(limit=market_limit, refresh=refresh_markets, use_cache_only=False)
        _log(f"Fetching markets {btc_series} refresh={refresh_markets}...")
        btc_markets = btc_client.fetch_markets(limit=market_limit, refresh=refresh_markets, use_cache_only=False)

        flatten_markets_to_csv(eth_markets, os.path.join(markets_dir, "eth_15m_markets.csv"))
        flatten_markets_to_csv(btc_markets, os.path.join(markets_dir, "btc_15m_markets.csv"))

        eth_ev = aggregate_events(eth_series, eth_markets)
        btc_ev = aggregate_events(btc_series, btc_markets)
        matched, diag_unmatched = match_events_by_window(eth_ev, btc_ev)
        if max_matched_events is not None:
            n = max(0, int(max_matched_events))
            matched = matched[-n:] if n else []

        pairs_rows = []
        for m in matched:
            pairs_rows.append(
                {
                    "window_utc": datetime.fromtimestamp(m.open_ts, tz=timezone.utc).isoformat(),
                    "eth_event_ticker": m.eth.event_ticker,
                    "btc_event_ticker": m.btc.event_ticker,
                    "eth_ticker": m.eth_ticker,
                    "btc_ticker": m.btc_ticker,
                    "open_ts": m.open_ts,
                    "close_ts": m.close_ts,
                    "overlap_seconds": m.close_ts - m.open_ts + 1,
                    "match_status": m.match_status,
                }
            )
        pd.DataFrame(pairs_rows).to_csv(os.path.join(markets_dir, "matched_pairs.csv"), index=False)

        trade_rows: List[dict] = []
        missing_rows: List[dict] = []
        overlap_rows: List[dict] = []
        spread_rows: List[dict] = []

        total_windows = len(matched)
        loop_t0 = time.time()

        for idx, m in enumerate(matched, start=1):
            t_lo, t_hi = m.open_ts, m.close_ts
            pdir = os.path.join(panels_root, window_dirname(t_lo))
            os.makedirs(pdir, exist_ok=True)

            eth_recs_all: List[dict] = []
            btc_recs_all: List[dict] = []
            eth_n = 0
            btc_n = 0
            for mk, client, tk, is_eth in (
                (m.eth.markets[0], eth_client, m.eth_ticker, True),
                (m.btc.markets[0], btc_client, m.btc_ticker, False),
            ):
                o_ts = to_unix_ts(mk.get("open_time")) or t_lo
                c_ts = to_unix_ts(mk.get("close_time")) or t_hi
                raw = client.fetch_market_trades(tk, int(o_ts), int(c_ts))
                recs = trades_to_records(raw, tk)
                n_life = len(recs)
                ov = trim_trades_records(recs, t_lo, t_hi)
                n_ov = len(ov)
                write_trades_parquet(recs, os.path.join(trades_dir, f"{tk}.parquet"))
                trade_rows.append(
                    {
                        "ticker": tk,
                        "series": eth_series if is_eth else btc_series,
                        "event_ticker": str(mk.get("event_ticker") or ""),
                        "window_utc": datetime.fromtimestamp(t_lo, tz=timezone.utc).isoformat(),
                        "n_trades_lifetime": n_life,
                        "n_trades_in_overlap": n_ov,
                        "first_trade_ts": recs[0]["ts"] if recs else "",
                        "last_trade_ts": recs[-1]["ts"] if recs else "",
                    }
                )
                if n_life == 0:
                    missing_rows.append({"ticker": tk, "window_utc": datetime.fromtimestamp(t_lo, tz=timezone.utc).isoformat(), "reason": "zero_trades_lifetime"})
                elif n_ov == 0:
                    missing_rows.append({"ticker": tk, "window_utc": datetime.fromtimestamp(t_lo, tz=timezone.utc).isoformat(), "reason": "zero_trades_in_window"})
                if is_eth:
                    eth_recs_all = ov
                    eth_n = n_life
                else:
                    btc_recs_all = ov
                    btc_n = n_life

            panel = build_aligned_2s_panel(eth_recs_all, btc_recs_all, t_lo, t_hi)
            wutc = datetime.fromtimestamp(t_lo, tz=timezone.utc).isoformat()
            if not panel.empty:
                try:
                    panel.to_parquet(os.path.join(pdir, "aligned_2s.parquet"), index=True)
                except Exception:
                    panel.reset_index().to_csv(os.path.join(pdir, "aligned_2s.csv"), index=False)

            st = compute_spread_stats(panel.reset_index(), eth_n_trades=eth_n, btc_n_trades=btc_n)
            spread_rows.append({"window_utc": wutc, **st})
            overlap_rows.append(
                overlap_stats_for_window(
                    wutc,
                    panel,
                    t_lo,
                    t_hi,
                    min_both_pct=min_both_pct,
                    min_overlap_bins=min_overlap_bins,
                )
            )

            elapsed = time.time() - loop_t0
            avg_per = elapsed / idx if idx else 0.0
            remaining = max(0, total_windows - idx)
            eta_sec = int(round(avg_per * remaining))
            eta_min = eta_sec // 60
            eta_rem = eta_sec % 60
            _log(
                f"[progress] {idx}/{total_windows} windows | elapsed={elapsed/60:.1f}m | "
                f"avg={avg_per:.2f}s/window | ETA={eta_min}m{eta_rem:02d}s | "
                f"window={window_dirname(t_lo)}"
            )

        match_summary: List[dict] = []
        for m in matched:
            match_summary.append(
                {
                    "window_utc": datetime.fromtimestamp(m.open_ts, tz=timezone.utc).isoformat(),
                    "eth_event": m.eth.event_ticker,
                    "btc_event": m.btc.event_ticker,
                    "open_ts": m.open_ts,
                    "close_ts": m.close_ts,
                    "overlap_seconds": m.close_ts - m.open_ts + 1,
                    "match_status": "matched",
                }
            )
        for row in diag_unmatched:
            match_summary.append(
                {
                    "window_utc": row.get("window_utc"),
                    "eth_event": row.get("eth_event", ""),
                    "btc_event": row.get("btc_event", ""),
                    "open_ts": row.get("open_ts", ""),
                    "close_ts": row.get("close_ts", ""),
                    "overlap_seconds": "",
                    "match_status": row.get("match_status"),
                    "reject_reason": row.get("reject_reason", ""),
                }
            )

        pd.DataFrame(match_summary).to_csv(os.path.join(diag_dir, "match_summary.csv"), index=False)
        pd.DataFrame(trade_rows).to_csv(os.path.join(diag_dir, "trade_counts.csv"), index=False)
        pd.DataFrame(missing_rows).to_csv(os.path.join(diag_dir, "missing_or_empty.csv"), index=False)
        pd.DataFrame(overlap_rows).to_csv(os.path.join(diag_dir, "overlap_stats.csv"), index=False)
        pd.DataFrame(spread_rows).to_csv(os.path.join(diag_dir, "spread_stats.csv"), index=False)

        render_recent_grid(matched, panels_root, charts_dir, n=render_chart_n)

        _log(f"Done. Matched windows: {len(matched)}. Output: {out_dir}")
        return {"out_dir": out_dir, "n_matched": len(matched), "matched_pairs": os.path.join(markets_dir, "matched_pairs.csv")}
    finally:
        eth_client.close()
        btc_client.close()
