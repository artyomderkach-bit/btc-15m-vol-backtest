"""
Kalshi WTI (KXWTIW) vs Brent (KXBRENTW) weekly pipeline helpers.

Fetches markets/trades via DataFetcher, matches events by same Friday (UTC close date),
builds overlap-trimmed per-strike and implied-price panels.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_fetcher import DataFetcher, _trade_created_unix, _trade_yes_price_dollars


def to_unix_ts(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv if iv < 1_000_000_000_000 else iv // 1000
    try:
        return int(datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def close_date_utc(m: dict) -> Optional[date]:
    ct = to_unix_ts(m.get("close_time"))
    if ct is None:
        return None
    return datetime.fromtimestamp(ct, tz=timezone.utc).date()


def freq_seconds(freq_label: str) -> int:
    m = {"1m": 60, "5m": 300, "15m": 900}.get(freq_label.strip().lower())
    if m is None:
        raise ValueError(f"Unsupported freq {freq_label!r}; use 1m, 5m, 15m")
    return m


def flatten_markets_to_csv(markets: Sequence[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not markets:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.json_normalize(list(markets))
    df.to_csv(path, index=False)


def group_markets_by_event(markets: Iterable[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for m in markets:
        et = str(m.get("event_ticker") or "")
        if not et:
            continue
        out.setdefault(et, []).append(m)
    return out


@dataclass
class EventAgg:
    event_ticker: str
    series_ticker: str
    markets: List[dict]
    close_ts: int = 0
    open_ts: int = 0
    friday_utc: Optional[date] = None

    def __post_init__(self) -> None:
        opens: List[int] = []
        closes: List[int] = []
        for m in self.markets:
            o = to_unix_ts(m.get("open_time"))
            c = to_unix_ts(m.get("close_time"))
            if o is not None:
                opens.append(o)
            if c is not None:
                closes.append(c)
        self.open_ts = min(opens) if opens else 0
        self.close_ts = max(closes) if closes else 0
        self.friday_utc = close_date_utc(self.markets[0]) if self.markets else None


def aggregate_events(series_ticker: str, markets: Sequence[dict]) -> Dict[str, EventAgg]:
    grouped = group_markets_by_event(markets)
    return {
        et: EventAgg(event_ticker=et, series_ticker=series_ticker, markets=mks)
        for et, mks in grouped.items()
    }


@dataclass
class MatchedPair:
    friday_utc: date
    wti: EventAgg
    brent: EventAgg
    t_lo: int
    t_hi: int
    match_status: str = "matched"
    reject_reason: str = ""


def match_events_by_friday(
    wti_events: Dict[str, EventAgg],
    brent_events: Dict[str, EventAgg],
    max_close_delta_seconds: int = 86400,
) -> Tuple[List[MatchedPair], List[dict]]:
    """
    Match WTI and Brent events whose market close_time dates (UTC) fall on the same Friday.
    Also emits unmatched rows for diagnostics.
    """
    wti_by_friday: Dict[date, EventAgg] = {}
    for ev in wti_events.values():
        if ev.friday_utc is None:
            continue
        # If duplicate Friday (shouldn't happen), keep latest close_ts
        prev = wti_by_friday.get(ev.friday_utc)
        if prev is None or ev.close_ts > prev.close_ts:
            wti_by_friday[ev.friday_utc] = ev

    brent_by_friday: Dict[date, EventAgg] = {}
    for ev in brent_events.values():
        if ev.friday_utc is None:
            continue
        prev = brent_by_friday.get(ev.friday_utc)
        if prev is None or ev.close_ts > prev.close_ts:
            brent_by_friday[ev.friday_utc] = ev

    matched: List[MatchedPair] = []
    diag_rows: List[dict] = []

    for fr, w in wti_by_friday.items():
        b = brent_by_friday.get(fr)
        if b is None:
            diag_rows.append(
                {
                    "friday_utc": fr.isoformat(),
                    "match_status": "unmatched_wti",
                    "wti_event": w.event_ticker,
                    "brent_event": "",
                    "wti_close_ts": w.close_ts,
                    "brent_close_ts": "",
                    "reject_reason": "no_brent_event_same_friday",
                }
            )
            continue
        delta = abs(w.close_ts - b.close_ts)
        if delta > max_close_delta_seconds:
            diag_rows.append(
                {
                    "friday_utc": fr.isoformat(),
                    "match_status": "rejected_close_delta",
                    "wti_event": w.event_ticker,
                    "brent_event": b.event_ticker,
                    "wti_close_ts": w.close_ts,
                    "brent_close_ts": b.close_ts,
                    "reject_reason": f"close_ts_delta_{delta}s_exceeds_{max_close_delta_seconds}",
                }
            )
            continue
        t_lo = max(w.open_ts, b.open_ts)
        t_hi = min(w.close_ts, b.close_ts)
        if t_lo > t_hi:
            diag_rows.append(
                {
                    "friday_utc": fr.isoformat(),
                    "match_status": "rejected_no_overlap",
                    "wti_event": w.event_ticker,
                    "brent_event": b.event_ticker,
                    "wti_open_ts": w.open_ts,
                    "wti_close_ts": w.close_ts,
                    "brent_open_ts": b.open_ts,
                    "brent_close_ts": b.close_ts,
                    "reject_reason": "open_close_windows_do_not_overlap",
                }
            )
            continue
        matched.append(MatchedPair(friday_utc=fr, wti=w, brent=b, t_lo=t_lo, t_hi=t_hi))

    for fr, b in brent_by_friday.items():
        if fr not in wti_by_friday:
            diag_rows.append(
                {
                    "friday_utc": fr.isoformat(),
                    "match_status": "unmatched_brent",
                    "wti_event": "",
                    "brent_event": b.event_ticker,
                    "wti_close_ts": "",
                    "brent_close_ts": b.close_ts,
                    "reject_reason": "no_wti_event_same_friday",
                }
            )

    matched.sort(key=lambda p: p.friday_utc)
    return matched, diag_rows


def trades_to_records(trades: Sequence[dict], ticker: str) -> List[dict]:
    out: List[dict] = []
    for tr in trades:
        ts = _trade_created_unix(tr)
        if ts is None:
            continue
        yp = _trade_yes_price_dollars(tr)
        if yp is None:
            continue
        try:
            cnt = float(tr.get("count_fp") or tr.get("count") or 0)
        except (TypeError, ValueError):
            cnt = 0.0
        tid = tr.get("trade_id") or tr.get("id") or ""
        out.append(
            {
                "ticker": ticker,
                "trade_id": str(tid) if tid is not None else "",
                "ts": int(ts),
                "yes_price": float(yp),
                "count": cnt,
                "taker_side": str(tr.get("taker_side") or ""),
                "raw_json": json.dumps(tr, separators=(",", ":")),
            }
        )
    out.sort(key=lambda r: r["ts"])
    return out


def trim_trades_records(records: List[dict], t_lo: int, t_hi: int) -> List[dict]:
    return [r for r in records if t_lo <= r["ts"] <= t_hi]


def write_trades_parquet(records: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["ticker", "trade_id", "ts", "yes_price", "count", "taker_side", "raw_json"])
    try:
        df.to_parquet(path, index=False)
    except Exception:
        csv_path = path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)


def _align_index_unix(t_lo: int, t_hi: int, step: int) -> List[int]:
    """Left edges of bins of width ``step`` that overlap [t_lo, t_hi] (inclusive)."""
    start = (t_lo // step) * step
    out: List[int] = []
    t = start
    while t <= t_hi:
        if t <= t_hi and t + step > t_lo:
            out.append(t)
        t += step
    return out


def build_per_strike_panel(
    trades_by_ticker: Dict[str, List[dict]],
    t_lo: int,
    t_hi: int,
    freq_label: str,
) -> pd.DataFrame:
    """
    Wide panel: index UTC minute (or bin left edge), one column per ticker (last YES in bin),
    plus <ticker>_vol for summed contract count in bin. Forward-fill each ticker only after
    its first non-null print inside the window.
    """
    step = freq_seconds(freq_label)
    index_ts = _align_index_unix(t_lo, t_hi, step)
    if not index_ts:
        return pd.DataFrame()

    idx_dt = pd.to_datetime([datetime.fromtimestamp(ts, tz=timezone.utc) for ts in index_ts])
    bin_index = {ts: i for i, ts in enumerate(index_ts)}
    col_data: Dict[str, List[float]] = {}

    for ticker, recs in trades_by_ticker.items():
        n = len(index_ts)
        last_px: List[float] = [math.nan] * n
        vol_acc: List[float] = [0.0] * n
        if recs:
            for r in sorted(recs, key=lambda x: x["ts"]):
                ts = int(r["ts"])
                bstart = (ts // step) * step
                i = bin_index.get(bstart)
                if i is None:
                    continue
                last_px[i] = float(r["yes_price"])
                vol_acc[i] += float(r.get("count") or 0.0)
            first_j: Optional[int] = None
            for j, v in enumerate(last_px):
                if not math.isnan(v):
                    first_j = j
                    break
            if first_j is not None:
                running = math.nan
                for j in range(n):
                    if not math.isnan(last_px[j]):
                        running = last_px[j]
                    elif j >= first_j and not math.isnan(running):
                        last_px[j] = running
        col_data[ticker] = last_px
        col_data[f"{ticker}_vol"] = vol_acc

    panel = pd.DataFrame(col_data, index=idx_dt)
    panel.index.name = "ts_utc"
    return panel


def _strike_num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class StrikeSpec:
    ticker: str
    strike_type: str
    floor: Optional[float] = None
    cap: Optional[float] = None
    mid: Optional[float] = None
    k_greater: Optional[float] = None  # P(S > k)
    k_less: Optional[float] = None  # P(S < k) approx from cap

    @staticmethod
    def from_market(m: dict) -> "StrikeSpec":
        st = str(m.get("strike_type") or "").lower()
        floor = _strike_num(m.get("floor_strike"))
        cap = _strike_num(m.get("cap_strike"))
        tk = str(m.get("ticker") or "")
        mid = None
        if st == "between" and floor is not None and cap is not None:
            mid = (floor + cap) / 2.0
        return StrikeSpec(
            ticker=tk,
            strike_type=st,
            floor=floor,
            cap=cap,
            mid=mid,
            k_greater=floor if st == "greater" else None,
            k_less=cap if st == "less" else None,
        )


def implied_price_from_row(
    row: pd.Series,
    specs: Sequence[StrikeSpec],
) -> Tuple[float, int]:
    """
    Returns (implied_usd_or_nan, n_rungs_with_print) using:
    - between-only: normalize YES weights, median of discrete distribution on mids
    - greater-only (typical Brent): interpolate K where P(S>K) crosses 0.5
    - mixed (typical WTI): use between ladder if any; else greater/less interpolation on tail strikes
    """
    betweens = [s for s in specs if s.strike_type == "between" and s.mid is not None]
    greaters = [s for s in specs if s.strike_type == "greater" and s.k_greater is not None]
    lesses = [s for s in specs if s.strike_type == "less" and s.k_less is not None]

    def yes_for(ticker: str) -> Optional[float]:
        if ticker not in row.index:
            return None
        v = row[ticker]
        if pd.isna(v):
            return None
        try:
            y = float(v)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, y))

    # 1) Between ladder
    mids: List[float] = []
    weights: List[float] = []
    for s in sorted(betweens, key=lambda x: x.floor or 0.0):
        y = yes_for(s.ticker)
        if y is None:
            continue
        mids.append(float(s.mid or 0.0))
        weights.append(y)
    if mids and sum(weights) > 1e-12:
        w = [wi / sum(weights) for wi in weights]
        order = sorted(range(len(mids)), key=lambda i: mids[i])
        cdf = 0.0
        med_val = mids[order[0]]
        for i in order:
            cdf += w[i]
            if cdf >= 0.5:
                med_val = mids[i]
                break
        n_print = len(weights)
        return float(med_val), n_print

    # 2) Greater-only: P(S>K) decreases in K; interpolate where P crosses 0.5
    pts: List[Tuple[float, float]] = []
    for s in greaters:
        y = yes_for(s.ticker)
        if y is None:
            continue
        pts.append((float(s.k_greater or 0.0), float(y)))
    if len(pts) >= 1:
        pts.sort(key=lambda x: x[0])
        # Enforce non-increasing P as K increases
        for i in range(1, len(pts)):
            k, p = pts[i]
            if p > pts[i - 1][1]:
                pts[i] = (k, min(p, pts[i - 1][1]))
    if len(pts) >= 2:
        if pts[0][1] < 0.5:
            return float(pts[0][0]), len(pts)
        if pts[-1][1] > 0.5:
            return float(pts[-1][0]), len(pts)
        for i in range(len(pts) - 1):
            k1, p1 = pts[i]
            k2, p2 = pts[i + 1]
            if p1 >= 0.5 >= p2:
                if abs(p2 - p1) < 1e-12:
                    return float((k1 + k2) / 2.0), len(pts)
                t = (0.5 - p1) / (p2 - p1)
                return float(k1 + t * (k2 - k1)), len(pts)

    if len(pts) == 1:
        return float(pts[0][0]), 1

    # 3) Less-only: P(S < cap) = y; approximate median cap where P crosses 0.5
    pts_l: List[Tuple[float, float]] = []
    for s in lesses:
        y = yes_for(s.ticker)
        if y is None or s.k_less is None:
            continue
        pts_l.append((float(s.k_less), float(y)))
    if len(pts_l) >= 2:
        pts_l.sort(key=lambda x: x[0])
        for i in range(1, len(pts_l)):
            k, p = pts_l[i]
            if p < pts_l[i - 1][1]:
                pts_l[i] = (k, max(p, pts_l[i - 1][1]))
        for i in range(len(pts_l) - 1):
            k1, p1 = pts_l[i]
            k2, p2 = pts_l[i + 1]
            if p1 <= 0.5 <= p2:
                if abs(p2 - p1) < 1e-12:
                    return float((k1 + k2) / 2.0), len(pts_l)
                t = (0.5 - p1) / (p2 - p1)
                return float(k1 + t * (k2 - k1)), len(pts_l)
    if len(pts_l) == 1:
        return float(pts_l[0][0]), 1

    return float("nan"), 0


def build_implied_price_panel(
    per_strike: pd.DataFrame,
    wti_specs: Sequence[StrikeSpec],
    brent_specs: Sequence[StrikeSpec],
) -> pd.DataFrame:
    # Only price columns present in panel
    wti_specs_f = [s for s in wti_specs if s.ticker in per_strike.columns]
    brent_specs_f = [s for s in brent_specs if s.ticker in per_strike.columns]

    rows: List[dict] = []
    for ts, row in per_strike.iterrows():
        wti_val, wti_n = implied_price_from_row(row, wti_specs_f)
        br_val, br_n = implied_price_from_row(row, brent_specs_f)
        rows.append(
            {
                "ts_utc": ts,
                "wti_implied_usd": wti_val,
                "brent_implied_usd": br_val,
                "wti_ladder_rungs_with_print": wti_n,
                "brent_ladder_rungs_with_print": br_n,
                "both_sides_have_print": (wti_n > 0) and (br_n > 0)
                and (not math.isnan(wti_val))
                and (not math.isnan(br_val)),
            }
        )
    return pd.DataFrame(rows)


def overlap_stats_for_friday(
    friday: date,
    per_strike_1m: pd.DataFrame,
    implied_1m: pd.DataFrame,
    t_lo: int,
    t_hi: int,
    min_both_pct: float = 0.5,
    min_overlap_minutes: int = 60,
) -> dict:
    overlap_minutes = len(per_strike_1m) if not per_strike_1m.empty else max(
        0, int((t_hi - t_lo) // 60) + (1 if (t_hi - t_lo) % 60 else 0)
    )
    if per_strike_1m.empty:
        return {
            "friday_utc": friday.isoformat(),
            "overlap_minutes": max(
                0, int((t_hi - t_lo) // 60) + (1 if (t_hi - t_lo) % 60 else 0)
            ),
            "wti_minutes_with_print": 0,
            "brent_minutes_with_print": 0,
            "both_sides_minutes_with_print": 0,
            "both_sides_pct": 0.0,
            "wti_implied_minutes": 0,
            "brent_implied_minutes": 0,
            "ready_for_cointegration": False,
        }

    # Heuristic: any column not ending with _vol and containing series prefix in name — split by first market list
    vol_cols = [c for c in per_strike_1m.columns if str(c).endswith("_vol")]
    px_cols = [c for c in per_strike_1m.columns if c not in vol_cols]

    wti_cols = [c for c in px_cols if str(c).startswith("KXWTIW")]
    br_cols = [c for c in px_cols if str(c).startswith("KXBRENTW")]

    wti_vol_cols = [f"{c}_vol" for c in wti_cols if f"{c}_vol" in per_strike_1m.columns]
    br_vol_cols = [f"{c}_vol" for c in br_cols if f"{c}_vol" in per_strike_1m.columns]
    wti_print = (per_strike_1m[wti_vol_cols].sum(axis=1) > 0) if wti_vol_cols else pd.Series(False, index=per_strike_1m.index)
    br_print = (per_strike_1m[br_vol_cols].sum(axis=1) > 0) if br_vol_cols else pd.Series(False, index=per_strike_1m.index)
    wti_min = int(wti_print.sum())
    br_min = int(br_print.sum())
    both_mask = wti_print & br_print
    both_min = int(both_mask.sum())
    n = len(per_strike_1m)
    both_pct = (both_min / n) if n else 0.0

    def _finite_series(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        return x.notna() & np.isfinite(x.to_numpy(dtype=float))

    wti_imp = int(_finite_series(implied_1m["wti_implied_usd"]).sum()) if not implied_1m.empty else 0
    br_imp = int(_finite_series(implied_1m["brent_implied_usd"]).sum()) if not implied_1m.empty else 0

    ready = both_pct >= min_both_pct and overlap_minutes >= min_overlap_minutes
    return {
        "friday_utc": friday.isoformat(),
        "overlap_minutes": overlap_minutes,
        "wti_minutes_with_print": wti_min,
        "brent_minutes_with_print": br_min,
        "both_sides_minutes_with_print": both_min,
        "both_sides_pct": round(both_pct, 6),
        "wti_implied_minutes": wti_imp,
        "brent_implied_minutes": br_imp,
        "ready_for_cointegration": bool(ready),
    }


def friday_dirname(friday: date) -> str:
    return f"{friday.year:04d}_{friday.month:02d}_{friday.day:02d}"


def run_pipeline(
    out_dir: str,
    wti_series: str = "KXWTIW",
    brent_series: str = "KXBRENTW",
    refresh_markets: bool = True,
    market_limit: int = 5000,
    max_matched_events: Optional[int] = None,
    freqs: Sequence[str] = ("1m", "5m", "15m"),
    min_both_pct: float = 0.5,
    min_overlap_minutes: int = 60,
    log: Optional[Callable[[str], None]] = print,
) -> dict:
    _log = log or (lambda *_: None)

    markets_dir = os.path.join(out_dir, "markets")
    trades_dir = os.path.join(out_dir, "trades", "raw")
    diag_dir = os.path.join(out_dir, "diagnostics")
    panels_root = os.path.join(out_dir, "panels")
    for d in (markets_dir, trades_dir, diag_dir, panels_root):
        os.makedirs(d, exist_ok=True)

    wti_path = os.path.join(markets_dir, "wti_weekly_markets.csv")
    brent_path = os.path.join(markets_dir, "brent_weekly_markets.csv")
    matched_pairs_path = os.path.join(markets_dir, "matched_pairs.csv")

    wti_client = DataFetcher(wti_series)
    brent_client = DataFetcher(brent_series)
    try:
        _log(f"Fetching markets for {wti_series} (refresh={refresh_markets})...")
        wti_markets = wti_client.fetch_markets(limit=market_limit, refresh=refresh_markets, use_cache_only=False)
        _log(f"Fetching markets for {brent_series} (refresh={refresh_markets})...")
        brent_markets = brent_client.fetch_markets(limit=market_limit, refresh=refresh_markets, use_cache_only=False)

        flatten_markets_to_csv(wti_markets, wti_path)
        flatten_markets_to_csv(brent_markets, brent_path)

        wti_ev = aggregate_events(wti_series, wti_markets)
        br_ev = aggregate_events(brent_series, brent_markets)
        matched, diag_unmatched = match_events_by_friday(wti_ev, br_ev)
        if max_matched_events is not None:
            n = max(0, int(max_matched_events))
            matched = matched[-n:] if n else []

        pairs_rows = []
        for p in matched:
            pairs_rows.append(
                {
                    "friday_utc": p.friday_utc.isoformat(),
                    "wti_event_ticker": p.wti.event_ticker,
                    "brent_event_ticker": p.brent.event_ticker,
                    "wti_open_ts": p.wti.open_ts,
                    "wti_close_ts": p.wti.close_ts,
                    "brent_open_ts": p.brent.open_ts,
                    "brent_close_ts": p.brent.close_ts,
                    "overlap_t_lo": p.t_lo,
                    "overlap_t_hi": p.t_hi,
                    "overlap_seconds": p.t_hi - p.t_lo + 1,
                    "match_status": p.match_status,
                }
            )
        pd.DataFrame(pairs_rows).to_csv(matched_pairs_path, index=False)

        trade_count_rows: List[dict] = []
        missing_rows: List[dict] = []
        overlap_rows: List[dict] = []

        for pair in matched:
            fdir = os.path.join(panels_root, friday_dirname(pair.friday_utc))
            os.makedirs(fdir, exist_ok=True)

            trades_by_ticker: Dict[str, List[dict]] = {}
            for m in pair.wti.markets + pair.brent.markets:
                tk = str(m.get("ticker") or "")
                if not tk:
                    continue
                series = str(
                    m.get("series_ticker")
                    or (wti_series if tk.startswith(wti_series) else brent_series)
                )
                is_wti = series == wti_series or tk.startswith(wti_series + "-")
                ev_open = pair.wti.open_ts if is_wti else pair.brent.open_ts
                ev_close = pair.wti.close_ts if is_wti else pair.brent.close_ts
                o_ts = to_unix_ts(m.get("open_time")) or ev_open
                c_ts = to_unix_ts(m.get("close_time")) or ev_close
                client = wti_client if is_wti else brent_client
                raw_tr = client.fetch_market_trades(tk, int(o_ts), int(c_ts))
                recs = trades_to_records(raw_tr, tk)
                n_life = len(recs)
                overlap_recs = trim_trades_records(recs, pair.t_lo, pair.t_hi)
                n_ov = len(overlap_recs)
                trades_by_ticker[tk] = overlap_recs

                trade_count_rows.append(
                    {
                        "ticker": tk,
                        "series": series,
                        "event_ticker": str(m.get("event_ticker") or ""),
                        "friday_utc": pair.friday_utc.isoformat(),
                        "n_trades_lifetime": n_life,
                        "n_trades_in_overlap": n_ov,
                        "first_trade_ts": recs[0]["ts"] if recs else "",
                        "last_trade_ts": recs[-1]["ts"] if recs else "",
                    }
                )
                if n_life == 0:
                    missing_rows.append(
                        {
                            "ticker": tk,
                            "friday_utc": pair.friday_utc.isoformat(),
                            "reason": "zero_trades_lifetime",
                        }
                    )
                elif n_ov == 0:
                    missing_rows.append(
                        {
                            "ticker": tk,
                            "friday_utc": pair.friday_utc.isoformat(),
                            "reason": "zero_trades_in_overlap_window",
                        }
                    )

                # Save raw lifetime parquet (plan: raw tick-level per ticker)
                write_trades_parquet(recs, os.path.join(trades_dir, f"{tk}.parquet"))

            wti_specs = [StrikeSpec.from_market(m) for m in pair.wti.markets]
            brent_specs = [StrikeSpec.from_market(m) for m in pair.brent.markets]

            per_strike_1m = pd.DataFrame()
            implied_1m = pd.DataFrame()
            for fl in freqs:
                panel = build_per_strike_panel(trades_by_ticker, pair.t_lo, pair.t_hi, fl)
                out_p = os.path.join(fdir, f"per_strike_{fl}.parquet")
                try:
                    panel.to_parquet(out_p, index=True)
                except Exception:
                    panel.to_csv(out_p.replace(".parquet", ".csv"), index=True)
                implied = build_implied_price_panel(panel, wti_specs, brent_specs)
                if fl == "1m":
                    per_strike_1m = panel
                    implied_1m = implied
                out_i = os.path.join(fdir, f"implied_price_{fl}.parquet")
                try:
                    implied.to_parquet(out_i, index=False)
                except Exception:
                    implied.to_csv(out_i.replace(".parquet", ".csv"), index=False)

            overlap_rows.append(
                overlap_stats_for_friday(
                    pair.friday_utc,
                    per_strike_1m,
                    implied_1m,
                    pair.t_lo,
                    pair.t_hi,
                    min_both_pct=min_both_pct,
                    min_overlap_minutes=min_overlap_minutes,
                )
            )

        match_summary_rows: List[dict] = []
        for p in matched:
            match_summary_rows.append(
                {
                    "friday_utc": p.friday_utc.isoformat(),
                    "wti_event": p.wti.event_ticker,
                    "brent_event": p.brent.event_ticker,
                    "wti_close_utc": datetime.fromtimestamp(p.wti.close_ts, tz=timezone.utc).isoformat(),
                    "brent_close_utc": datetime.fromtimestamp(p.brent.close_ts, tz=timezone.utc).isoformat(),
                    "t_lo": p.t_lo,
                    "t_hi": p.t_hi,
                    "overlap_seconds": p.t_hi - p.t_lo + 1,
                    "match_status": "matched",
                }
            )
        for row in diag_unmatched:
            match_summary_rows.append(
                {
                    "friday_utc": row.get("friday_utc"),
                    "wti_event": row.get("wti_event"),
                    "brent_event": row.get("brent_event"),
                    "wti_close_utc": row.get("wti_close_ts", ""),
                    "brent_close_utc": row.get("brent_close_ts", ""),
                    "t_lo": "",
                    "t_hi": "",
                    "overlap_seconds": "",
                    "match_status": row.get("match_status"),
                    "reject_reason": row.get("reject_reason", ""),
                }
            )
        pd.DataFrame(match_summary_rows).to_csv(os.path.join(diag_dir, "match_summary.csv"), index=False)
        pd.DataFrame(trade_count_rows).to_csv(os.path.join(diag_dir, "trade_counts.csv"), index=False)
        pd.DataFrame(missing_rows).to_csv(os.path.join(diag_dir, "missing_or_empty.csv"), index=False)
        pd.DataFrame(overlap_rows).to_csv(os.path.join(diag_dir, "overlap_stats.csv"), index=False)

        _log(f"Wrote markets to {wti_path} and {brent_path}")
        _log(f"Matched pairs: {len(matched)} -> {matched_pairs_path}")
        _log(f"Diagnostics -> {diag_dir}")

        return {
            "out_dir": out_dir,
            "n_wti_markets": len(wti_markets),
            "n_brent_markets": len(brent_markets),
            "n_matched_fridays": len(matched),
            "matched_pairs_csv": matched_pairs_path,
        }
    finally:
        wti_client.close()
        brent_client.close()
