#!/usr/bin/env python3
"""
Kalshi BTC weekly probability-basket scanner.

Finds potential buy-all-outcomes arbitrage / near-arbitrage in BTC weekly range buckets
using executable prices derived from the bid-only orderbook.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from data_fetcher import DataFetcher

NY_TZ = ZoneInfo("America/New_York")
EPS = 1e-9


@dataclass
class IntervalSpec:
    low: Optional[float]  # None => -inf
    high: Optional[float]  # None => +inf
    label: str
    parse_warning: str = ""


def _to_unix_ts(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv if iv < 1_000_000_000_000 else iv // 1000
    try:
        return int(datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def _format_expiration(ts: Optional[int]) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=NY_TZ).isoformat()


def _is_friday_5pm_et(ts: Optional[int]) -> bool:
    if not ts:
        return False
    dt = datetime.fromtimestamp(ts, tz=NY_TZ)
    return dt.weekday() == 4 and dt.hour == 17 and dt.minute == 0


def _is_btc_like_text(*parts: str) -> bool:
    hay = " ".join(str(x or "") for x in parts).upper()
    return ("BTC" in hay) or ("BITCOIN" in hay)


def _strike_num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_interval_from_market(market: dict) -> IntervalSpec:
    strike_type = str(market.get("strike_type") or "").lower()
    title = str(market.get("title") or "")
    subtitle = str(market.get("subtitle") or "")
    functional = str(market.get("functional_strike") or "")
    custom = str(market.get("custom_strike") or "")

    floor = _strike_num(market.get("floor_strike"))
    cap = _strike_num(market.get("cap_strike"))

    if strike_type == "between" and floor is not None and cap is not None:
        return IntervalSpec(floor, cap, f"[{floor:g},{cap:g})")
    if strike_type == "greater" and floor is not None:
        return IntervalSpec(floor, None, f"[{floor:g},+inf)")
    if strike_type == "less" and cap is not None:
        return IntervalSpec(None, cap, f"(-inf,{cap:g}]")

    text = " ".join([title, subtitle, functional, custom]).replace(",", "")
    m = re.search(r"between\s+\$?(-?\d+(?:\.\d+)?)\s+and\s+\$?(-?\d+(?:\.\d+)?)", text, re.I)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return IntervalSpec(lo, hi, f"[{lo:g},{hi:g})", "parsed_from_text")

    m = re.search(r"(?:above|over|>\s*=?\s*)\s*\$?(-?\d+(?:\.\d+)?)", text, re.I)
    if m:
        lo = float(m.group(1))
        return IntervalSpec(lo, None, f"[{lo:g},+inf)", "parsed_from_text")

    m = re.search(r"(?:below|under|<\s*=?\s*)\s*\$?(-?\d+(?:\.\d+)?)", text, re.I)
    if m:
        hi = float(m.group(1))
        return IntervalSpec(None, hi, f"(-inf,{hi:g}]", "parsed_from_text")

    return IntervalSpec(None, None, "UNPARSED", "missing_strike_bounds")


def _settlement_signature(market: dict, event_obj: Optional[dict]) -> str:
    parts = [
        str(market.get("event_ticker") or ""),
        str(market.get("series_ticker") or ""),
        str((event_obj or {}).get("settlement_sources") or ""),
    ]
    return "|".join(parts)


def _group_id(market: dict, event_obj: Optional[dict], expiration_ts: int) -> str:
    return "|".join(
        [
            str(market.get("series_ticker") or ""),
            str(market.get("event_ticker") or ""),
            str(expiration_ts),
            _settlement_signature(market, event_obj),
        ]
    )


def _rules_family_key(market: dict) -> str:
    text = " ".join(
        [
            str(market.get("rules_primary") or ""),
            str(market.get("rules_secondary") or ""),
        ]
    ).lower()
    # Remove numeric thresholds and dates so range buckets stay in one family.
    text = re.sub(r"[-+]?\d+(?:\.\d+)?", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _price_to_cents(v) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if fv <= 1.0:
        return fv * 100.0
    return fv


def _best_bid_with_size(levels: Sequence[Sequence]) -> Tuple[Optional[float], float]:
    best_px = None
    best_sz = 0.0
    for lv in levels or []:
        if not lv or len(lv) < 2:
            continue
        px = _price_to_cents(lv[0])
        sz = float(lv[1]) if lv[1] is not None else 0.0
        if px is None:
            continue
        if (best_px is None) or (px > best_px + EPS):
            best_px = px
            best_sz = max(0.0, sz)
        elif abs(px - best_px) <= EPS:
            best_sz += max(0.0, sz)
    return best_px, best_sz


def _exec_quotes_from_orderbook(ob: dict) -> Dict[str, Optional[float]]:
    ob_data = ob.get("orderbook_fp") or ob.get("orderbook") or {}
    yes_levels = ob_data.get("yes_dollars") or ob_data.get("yes") or []
    no_levels = ob_data.get("no_dollars") or ob_data.get("no") or []

    yes_bid, yes_bid_size = _best_bid_with_size(yes_levels)
    no_bid, no_bid_size = _best_bid_with_size(no_levels)

    yes_ask = (100.0 - no_bid) if no_bid is not None else None
    no_ask = (100.0 - yes_bid) if yes_bid is not None else None
    yes_ask_size = no_bid_size if yes_ask is not None else 0.0
    no_ask_size = yes_bid_size if no_ask is not None else 0.0

    return {
        "best_yes_bid": yes_bid,
        "best_no_bid": no_bid,
        "best_yes_ask": yes_ask,
        "best_no_ask": no_ask,
        "best_yes_bid_size": yes_bid_size,
        "best_no_bid_size": no_bid_size,
        "best_yes_ask_size": yes_ask_size,
        "best_no_ask_size": no_ask_size,
    }


def _validate_intervals(intervals: List[IntervalSpec]) -> Tuple[bool, bool, List[str]]:
    warnings: List[str] = []
    if not intervals:
        return False, False, ["empty_group"]
    if any(iv.parse_warning == "missing_strike_bounds" for iv in intervals):
        warnings.append("unparsed_range")
        return False, False, warnings

    def low_key(iv: IntervalSpec) -> float:
        return -math.inf if iv.low is None else iv.low

    sorted_iv = sorted(intervals, key=low_key)

    mutually_exclusive = True
    for a, b in zip(sorted_iv[:-1], sorted_iv[1:]):
        a_hi = math.inf if a.high is None else a.high
        b_lo = -math.inf if b.low is None else b.low
        if b_lo < a_hi - EPS:
            mutually_exclusive = False
            warnings.append("overlap_detected")
            break

    exhaustive = True
    if sorted_iv[0].low is not None:
        exhaustive = False
        warnings.append("missing_left_tail")
    if sorted_iv[-1].high is not None:
        exhaustive = False
        warnings.append("missing_right_tail")
    for a, b in zip(sorted_iv[:-1], sorted_iv[1:]):
        if (a.high is not None) and (b.low is not None) and (b.low > a.high + EPS):
            exhaustive = False
            warnings.append("gap_between_ranges")
            break
    return mutually_exclusive, exhaustive, list(dict.fromkeys(warnings))


def _calc_group_metrics(
    rows: List[dict],
    fee_cents_per_contract: float,
    slippage_cents_per_contract: float,
    near_arb_threshold_cents: float,
    can_assume_single_yes_guarantee: bool,
) -> dict:
    n = len(rows)
    sum_yes_ask = 0.0
    sum_no_ask = 0.0
    yes_depth = math.inf
    no_depth = math.inf
    warnings: List[str] = []
    missing_best_ask = False

    for r in rows:
        ya = r.get("best_yes_ask")
        na = r.get("best_no_ask")
        if ya is None or na is None:
            warnings.append("missing_best_ask")
            missing_best_ask = True
            continue
        sum_yes_ask += float(ya)
        sum_no_ask += float(na)
        yes_depth = min(yes_depth, float(r.get("best_yes_ask_size") or 0.0))
        no_depth = min(no_depth, float(r.get("best_no_ask_size") or 0.0))

    yes_depth = 0.0 if yes_depth is math.inf else yes_depth
    no_depth = 0.0 if no_depth is math.inf else no_depth
    total_per_contract_cost = fee_cents_per_contract + slippage_cents_per_contract

    yes_theoretical = 100.0 - sum_yes_ask
    no_guarantee_eligible = can_assume_single_yes_guarantee and (not missing_best_ask)
    if no_guarantee_eligible:
        no_theoretical = (max(0, n - 1) * 100.0) - sum_no_ask
        no_adjusted = no_theoretical - (n * total_per_contract_cost)
    else:
        no_theoretical = float("nan")
        no_adjusted = float("nan")
        warnings.append("no_basket_guarantee_not_applicable")
    yes_adjusted = yes_theoretical - (n * total_per_contract_cost)

    theoretical_profit = yes_theoretical
    fee_adjusted_profit = yes_adjusted
    if not math.isnan(no_theoretical):
        theoretical_profit = max(theoretical_profit, no_theoretical)
    if not math.isnan(no_adjusted):
        fee_adjusted_profit = max(fee_adjusted_profit, no_adjusted)

    return {
        "sum_yes_ask": sum_yes_ask,
        "sum_no_ask": sum_no_ask,
        "sum_yes_bid": sum(float(r.get("best_yes_bid") or 0.0) for r in rows),
        "number_of_contracts": n,
        "expected_guaranteed_payout_cents": 100.0,  # exactly one bucket YES in valid exhaustive partitions
        "yes_theoretical_profit_cents": yes_theoretical,
        "yes_fee_adjusted_profit_cents": yes_adjusted,
        "no_theoretical_profit_cents": no_theoretical,
        "no_fee_adjusted_profit_cents": no_adjusted,
        "theoretical_profit_cents": theoretical_profit,
        "fee_adjusted_profit_cents": fee_adjusted_profit,
        "liquidity_available_at_best_price": f"yes_min={yes_depth:g};no_min={no_depth:g}",
        "insufficient_depth": yes_depth < 1.0 or no_depth < 1.0,
        "no_guarantee_eligible": no_guarantee_eligible,
        "warnings": warnings,
        "near_arb_threshold_cents": near_arb_threshold_cents,
    }


def _conclusion(
    valid_group: bool,
    exhaustive: bool,
    group_metrics: dict,
    extra_warnings: List[str],
) -> Tuple[str, str]:
    warnings = list(extra_warnings) + list(group_metrics.get("warnings", []))
    if group_metrics.get("insufficient_depth"):
        warnings.append("insufficient_orderbook_depth")

    yes_adj = float(group_metrics.get("yes_fee_adjusted_profit_cents", -1e9))
    no_adj = float(group_metrics.get("no_fee_adjusted_profit_cents", -1e9))
    if math.isnan(no_adj):
        no_adj = -1e9
    near_thr = float(group_metrics.get("near_arb_threshold_cents", 1.0))

    if not valid_group:
        return "INVALID_GROUP", ";".join(sorted(set(warnings)))
    if not exhaustive:
        warnings.append("ranges_not_collectively_exhaustive")

    if exhaustive and (yes_adj > 0 or no_adj > 0):
        return "ARB", ";".join(sorted(set(warnings)))
    if exhaustive and max(yes_adj, no_adj) >= -near_thr:
        return "NEAR_ARB", ";".join(sorted(set(warnings)))
    return "NO_ARB", ";".join(sorted(set(warnings)))


def _fetch_open_events_for_series(client: DataFetcher, series_ticker: str, limit_pages: int = 20) -> Tuple[List[dict], bool]:
    out: List[dict] = []
    seen = set()
    seen_cursors = set()
    cursor = None
    truncated = False
    for _ in range(limit_pages):
        params = {"status": "open", "series_ticker": series_ticker, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = client._api_get("/events", params=params)
        batch = resp.get("events", [])
        for e in batch:
            et = str(e.get("event_ticker") or "")
            if et and et in seen:
                continue
            if et:
                seen.add(et)
            out.append(e)
        cursor = resp.get("cursor")
        if not batch or not cursor:
            break
        if cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
    else:
        truncated = True
    return out, truncated


def main() -> None:
    p = argparse.ArgumentParser(description="Scan BTC weekly basket arbitrage using executable orderbook prices")
    p.add_argument("--fee-cents-per-contract", type=float, default=0.0, help="Estimated fee per contract in cents")
    p.add_argument("--slippage-cents-per-contract", type=float, default=0.25, help="Extra slippage buffer per contract in cents")
    p.add_argument("--near-arb-threshold-cents", type=float, default=1.0, help="Threshold for NEAR_ARB classification")
    p.add_argument("--orderbook-depth", type=int, default=1, help="Orderbook depth to request per market")
    p.add_argument("--range-series-ticker", default="KXBTC", help="BTC range series ticker to scan (default KXBTC)")
    p.add_argument("--max-open-event-pages", type=int, default=20, help="Safety cap on /events pages for the series")
    p.add_argument("--output-csv", default=None, help="Optional output CSV path")
    args = p.parse_args()

    out_csv = args.output_csv
    if not out_csv:
        out_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "outputs",
            "analysis",
            "btc_weekly_probability_basket_scan.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    client = DataFetcher("KXBTC15M")
    try:
        open_events, truncated = _fetch_open_events_for_series(
            client,
            series_ticker=args.range_series_ticker,
            limit_pages=max(1, args.max_open_event_pages),
        )
        print(f"Open events fetched for {args.range_series_ticker}: {len(open_events)}")
        if truncated:
            print(
                f"WARNING: open event scan truncated at {args.max_open_event_pages} pages. "
                "Increase --max-open-event-pages for full coverage."
            )

        # Expand event -> markets, then filter to BTC Friday 5pm range contracts.
        candidates = []
        event_cache: Dict[str, Optional[dict]] = {}
        for e in open_events:
            et = str(e.get("event_ticker") or "")
            if not et:
                continue
            event_resp = client._api_get(f"/events/{et}")
            event_obj = event_resp.get("event") if isinstance(event_resp, dict) else None
            event_markets = event_resp.get("markets", []) if isinstance(event_resp, dict) else []
            event_cache[et] = event_obj
            for m in event_markets:
                close_ts = _to_unix_ts(m.get("close_time"))
                if not _is_friday_5pm_et(close_ts):
                    continue
                if not _is_btc_like_text(
                    m.get("ticker"),
                    m.get("series_ticker") or (event_obj or {}).get("series_ticker"),
                    m.get("event_ticker"),
                    m.get("title"),
                    m.get("subtitle"),
                ):
                    continue
                market_series = str(m.get("series_ticker") or (event_obj or {}).get("series_ticker") or "")
                if market_series != str(args.range_series_ticker):
                    continue
                if m.get("series_ticker") is None and market_series:
                    m = dict(m)
                    m["series_ticker"] = market_series
                candidates.append(m)
        print(f"BTC Friday-5pm candidate markets: {len(candidates)}")

        if not candidates:
            pd.DataFrame(
                columns=[
                    "group_id",
                    "expiration",
                    "market_tickers",
                    "range_labels",
                    "best_yes_bid",
                    "best_yes_ask",
                    "best_no_bid",
                    "best_no_ask",
                    "sum_yes_ask",
                    "sum_no_ask",
                    "theoretical_profit_cents",
                    "fee_adjusted_profit_cents",
                    "missing_range_warning",
                    "liquidity_available_at_best_price",
                    "conclusion",
                ]
            ).to_csv(out_csv, index=False)
            print(f"No BTC weekly candidate markets found. Wrote empty CSV to {out_csv}")
            return

        groups: Dict[str, List[dict]] = {}
        for m in candidates:
            ev = str(m.get("event_ticker") or "")
            event_obj = event_cache.get(ev)
            close_ts = _to_unix_ts(m.get("close_time")) or 0
            gid = _group_id(m, event_obj, close_ts)
            groups.setdefault(gid, []).append(m)

        output_rows: List[dict] = []
        for gid, mkts in groups.items():
            event_obj = event_cache.get(str(mkts[0].get("event_ticker") or ""))
            expiration_ts = _to_unix_ts(mkts[0].get("close_time"))
            intervals = [_parse_interval_from_market(m) for m in mkts]
            mutually_exclusive, exhaustive, iv_warnings = _validate_intervals(intervals)
            rules_warn = []
            if len({_rules_family_key(m) for m in mkts}) > 1:
                rules_warn.append("settlement_rules_differ")

            per_market_rows: List[dict] = []
            for m, iv in zip(mkts, intervals):
                ticker = m.get("ticker")
                try:
                    ob = client._api_get(f"/markets/{ticker}/orderbook", params={"depth": args.orderbook_depth}, retries=2)
                    q = _exec_quotes_from_orderbook(ob)
                except Exception:
                    q = {
                        "best_yes_bid": None,
                        "best_no_bid": None,
                        "best_yes_ask": None,
                        "best_no_ask": None,
                        "best_yes_bid_size": 0.0,
                        "best_no_bid_size": 0.0,
                        "best_yes_ask_size": 0.0,
                        "best_no_ask_size": 0.0,
                    }

                per_market_rows.append(
                    {
                        "ticker": ticker,
                        "interval_label": iv.label,
                        **q,
                    }
                )

            gm = _calc_group_metrics(
                rows=per_market_rows,
                fee_cents_per_contract=args.fee_cents_per_contract,
                slippage_cents_per_contract=args.slippage_cents_per_contract,
                near_arb_threshold_cents=args.near_arb_threshold_cents,
                can_assume_single_yes_guarantee=(mutually_exclusive and exhaustive and len(mkts) >= 2),
            )
            valid_group = mutually_exclusive and len(mkts) >= 2
            conclusion, warning_text = _conclusion(valid_group, exhaustive, gm, iv_warnings + rules_warn)

            output_rows.append(
                {
                    "group_id": gid,
                    "expiration": _format_expiration(expiration_ts),
                    "event_ticker": mkts[0].get("event_ticker"),
                    "series_ticker": mkts[0].get("series_ticker"),
                    "market_tickers": "|".join(str(r["ticker"]) for r in per_market_rows),
                    "range_labels": "|".join(str(r["interval_label"]) for r in per_market_rows),
                    "best_yes_bid": "|".join("" if r["best_yes_bid"] is None else f"{r['best_yes_bid']:.2f}" for r in per_market_rows),
                    "best_yes_ask": "|".join("" if r["best_yes_ask"] is None else f"{r['best_yes_ask']:.2f}" for r in per_market_rows),
                    "best_no_bid": "|".join("" if r["best_no_bid"] is None else f"{r['best_no_bid']:.2f}" for r in per_market_rows),
                    "best_no_ask": "|".join("" if r["best_no_ask"] is None else f"{r['best_no_ask']:.2f}" for r in per_market_rows),
                    "sum_yes_ask": round(gm["sum_yes_ask"], 4),
                    "sum_yes_bid": round(gm["sum_yes_bid"], 4),
                    "sum_no_ask": round(gm["sum_no_ask"], 4),
                    "number_of_contracts": gm["number_of_contracts"],
                    "expected_guaranteed_payout_cents": round(gm["expected_guaranteed_payout_cents"], 4),
                    "yes_theoretical_profit_cents": round(gm["yes_theoretical_profit_cents"], 4),
                    "yes_fee_adjusted_profit_cents": round(gm["yes_fee_adjusted_profit_cents"], 4),
                    "no_theoretical_profit_cents": round(gm["no_theoretical_profit_cents"], 4),
                    "no_fee_adjusted_profit_cents": round(gm["no_fee_adjusted_profit_cents"], 4),
                    "theoretical_profit_cents": round(gm["theoretical_profit_cents"], 4),
                    "fee_adjusted_profit_cents": round(gm["fee_adjusted_profit_cents"], 4),
                    "missing_range_warning": warning_text,
                    "liquidity_available_at_best_price": gm["liquidity_available_at_best_price"],
                    "conclusion": conclusion,
                    "settlement_source": str((event_obj or {}).get("settlement_sources") or ""),
                }
            )

        df = pd.DataFrame(output_rows).sort_values(
            ["conclusion", "fee_adjusted_profit_cents", "theoretical_profit_cents"],
            ascending=[True, False, False],
        )
        df.to_csv(out_csv, index=False)

        print(f"Wrote {len(df)} groups to {out_csv}")
        if not df.empty:
            print(df[["group_id", "expiration", "number_of_contracts", "sum_yes_ask", "sum_no_ask", "fee_adjusted_profit_cents", "conclusion"]].to_string(index=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
