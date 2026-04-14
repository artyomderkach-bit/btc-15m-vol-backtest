"""
Conservative grouping of Kalshi markets into ordered threshold ladders for monotonicity checks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Re-export for tests
__all__ = [
    "ThresholdDirection",
    "LadderRejection",
    "MarketDescriptor",
    "LadderGroup",
    "RejectedLadder",
    "parse_market_descriptor",
    "group_markets_into_ladders",
    "build_settlement_signature",
]


class ThresholdDirection(str, Enum):
    """Semantic direction for nested binary strikes."""

    ABOVE = "above"  # greater / floor strike — easier = lower threshold
    BELOW = "below"  # less / cap strike — easier = higher threshold


class LadderRejection(str, Enum):
    """Reasons a bucket or ladder was not used."""

    mixed_close_time = "mixed_close_time"
    mixed_event = "mixed_event"
    mixed_direction = "mixed_direction"
    mixed_strike_semantics = "mixed_strike_semantics"
    mixed_open_time = "mixed_open_time"
    ambiguous_threshold = "ambiguous_threshold"
    insufficient_strikes = "insufficient_strikes"
    strict_mode_title_mismatch = "strict_mode_title_mismatch"
    not_settled = "not_settled"
    missing_event_key = "missing_event_key"
    non_monotone_thresholds = "non_monotone_thresholds"
    single_market = "single_market"


def _to_unix_ts(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv // 1000 if iv > 1_000_000_000_000 else iv
    try:
        import pandas as pd
        return int(pd.Timestamp(str(v)).timestamp())
    except Exception:
        return None


def build_settlement_signature(m: Dict[str, Any]) -> str:
    """Normalized rule identity for logging (excludes numeric strike so ladders can differ by threshold)."""
    st = str(m.get("strike_type") or "")
    extras = str(m.get("settlement_timer_seconds") or "")
    return f"{st}|{extras}"


def _parse_threshold_from_market(
    market: Dict[str, Any], strict_grouping: bool
) -> Tuple[Optional[float], Optional[ThresholdDirection]]:
    """
    Extract (threshold, direction). Prefer API strike fields; regex only if not strict.
    """
    strike_type = (market.get("strike_type") or "").lower()

    if strike_type == "between":
        return None, None

    if strike_type == "greater":
        v = market.get("floor_strike")
        if v is not None:
            try:
                return float(v), ThresholdDirection.ABOVE
            except (TypeError, ValueError):
                pass
    elif strike_type == "less":
        v = market.get("cap_strike")
        if v is not None:
            try:
                return float(v), ThresholdDirection.BELOW
            except (TypeError, ValueError):
                pass

    if strict_grouping:
        return None, None

    title = (market.get("title") or "") + " " + (market.get("subtitle") or "")
    ticker = market.get("ticker") or ""

    m = re.search(r"(?:above|>\s*=?\s*|>\s*)(\$?\s*[\d,]+(?:\.\d+)?)", title, re.I)
    if m:
        num = re.sub(r"[^\d.]", "", m.group(1))
        try:
            return float(num), ThresholdDirection.ABOVE
        except ValueError:
            pass

    m = re.search(r"(?:below|<\s*=?\s*|<\s*)(\$?\s*[\d,]+(?:\.\d+)?)", title, re.I)
    if m:
        num = re.sub(r"[^\d.]", "", m.group(1))
        try:
            return float(num), ThresholdDirection.BELOW
        except ValueError:
            pass

    m = re.search(r"[-_]T([\d.]+)(?=[^:\d]|$)", ticker)
    if m:
        try:
            return float(m.group(1)), ThresholdDirection.ABOVE
        except ValueError:
            pass

    return None, None


def _underlying_hint(market: Dict[str, Any], series_ticker: str) -> str:
    """Loose identifier for same-underlying (BTC vs index vs other)."""
    t = f"{market.get('title', '')} {market.get('subtitle', '')}".upper()
    st = (market.get("series_ticker") or "").upper()
    if "BTC" in st or "BITCOIN" in t:
        return "BTC"
    if series_ticker and series_ticker.upper() in st:
        return series_ticker.upper()
    return "UNKNOWN"


def _event_group_key(market: Dict[str, Any], strict_grouping: bool) -> Optional[str]:
    et = market.get("event_ticker")
    if et:
        return str(et)
    if strict_grouping:
        return None
    ticker = market.get("ticker") or ""
    parts = ticker.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return None


@dataclass
class MarketDescriptor:
    ticker: str
    title: str
    subtitle: str
    event_ticker: Optional[str]
    open_ts: int
    close_ts: int
    threshold: float
    direction: ThresholdDirection
    strike_type: str
    floor_strike: Optional[float]
    cap_strike: Optional[float]
    settlement_signature: str
    raw_market: Dict[str, Any] = field(repr=False)


@dataclass
class LadderGroup:
    """Ordered easy → hard markets for monotonicity (same event, same expiry)."""

    group_key: str
    ladder_id: str
    markets: List[MarketDescriptor]
    comparable: bool = True
    reject_reason: Optional[LadderRejection] = None
    reject_detail: str = ""

    @property
    def close_ts(self) -> int:
        return self.markets[0].close_ts if self.markets else 0

    @property
    def open_ts(self) -> int:
        return self.markets[0].open_ts if self.markets else 0


@dataclass
class RejectedLadder:
    """A candidate bucket that did not become a valid ladder."""

    group_key: str
    reason: LadderRejection
    detail: str
    tickers: Tuple[str, ...]


def parse_market_descriptor(
    market: Dict[str, Any],
    series_ticker: str,
    strict_grouping: bool,
) -> Optional[MarketDescriptor]:
    """Build descriptor or None if market cannot be classified."""
    ticker = market.get("ticker") or ""
    if not ticker:
        return None

    close_ts = _to_unix_ts(market.get("close_time"))
    open_ts = _to_unix_ts(market.get("open_time"))
    if close_ts is None:
        return None
    if open_ts is None or open_ts >= close_ts:
        open_ts = close_ts - 15 * 60

    thr, direction = _parse_threshold_from_market(market, strict_grouping)
    if thr is None or direction is None:
        return None

    sig = build_settlement_signature(market)
    return MarketDescriptor(
        ticker=ticker,
        title=str(market.get("title") or ""),
        subtitle=str(market.get("subtitle") or ""),
        event_ticker=market.get("event_ticker"),
        open_ts=int(open_ts),
        close_ts=int(close_ts),
        threshold=float(thr),
        direction=direction,
        strike_type=str(market.get("strike_type") or ""),
        floor_strike=_float_or_none(market.get("floor_strike")),
        cap_strike=_float_or_none(market.get("cap_strike")),
        settlement_signature=sig,
        raw_market=dict(market),
    )


def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _strict_title_compatible(descriptors: Sequence[MarketDescriptor], series_ticker: str) -> bool:
    """Require same underlying hint from titles when strict."""
    hints = {_underlying_hint(d.raw_market, series_ticker) for d in descriptors}
    hints.discard("UNKNOWN")
    return len(hints) <= 1


def _order_key(d: MarketDescriptor) -> float:
    """Sort easy → hard: ABOVE: lower strike easier; BELOW: higher strike easier."""
    if d.direction == ThresholdDirection.ABOVE:
        return d.threshold
    return -d.threshold


def group_markets_into_ladders(
    markets: Sequence[Dict[str, Any]],
    series_ticker: str,
    strict_grouping: bool = True,
) -> Tuple[List[LadderGroup], List[RejectedLadder]]:
    """
    Group settled markets into comparable ladders.

    Conservative rules:
    - Same event key (event_ticker, or derived only if not strict_grouping)
    - Same close_ts and open_ts across members
    - Same ThresholdDirection and settlement_signature family
    - At least 2 strictly monotone thresholds after ordering easy→hard
    """
    descriptors: List[MarketDescriptor] = []
    for m in markets:
        result = m.get("result")
        if result not in ("yes", "no"):
            continue
        d = parse_market_descriptor(m, series_ticker, strict_grouping)
        if d is None:
            continue
        descriptors.append(d)

    # Bucket: (event_key, close_ts, open_ts, direction, settlement_signature)
    buckets: Dict[Tuple[Any, ...], List[MarketDescriptor]] = {}
    for d in descriptors:
        ek = _event_group_key(d.raw_market, strict_grouping)
        if ek is None:
            continue
        # Do not include per-strike fields in key — only event window + direction family.
        key = (ek, d.close_ts, d.open_ts, d.direction)
        buckets.setdefault(key, []).append(d)

    ladders: List[LadderGroup] = []
    rejected: List[RejectedLadder] = []

    for key, group in buckets.items():
        event_key, close_ts, open_ts, direction = key
        tickers = tuple(sorted(x.ticker for x in group))

        if len(group) < 2:
            rejected.append(
                RejectedLadder(
                    group_key=str(event_key),
                    reason=LadderRejection.insufficient_strikes,
                    detail="fewer than 2 markets in bucket",
                    tickers=tickers,
                )
            )
            continue

        if strict_grouping and not _strict_title_compatible(group, series_ticker):
            rejected.append(
                RejectedLadder(
                    group_key=str(event_key),
                    reason=LadderRejection.strict_mode_title_mismatch,
                    detail="underlying hint mismatch across titles",
                    tickers=tickers,
                )
            )
            continue

        ordered = sorted(group, key=_order_key)
        thr_list = [x.threshold for x in ordered]

        if direction == ThresholdDirection.ABOVE:
            monotone = all(thr_list[i] < thr_list[i + 1] for i in range(len(thr_list) - 1))
        else:
            monotone = all(thr_list[i] > thr_list[i + 1] for i in range(len(thr_list) - 1))

        if not monotone:
            rejected.append(
                RejectedLadder(
                    group_key=str(event_key),
                    reason=LadderRejection.non_monotone_thresholds,
                    detail=str(thr_list),
                    tickers=tickers,
                )
            )
            continue

        ladder_id = f"{event_key}|{close_ts}|{direction.value}"
        ladders.append(
            LadderGroup(
                group_key=str(event_key),
                ladder_id=ladder_id,
                markets=ordered,
                comparable=True,
            )
        )

    return ladders, rejected


def validate_ladder_homogeneous(ladder: LadderGroup) -> Optional[RejectedLadder]:
    """Extra validation for engine (mixed times inside a pre-built ladder)."""
    if len(ladder.markets) < 2:
        return RejectedLadder(
            group_key=ladder.group_key,
            reason=LadderRejection.insufficient_strikes,
            detail="ladder has < 2 markets",
            tickers=tuple(m.ticker for m in ladder.markets),
        )
    close_ts = {m.close_ts for m in ladder.markets}
    open_ts = {m.open_ts for m in ladder.markets}
    if len(close_ts) > 1:
        return RejectedLadder(
            group_key=ladder.group_key,
            reason=LadderRejection.mixed_close_time,
            detail=str(close_ts),
            tickers=tuple(m.ticker for m in ladder.markets),
        )
    if len(open_ts) > 1:
        return RejectedLadder(
            group_key=ladder.group_key,
            reason=LadderRejection.mixed_open_time,
            detail=str(open_ts),
            tickers=tuple(m.ticker for m in ladder.markets),
        )
    directions = {m.direction for m in ladder.markets}
    if len(directions) > 1:
        return RejectedLadder(
            group_key=ladder.group_key,
            reason=LadderRejection.mixed_direction,
            detail=str(directions),
            tickers=tuple(m.ticker for m in ladder.markets),
        )
    sts = {m.strike_type for m in ladder.markets}
    if len(sts) > 1:
        return RejectedLadder(
            group_key=ladder.group_key,
            reason=LadderRejection.mixed_strike_semantics,
            detail=str(sts),
            tickers=tuple(m.ticker for m in ladder.markets),
        )
    return None
