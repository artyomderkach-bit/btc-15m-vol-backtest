"""
Parse Kalshi weather-market candlesticks (dollar-string price keys, volume_fp).

Used by NYC climatology backtest only; kept out of the main BTC engine.
"""
from typing import Optional

from models import CandleBar


def _dollars_to_frac(v) -> Optional[float]:
    """Convert dollar string like '0.0700' to 0-1 fraction."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _extract_field(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def parse_candle_weather(raw) -> Optional[CandleBar]:
    """
    Parse a Kalshi weather-market candlestick.

    Weather markets use *_dollars suffix keys (e.g., open_dollars='0.0700')
    and volume_fp rather than the integer-cent keys used in BTC markets.
    For bars where no trades occurred (empty price dict) we fall back to the
    bid/ask midpoint so the strategy can still observe the current market price.
    """
    ts = _extract_field(raw, 'end_period_ts')
    if ts is None:
        return None
    ts = int(ts)

    price = _extract_field(raw, 'price') or {}
    ask = _extract_field(raw, 'yes_ask') or {}
    bid = _extract_field(raw, 'yes_bid') or {}

    vol_raw = _extract_field(raw, 'volume_fp')
    try:
        vol = int(float(vol_raw)) if vol_raw is not None else 0
    except (TypeError, ValueError):
        vol = 0

    def _dp(key):
        return _dollars_to_frac(price.get(key + '_dollars'))

    price_open = _dp('open')
    price_high = _dp('high')
    price_low = _dp('low')
    price_close = _dp('close')

    if price_close is None:
        bid_c = _dollars_to_frac(bid.get('close_dollars'))
        ask_c = _dollars_to_frac(ask.get('close_dollars'))
        if bid_c is not None and ask_c is not None:
            mid = (bid_c + ask_c) / 2.0
        elif bid_c is not None:
            mid = bid_c
        elif ask_c is not None:
            mid = ask_c
        else:
            mid = None
        if mid is not None:
            price_open = price_open if price_open is not None else mid
            price_close = mid
            price_high = price_high if price_high is not None else mid
            price_low = price_low if price_low is not None else mid

    return CandleBar(
        ts=ts,
        price_open=price_open,
        price_high=price_high,
        price_low=price_low,
        price_close=price_close,
        ask_low=_dollars_to_frac(ask.get('low_dollars')),
        ask_high=_dollars_to_frac(ask.get('high_dollars')),
        bid_low=_dollars_to_frac(bid.get('low_dollars')),
        bid_high=_dollars_to_frac(bid.get('high_dollars')),
        volume=vol,
    )
