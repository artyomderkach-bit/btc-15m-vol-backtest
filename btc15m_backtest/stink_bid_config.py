"""
Config for BTC 15m stink-bid-at-open backtests.

This module is intentionally standalone so existing strategies are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class StinkBidConfig:
    series_ticker: str = "KXBTC15M"
    opening_delay_ms: int = 0
    cancel_timeout_ms: int = 500
    sides: Tuple[str, ...] = ("YES",)
    num_levels: int = 5

    # Price-level generation
    bad_price_mode: str = "absolute"  # absolute | percentile_from_inside
    bad_prices_yes: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.10)
    bad_prices_no: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.10)
    pct_offsets_below_best_bid: Tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.70)

    # Risk / placement caps
    max_contracts_per_level: int = 1
    max_notional_per_market: float = 50.0
    max_open_orders: int = 10
    volume_fill_pct: float = 0.10
    replace_canceled: bool = False

    # Exit
    exit_mode: str = "time_exit"  # time_exit | hold_to_expiry
    time_exit_minutes: int = 5

    # Fees: 0.0 means use kalshi_fee model from strategy.py
    assume_fee_per_contract: float = 0.0

    # Price legality
    min_legal_price: float = 0.01
    max_legal_price: float = 0.99
    tick: float = 0.01

    def __post_init__(self):
        self.opening_delay_ms = max(0, int(self.opening_delay_ms))
        self.cancel_timeout_ms = max(1, int(self.cancel_timeout_ms))
        self.num_levels = max(1, int(self.num_levels))
        self.max_contracts_per_level = max(1, int(self.max_contracts_per_level))
        self.max_open_orders = max(1, int(self.max_open_orders))
        self.volume_fill_pct = max(0.0, min(1.0, float(self.volume_fill_pct)))
        self.time_exit_minutes = max(1, int(self.time_exit_minutes))
        self.bad_price_mode = (self.bad_price_mode or "absolute").strip().lower()
        self.exit_mode = (self.exit_mode or "time_exit").strip().lower()
        self.sides = tuple(s.strip().upper() for s in self.sides if s and s.strip())
        if not self.sides:
            self.sides = ("YES",)
