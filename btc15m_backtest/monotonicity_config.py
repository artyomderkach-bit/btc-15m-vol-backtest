"""
Configuration for monotonicity (ladder) arbitrage backtests.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MonotonicityConfig:
    """Tuning parameters for grouped ladder backtests."""

    min_raw_violation: float = 0.0
    min_executable_violation: float = 0.0
    min_net_violation: float = 0.02
    max_contracts_per_leg: int = 10
    max_notional_per_ladder: float = 100.0
    max_event_family_exposure: float = 500.0
    assume_slippage: float = 0.0
    assume_fee_per_contract: float = 0.0
    entry_cooldown_seconds: int = 120
    use_midpoint: bool = True
    strict_grouping: bool = True
    allow_non_adjacent_pairs: bool = True
    fill_minute_gaps: bool = True
    min_liquidity_price: float = 0.02
    max_liquidity_price: float = 0.98

    def __post_init__(self):
        self.max_contracts_per_leg = max(1, int(self.max_contracts_per_leg))
        self.entry_cooldown_seconds = max(0, int(self.entry_cooldown_seconds))
