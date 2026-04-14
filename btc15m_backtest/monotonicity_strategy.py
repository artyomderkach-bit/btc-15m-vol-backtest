"""
Monotonicity (ladder) arbitrage strategy: synchronized ladder bars, paired YES legs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models import CandleBar, Side
from engine import mid_from_bar
from ladder_grouping import LadderGroup, MarketDescriptor, ThresholdDirection
from monotonicity_config import MonotonicityConfig
from strategy import kalshi_fee


def yes_ask_from_bar(bar: CandleBar) -> Optional[float]:
    return bar.ask_close


def yes_bid_from_bar(bar: CandleBar) -> Optional[float]:
    return bar.bid_close


def price_proxy_for_raw(bar: CandleBar, use_midpoint: bool) -> Optional[float]:
    if use_midpoint:
        m = mid_from_bar(bar)
        if m is not None:
            return m
    return bar.price_close


def _fee_leg(qty: int, price: float, per_contract: float) -> float:
    if per_contract > 0:
        return per_contract * qty
    return kalshi_fee(qty, price)


@dataclass
class _OpenPair:
    easier_ticker: str
    harder_ticker: str
    easier_threshold: float
    harder_threshold: float
    qty: int
    entry_ts: int
    long_avg: float
    short_avg: float
    fee_long: float
    fee_short: float


class MonotonicityStrategy:
    """
    Detects P(easier) < P(harder) violations (ABOVE ladders) and enters
    long YES easier + short YES harder. BELOW ladders: symmetric logic.
    """

    def __init__(self, bankroll: float, config: Optional[MonotonicityConfig] = None):
        self.config = config or MonotonicityConfig()
        self.initial_bankroll = float(bankroll)
        self.bankroll = float(bankroll)
        self.trade_log: List[dict] = []
        self.peak_bankroll = float(bankroll)
        self.max_drawdown = 0.0

        self._last_entry_ts: Dict[str, int] = {}
        self._open_by_ladder: Dict[str, _OpenPair] = {}
        self._event_exposure: Dict[str, float] = {}

        self.opportunities_bars = 0
        self.raw_violation_samples: List[float] = []
        self.executable_violation_samples: List[float] = []
        self.net_violation_samples: List[float] = []

    def _update_drawdown(self):
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        dd = self.peak_bankroll - self.bankroll
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def _log(
        self,
        ladder: LadderGroup,
        market: str,
        threshold: float,
        side: str,
        action: str,
        qty: int,
        entry_price: Optional[float],
        exit_price: Optional[float],
        pnl: float,
        ts: int,
        paired_ticker: str = "",
        signal_strength: float = 0.0,
        raw_violation: float = 0.0,
        executable_violation: float = 0.0,
        net_violation: float = 0.0,
        leg: str = "",
    ):
        row = {
            "ladder_id": ladder.ladder_id,
            "group_key": ladder.group_key,
            "market": market,
            "threshold": threshold,
            "side": side,
            "action": action,
            "qty": abs(qty),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "ts": ts,
            "bankroll": self.bankroll,
            "signal_strength": signal_strength,
            "raw_violation": raw_violation,
            "executable_violation": executable_violation,
            "net_violation": net_violation,
            "paired_ticker": paired_ticker,
        }
        if leg:
            row["leg"] = leg
        self.trade_log.append(row)

    def on_ladder_open(self, ladder: LadderGroup, start_ts: int):
        """Hook for future ladder-level init; validation is done in the engine."""
        return

    def _pair_indices(self, n: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for i in range(n - 1):
            out.append((i, i + 1))
        if self.config.allow_non_adjacent_pairs:
            for i in range(n):
                for j in range(i + 2, n):
                    out.append((i, j))
        return out

    def _violations_for_pair(
        self,
        easier: MarketDescriptor,
        harder: MarketDescriptor,
        bar_e: CandleBar,
        bar_h: CandleBar,
    ) -> Tuple[float, float, float, str]:
        """
        Returns (raw_violation, executable_violation, net_violation, skip_reason).
        For ABOVE: easier has lower strike, should have higher YES price.
        """
        cfg = self.config
        raw_e = price_proxy_for_raw(bar_e, cfg.use_midpoint)
        raw_h = price_proxy_for_raw(bar_h, cfg.use_midpoint)
        ask_e = yes_ask_from_bar(bar_e)
        bid_h = yes_bid_from_bar(bar_h)

        if raw_e is None or raw_h is None:
            return 0.0, 0.0, 0.0, "no_raw"

        # Easier market should have >= YES price than harder (nested events).
        raw_v = max(0.0, raw_h - raw_e)
        if ask_e is not None and bid_h is not None:
            exe_v = max(0.0, bid_h - ask_e)
        else:
            ae = ask_e if ask_e is not None else raw_e + cfg.assume_slippage
            bh = bid_h if bid_h is not None else raw_h - cfg.assume_slippage
            exe_v = max(0.0, bh - ae)

        if ask_e is None:
            ask_e = raw_e + cfg.assume_slippage
        if bid_h is None:
            bid_h = raw_h - cfg.assume_slippage

        qty_probe = min(cfg.max_contracts_per_leg, max(1, int(cfg.max_notional_per_ladder / max(ask_e + bid_h, 1e-6))))
        fee_long = _fee_leg(qty_probe, ask_e, cfg.assume_fee_per_contract)
        fee_short = _fee_leg(qty_probe, bid_h, cfg.assume_fee_per_contract)
        fee_per_unit = (fee_long + fee_short) / float(qty_probe) if qty_probe else 0.0

        net_v = max(0.0, exe_v - fee_per_unit)
        return raw_v, exe_v, net_v, ""

    def _liquidity_ok(self, ask: float, bid: float) -> bool:
        lo, hi = self.config.min_liquidity_price, self.config.max_liquidity_price
        return lo <= ask <= hi and lo <= bid <= hi

    def on_ladder_bar(
        self,
        ladder: LadderGroup,
        bar_ts: int,
        bar_map: Dict[str, CandleBar],
        open_ts: int,
        close_ts: int,
    ):
        if not ladder.markets or not ladder.comparable:
            return

        lid = ladder.ladder_id
        if lid in self._open_by_ladder:
            return

        last_ts = self._last_entry_ts.get(lid, 0)
        if last_ts > 0 and (bar_ts - last_ts) < self.config.entry_cooldown_seconds:
            return

        ms = ladder.markets
        n = len(ms)
        best: Optional[Tuple[float, float, float, int, int, float, float, float]] = None

        any_opp = False
        sample_r = sample_e = sample_n = 0.0
        for i, j in self._pair_indices(n):
            e, h = ms[i], ms[j]
            be, bh = bar_map.get(e.ticker), bar_map.get(h.ticker)
            if be is None or bh is None:
                continue
            raw_v, exe_v, net_v, _ = self._violations_for_pair(e, h, be, bh)
            if raw_v > 1e-12:
                any_opp = True
                if raw_v > sample_r:
                    sample_r, sample_e, sample_n = raw_v, exe_v, net_v

            ask_e = yes_ask_from_bar(be) or (price_proxy_for_raw(be, True) or 0) + self.config.assume_slippage
            bid_h = yes_bid_from_bar(bh) or (price_proxy_for_raw(bh, True) or 0) - self.config.assume_slippage

            if not self._liquidity_ok(ask_e, bid_h):
                continue

            if raw_v < self.config.min_raw_violation:
                continue
            if exe_v < self.config.min_executable_violation:
                continue
            if net_v < self.config.min_net_violation:
                continue

            score = net_v
            if best is None or score > best[0]:
                best = (score, raw_v, exe_v, i, j, ask_e, bid_h, net_v)

        if any_opp:
            self.opportunities_bars += 1
            self.raw_violation_samples.append(sample_r)
            self.executable_violation_samples.append(sample_e)
            self.net_violation_samples.append(sample_n)

        if best is None:
            return

        _, raw_v, exe_v, i, j, ask_e, bid_h, net_v = best
        e, h = ms[i], ms[j]
        be, bh = bar_map[e.ticker], bar_map[h.ticker]

        exp_key = ladder.group_key
        exp = self._event_exposure.get(exp_key, 0.0)
        if exp >= self.config.max_event_family_exposure:
            return

        qty = min(
            self.config.max_contracts_per_leg,
            max(1, int(self.config.max_notional_per_ladder / max(ask_e + bid_h, 1e-6))),
        )
        fee_long = _fee_leg(qty, ask_e, self.config.assume_fee_per_contract)
        fee_short = _fee_leg(qty, bid_h, self.config.assume_fee_per_contract)

        cost_long = qty * ask_e + fee_long
        proceeds_short = qty * bid_h - fee_short
        if self.bankroll < cost_long - proceeds_short + 1e-9:
            return

        self.bankroll -= cost_long
        self.bankroll += proceeds_short
        self._event_exposure[exp_key] = exp + cost_long
        self._update_drawdown()

        op = _OpenPair(
            easier_ticker=e.ticker,
            harder_ticker=h.ticker,
            easier_threshold=e.threshold,
            harder_threshold=h.threshold,
            qty=qty,
            entry_ts=bar_ts,
            long_avg=ask_e,
            short_avg=bid_h,
            fee_long=fee_long,
            fee_short=fee_short,
        )
        self._open_by_ladder[lid] = op
        self._last_entry_ts[lid] = bar_ts

        self._log(
            ladder,
            e.ticker,
            e.threshold,
            Side.YES.value,
            "mono_entry",
            qty,
            ask_e,
            bid_h,
            -(cost_long - proceeds_short),
            bar_ts,
            paired_ticker=h.ticker,
            signal_strength=net_v,
            raw_violation=raw_v,
            executable_violation=exe_v,
            net_violation=net_v,
            leg="long_yes",
        )
        self._log(
            ladder,
            h.ticker,
            h.threshold,
            Side.YES.value,
            "mono_entry",
            qty,
            bid_h,
            ask_e,
            0.0,
            bar_ts,
            paired_ticker=e.ticker,
            signal_strength=net_v,
            raw_violation=raw_v,
            executable_violation=exe_v,
            net_violation=net_v,
            leg="short_yes",
        )

    def on_ladder_settle(self, ladder: LadderGroup, results_map: Dict[str, str]):
        lid = ladder.ladder_id
        op = self._open_by_ladder.pop(lid, None)
        if op is None:
            return

        settle_ts = ladder.close_ts

        res_e = results_map.get(op.easier_ticker, "")
        res_h = results_map.get(op.harder_ticker, "")
        won_e = res_e == "yes"
        won_h = res_h == "yes"

        qty = op.qty
        payout_e = qty * 1.0 if won_e else 0.0
        pnl_e = payout_e - qty * op.long_avg - op.fee_long

        self.bankroll += payout_e

        if won_h:
            self.bankroll -= qty * 1.0
        pnl_h = qty * op.short_avg - op.fee_short - (qty * 1.0 if won_h else 0.0)

        self._update_drawdown()

        exp_key = ladder.group_key
        self._event_exposure[exp_key] = max(0.0, self._event_exposure.get(exp_key, 0.0) - qty * op.long_avg)

        easier = next(m for m in ladder.markets if m.ticker == op.easier_ticker)
        harder = next(m for m in ladder.markets if m.ticker == op.harder_ticker)

        self._log(
            ladder,
            op.easier_ticker,
            easier.threshold,
            Side.YES.value,
            "mono_settlement",
            qty,
            op.long_avg,
            1.0 if won_e else 0.0,
            pnl_e,
            settle_ts,
            paired_ticker=op.harder_ticker,
            leg="long_yes",
        )
        self._log(
            ladder,
            op.harder_ticker,
            harder.threshold,
            Side.YES.value,
            "mono_settlement",
            qty,
            op.short_avg,
            1.0 if won_h else 0.0,
            pnl_h,
            settle_ts,
            paired_ticker=op.easier_ticker,
            leg="short_yes",
        )
