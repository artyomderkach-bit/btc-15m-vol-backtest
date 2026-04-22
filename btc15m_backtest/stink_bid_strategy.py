"""
Stink-bid strategy logic for BTC 15m markets.

Limitations:
- No queue position model; fills are trade-through based.
- Partial-fill approximation uses trade size and config.volume_fill_pct.
- Millisecond timing is only as accurate as trade timestamps in the source data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from strategy import kalshi_fee
from stink_bid_config import StinkBidConfig


@dataclass
class SimOrder:
    order_id: str
    market: str
    side: str  # YES / NO
    level_idx: int
    price: float
    qty_total: int
    qty_filled: int
    ts_submit: float
    ts_expire: float
    status: str = "open"
    ts_first_fill: Optional[float] = None
    ts_last_fill: Optional[float] = None

    @property
    def remaining(self) -> int:
        return max(0, self.qty_total - self.qty_filled)


@dataclass
class FillLot:
    market: str
    side: str  # YES / NO
    qty: int
    entry_price: float
    ts_fill: float
    level_idx: int
    fees_entry: float
    ts_submit: float
    open_ts: int
    close_ts: int
    used_trade_data: bool


class StinkBidStrategy:
    def __init__(self, bankroll: float, config: Optional[StinkBidConfig] = None):
        self.config = config or StinkBidConfig()
        self.initial_bankroll = float(bankroll)
        self.bankroll = float(bankroll)
        self.trade_log: List[dict] = []
        self.peak_bankroll = float(bankroll)
        self.max_drawdown = 0.0
        self._order_seq = 0
        self._open_lots: Dict[str, List[FillLot]] = {}
        self._open_orders: Dict[str, List[SimOrder]] = {}
        self._exposure_by_market: Dict[str, float] = {}
        self._inventory_by_market: Dict[str, int] = {}
        self._max_inventory = 0
        self._max_exposure = 0.0
        self._submit_latency_ms: List[float] = []
        self._fill_latency_ms: List[float] = []
        self._rested_too_long = 0

    @staticmethod
    def snap_to_tick(price: float, tick: float, lo: float, hi: float) -> Optional[float]:
        if tick <= 0:
            return None
        if not math.isfinite(price):
            return None
        p = round(round(price / tick) * tick, 10)
        if p < lo or p > hi:
            return None
        return p

    def _update_drawdown(self) -> None:
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        dd = self.peak_bankroll - self.bankroll
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def _fee(self, qty: int, price: float) -> float:
        if qty <= 0:
            return 0.0
        if self.config.assume_fee_per_contract > 0:
            return float(qty) * float(self.config.assume_fee_per_contract)
        return kalshi_fee(qty, price)

    @staticmethod
    def _trade_ts(tr: dict) -> Optional[float]:
        for k in ("created_time", "created_ts", "ts"):
            v = tr.get(k)
            if v is None:
                continue
            if isinstance(v, (int, float)):
                iv = float(v)
                return iv / 1000.0 if iv > 1_000_000_000_000 else iv
        return None

    @staticmethod
    def _trade_yes_price(tr: dict) -> Optional[float]:
        yd = tr.get("yes_price_dollars")
        if yd is not None and yd != "":
            try:
                return float(yd)
            except Exception:
                pass
        yc = tr.get("yes_price")
        if yc is not None:
            try:
                val = float(yc)
                return val / 100.0 if val > 1.0 else val
            except Exception:
                pass
        p = tr.get("price")
        if p is not None:
            try:
                val = float(p)
                return val / 100.0 if val > 1.0 else val
            except Exception:
                pass
        return None

    @staticmethod
    def _trade_qty(tr: dict) -> int:
        for k in ("count_fp", "count", "quantity", "size"):
            v = tr.get(k)
            if v is None:
                continue
            try:
                return max(0, int(float(v)))
            except Exception:
                continue
        return 1

    def _make_order(self, market: str, side: str, level_idx: int, price: float, qty: int, ts_submit: float, ts_expire: float) -> SimOrder:
        self._order_seq += 1
        return SimOrder(
            order_id=f"sb-{self._order_seq}",
            market=market,
            side=side,
            level_idx=level_idx,
            price=price,
            qty_total=max(0, int(qty)),
            qty_filled=0,
            ts_submit=ts_submit,
            ts_expire=ts_expire,
        )

    def build_price_levels(
        self,
        side: str,
        best_bid_yes: Optional[float],
        absolute_prices: Sequence[float],
    ) -> List[float]:
        out: List[float] = []
        if self.config.bad_price_mode == "percentile_from_inside" and best_bid_yes is not None:
            for off in self.config.pct_offsets_below_best_bid[: self.config.num_levels]:
                if side == "YES":
                    candidate = best_bid_yes * (1.0 - float(off))
                else:
                    no_best_bid = max(0.0, 1.0 - best_bid_yes)
                    candidate = no_best_bid * (1.0 - float(off))
                snapped = self.snap_to_tick(candidate, self.config.tick, self.config.min_legal_price, self.config.max_legal_price)
                if snapped is not None:
                    out.append(snapped)
        else:
            for candidate in list(absolute_prices)[: self.config.num_levels]:
                snapped = self.snap_to_tick(float(candidate), self.config.tick, self.config.min_legal_price, self.config.max_legal_price)
                if snapped is not None:
                    out.append(snapped)
        dedup = sorted(set(out))
        return dedup[: self.config.num_levels]

    def submit_open_ladder(
        self,
        market: str,
        open_ts: int,
        close_ts: int,
        best_bid_yes: Optional[float] = None,
    ) -> List[SimOrder]:
        submit_ts = float(open_ts) + self.config.opening_delay_ms / 1000.0
        expire_ts = submit_ts + self.config.cancel_timeout_ms / 1000.0
        allowed_sides = [s for s in self.config.sides if s in ("YES", "NO")]
        orders: List[SimOrder] = []
        slots_left = int(self.config.max_open_orders)
        market_exposure = self._exposure_by_market.get(market, 0.0)
        for side in allowed_sides:
            abs_prices = self.config.bad_prices_yes if side == "YES" else self.config.bad_prices_no
            for i, px in enumerate(self.build_price_levels(side, best_bid_yes, abs_prices)):
                if slots_left <= 0:
                    break
                if market_exposure >= self.config.max_notional_per_market:
                    break
                max_qty_exposure = int((self.config.max_notional_per_market - market_exposure) / max(px, 1e-9))
                qty = max(0, min(self.config.max_contracts_per_level, max_qty_exposure))
                if qty <= 0:
                    continue
                o = self._make_order(market, side, i, px, qty, submit_ts, expire_ts)
                orders.append(o)
                market_exposure += qty * px
                slots_left -= 1
                self._submit_latency_ms.append((submit_ts - float(open_ts)) * 1000.0)
                self.trade_log.append(
                    {
                        "market": market,
                        "side": side,
                        "action": "submit",
                        "order_id": o.order_id,
                        "level_idx": i,
                        "price": px,
                        "qty": qty,
                        "ts_submit": submit_ts,
                        "ts_fill": None,
                        "ts_cancel": None,
                        "ts_exit": None,
                        "entry_price": None,
                        "exit_price": None,
                        "pnl_gross": 0.0,
                        "fees": 0.0,
                        "pnl_net": 0.0,
                        "bankroll": self.bankroll,
                        "open_ts": open_ts,
                        "close_ts": close_ts,
                        "latency_ms_submit_from_open": (submit_ts - float(open_ts)) * 1000.0,
                        "fill_latency_ms": None,
                        "used_trade_data": None,
                        "note": "",
                    }
                )
        self._open_orders[market] = orders
        self._exposure_by_market[market] = market_exposure
        self._max_exposure = max(self._max_exposure, market_exposure)
        return orders

    def _order_fillable(self, order: SimOrder, yes_px: float) -> bool:
        if order.side == "YES":
            return yes_px <= order.price + 1e-12
        return yes_px >= (1.0 - order.price) - 1e-12

    def process_trade_window(self, market: str, open_ts: int, close_ts: int, trades: Sequence[dict], used_trade_data: bool) -> None:
        orders = self._open_orders.get(market, [])
        if not orders:
            return
        for tr in trades:
            ts = self._trade_ts(tr)
            yes_px = self._trade_yes_price(tr)
            if ts is None or yes_px is None:
                continue
            for order in orders:
                if order.status not in ("open", "partial"):
                    continue
                if ts < order.ts_submit or ts > order.ts_expire:
                    continue
                if not self._order_fillable(order, yes_px):
                    continue
                trade_qty = self._trade_qty(tr)
                per_trade_cap = max(1, int(trade_qty * self.config.volume_fill_pct))
                fill_qty = min(order.remaining, per_trade_cap)
                if fill_qty <= 0:
                    continue
                order.qty_filled += fill_qty
                if order.ts_first_fill is None:
                    order.ts_first_fill = ts
                order.ts_last_fill = ts
                order.status = "filled" if order.remaining == 0 else "partial"
                entry_fee = self._fee(fill_qty, order.price)
                self.bankroll -= (fill_qty * order.price + entry_fee)
                self._update_drawdown()
                lot = FillLot(
                    market=market,
                    side=order.side,
                    qty=fill_qty,
                    entry_price=order.price,
                    ts_fill=ts,
                    level_idx=order.level_idx,
                    fees_entry=entry_fee,
                    ts_submit=order.ts_submit,
                    open_ts=open_ts,
                    close_ts=close_ts,
                    used_trade_data=used_trade_data,
                )
                self._open_lots.setdefault(market, []).append(lot)
                inv = self._inventory_by_market.get(market, 0) + fill_qty
                self._inventory_by_market[market] = inv
                self._max_inventory = max(self._max_inventory, inv)
                latency_ms = (ts - order.ts_submit) * 1000.0
                self._fill_latency_ms.append(latency_ms)
                self.trade_log.append(
                    {
                        "market": market,
                        "side": order.side,
                        "action": "fill",
                        "order_id": order.order_id,
                        "level_idx": order.level_idx,
                        "price": order.price,
                        "qty": fill_qty,
                        "ts_submit": order.ts_submit,
                        "ts_fill": ts,
                        "ts_cancel": None,
                        "ts_exit": None,
                        "entry_price": order.price,
                        "exit_price": None,
                        "pnl_gross": 0.0,
                        "fees": entry_fee,
                        "pnl_net": -entry_fee,
                        "bankroll": self.bankroll,
                        "open_ts": open_ts,
                        "close_ts": close_ts,
                        "latency_ms_submit_from_open": (order.ts_submit - float(open_ts)) * 1000.0,
                        "fill_latency_ms": latency_ms,
                        "used_trade_data": used_trade_data,
                        "note": "",
                    }
                )
        for order in orders:
            if order.status in ("filled",):
                continue
            order.status = "cancelled"
            self._rested_too_long += 1
            self.trade_log.append(
                {
                    "market": market,
                    "side": order.side,
                    "action": "cancel",
                    "order_id": order.order_id,
                    "level_idx": order.level_idx,
                    "price": order.price,
                    "qty": order.remaining,
                    "ts_submit": order.ts_submit,
                    "ts_fill": order.ts_last_fill,
                    "ts_cancel": order.ts_expire,
                    "ts_exit": None,
                    "entry_price": None,
                    "exit_price": None,
                    "pnl_gross": 0.0,
                    "fees": 0.0,
                    "pnl_net": 0.0,
                    "bankroll": self.bankroll,
                    "open_ts": open_ts,
                    "close_ts": close_ts,
                    "latency_ms_submit_from_open": (order.ts_submit - float(open_ts)) * 1000.0,
                    "fill_latency_ms": None,
                    "used_trade_data": used_trade_data,
                    "note": "timeout_cancel",
                }
            )

    def _quote_for_exit(self, side: str, bar_yes_bid: Optional[float], bar_yes_ask: Optional[float], bar_mid: Optional[float]) -> Optional[float]:
        if side == "YES":
            return bar_yes_bid if bar_yes_bid is not None else bar_mid
        no_bid = (1.0 - bar_yes_ask) if bar_yes_ask is not None else None
        return no_bid if no_bid is not None else (1.0 - bar_mid if bar_mid is not None else None)

    def exit_lots_time(self, market: str, ts_exit: float, yes_bid: Optional[float], yes_ask: Optional[float], yes_mid: Optional[float]) -> None:
        lots = self._open_lots.get(market, [])
        if not lots:
            return
        keep: List[FillLot] = []
        for lot in lots:
            if self.config.exit_mode != "time_exit":
                keep.append(lot)
                continue
            px = self._quote_for_exit(lot.side, yes_bid, yes_ask, yes_mid)
            if px is None:
                keep.append(lot)
                continue
            fee = self._fee(lot.qty, px)
            gross = lot.qty * (px - lot.entry_price)
            if lot.side == "NO":
                gross = lot.qty * ((1.0 - px) - lot.entry_price)
            pnl_net = gross - lot.fees_entry - fee
            self.bankroll += lot.qty * px - fee if lot.side == "YES" else lot.qty * (1.0 - px) - fee
            self._update_drawdown()
            self.trade_log.append(
                {
                    "market": lot.market,
                    "side": lot.side,
                    "action": "time_exit",
                    "order_id": "",
                    "level_idx": lot.level_idx,
                    "price": lot.entry_price,
                    "qty": lot.qty,
                    "ts_submit": lot.ts_submit,
                    "ts_fill": lot.ts_fill,
                    "ts_cancel": None,
                    "ts_exit": ts_exit,
                    "entry_price": lot.entry_price,
                    "exit_price": px,
                    "pnl_gross": gross,
                    "fees": lot.fees_entry + fee,
                    "pnl_net": pnl_net,
                    "bankroll": self.bankroll,
                    "open_ts": lot.open_ts,
                    "close_ts": lot.close_ts,
                    "latency_ms_submit_from_open": (lot.ts_submit - float(lot.open_ts)) * 1000.0,
                    "fill_latency_ms": (lot.ts_fill - lot.ts_submit) * 1000.0,
                    "used_trade_data": lot.used_trade_data,
                    "note": "",
                }
            )
            self._inventory_by_market[market] = max(0, self._inventory_by_market.get(market, 0) - lot.qty)
        self._open_lots[market] = keep

    def settle_market(self, market: str, result: str, ts_settle: int) -> None:
        lots = self._open_lots.get(market, [])
        if not lots:
            return
        keep: List[FillLot] = []
        result_yes = (result or "").strip().lower() == "yes"
        for lot in lots:
            if self.config.exit_mode == "time_exit":
                keep.append(lot)
                continue
            won = result_yes if lot.side == "YES" else (not result_yes)
            payout = float(lot.qty) if won else 0.0
            fee = self._fee(lot.qty, 1.0 if won else 0.0)
            gross = lot.qty * ((1.0 - lot.entry_price) if won else -lot.entry_price)
            pnl_net = gross - lot.fees_entry - fee
            self.bankroll += payout - fee
            self._update_drawdown()
            self.trade_log.append(
                {
                    "market": lot.market,
                    "side": lot.side,
                    "action": "settlement",
                    "order_id": "",
                    "level_idx": lot.level_idx,
                    "price": lot.entry_price,
                    "qty": lot.qty,
                    "ts_submit": lot.ts_submit,
                    "ts_fill": lot.ts_fill,
                    "ts_cancel": None,
                    "ts_exit": ts_settle,
                    "entry_price": lot.entry_price,
                    "exit_price": 1.0 if won else 0.0,
                    "pnl_gross": gross,
                    "fees": lot.fees_entry + fee,
                    "pnl_net": pnl_net,
                    "bankroll": self.bankroll,
                    "open_ts": lot.open_ts,
                    "close_ts": lot.close_ts,
                    "latency_ms_submit_from_open": (lot.ts_submit - float(lot.open_ts)) * 1000.0,
                    "fill_latency_ms": (lot.ts_fill - lot.ts_submit) * 1000.0,
                    "used_trade_data": lot.used_trade_data,
                    "note": "",
                }
            )
            self._inventory_by_market[market] = max(0, self._inventory_by_market.get(market, 0) - lot.qty)
        self._open_lots[market] = keep

    def summarize(self) -> dict:
        submitted = sum(1 for r in self.trade_log if r.get("action") == "submit")
        fills = [r for r in self.trade_log if r.get("action") == "fill"]
        exits = [r for r in self.trade_log if r.get("action") in ("time_exit", "settlement")]
        cancelled = sum(1 for r in self.trade_log if r.get("action") == "cancel")
        by_level: Dict[int, int] = {}
        for r in fills:
            idx = int(r.get("level_idx", -1))
            by_level[idx] = by_level.get(idx, 0) + int(r.get("qty", 0))
        edge_vals = [float(r.get("entry_price", 0.0)) for r in fills]
        adverse_1m = [float(r.get("entry_price", 0.0)) - float(r.get("exit_price", 0.0)) for r in exits if r.get("action") == "time_exit"]
        pnl_gross_sum = float(sum(float(r.get("pnl_gross", 0.0)) for r in exits))
        entry_fees = float(sum(float(r.get("fees", 0.0)) for r in fills))
        exit_fees = float(sum(float(r.get("fees", 0.0)) for r in exits))
        return {
            "total_orders_submitted": submitted,
            "total_fills": len(fills),
            "overall_fill_rate": (len(fills) / submitted) if submitted else 0.0,
            "fills_by_level": by_level,
            "avg_fill_latency_ms": (sum(self._fill_latency_ms) / len(self._fill_latency_ms)) if self._fill_latency_ms else 0.0,
            "median_fill_latency_ms": median(self._fill_latency_ms) if self._fill_latency_ms else 0.0,
            "avg_submit_latency_ms": (sum(self._submit_latency_ms) / len(self._submit_latency_ms)) if self._submit_latency_ms else 0.0,
            "resting_too_long_count": self._rested_too_long,
            "canceled_count": cancelled,
            "pnl_gross_sum": pnl_gross_sum,
            "fees_sum": entry_fees + exit_fees,
            "pnl_net_sum": pnl_gross_sum - entry_fees - exit_fees,
            "avg_edge_captured_at_fill": (sum(edge_vals) / len(edge_vals)) if edge_vals else 0.0,
            "adverse_selection_1min": (sum(adverse_1m) / len(adverse_1m)) if adverse_1m else 0.0,
            "adverse_selection_5min": (sum(adverse_1m) / len(adverse_1m)) if adverse_1m else 0.0,
            "max_inventory_contracts": self._max_inventory,
            "max_exposure_dollars": self._max_exposure,
            "final_bankroll": self.bankroll,
            "max_drawdown_pct": (self.max_drawdown / self.peak_bankroll * 100.0) if self.peak_bankroll else 0.0,
        }
