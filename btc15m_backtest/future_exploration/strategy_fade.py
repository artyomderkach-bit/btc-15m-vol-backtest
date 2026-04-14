"""
AdaptiveLimitFadeStrategy: Buy panic at 10¢ during high-volatility periods
in KXBTC15M markets and sell the bounce at two take-profit levels.

Only active when early volatility suggests an overreaction:
  1. Watch first 3 minutes for price_range / open >= VOL_THRESHOLD
  2. If triggered, place YES+NO limit buys at ENTRY_PRICE
  3. On fill: cancel opposite, place TP1 (50% at 33¢) + TP2 (50% at 45¢)
  4. Cancel unfilled entries after ENTRY_EXPIRY_MINUTES from market open
  5. Time-stop: force sell at market price if no TP by STOP_MINUTES
  6. Filters: skip weekends and off-hours (only trade US session)
"""
import math
import random
import uuid
import datetime as _dt
from typing import Dict, List, Tuple

from models import Order, Position, Side, OrderType, OrderStatus, CandleBar

MAKER_RATE = 0.0175


def kalshi_fee(contracts: int, price: float) -> float:
    if contracts <= 0:
        return 0.0
    return math.ceil(MAKER_RATE * contracts * price * (1 - price) * 100) / 100


class AdaptiveLimitFadeStrategy:
    def __init__(self, bankroll: float, risk_pct: float = 0.015,
                 vol_threshold: float = 0.80,
                 entry_price: float = 0.10,
                 tp1_price: float = 0.33,
                 tp2_price: float = 0.50,
                 entry_expiry_minutes: int = 6,
                 vol_window_minutes: int = 3,
                 stop_minutes: int = 7,
                 active_hours: Tuple[int, int] = (13, 20),
                 skip_weekends: bool = True,
                 volume_fill_pct: float = 0.10,
                 tp_fill_rate: float = 1.0):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.risk_pct = risk_pct
        self.vol_threshold = vol_threshold
        self.entry_price = entry_price
        self.buy_price = entry_price
        self.tp1_price = tp1_price
        self.tp2_price = tp2_price
        self.sell_price = tp1_price
        self.entry_expiry_seconds = entry_expiry_minutes * 60
        self.vol_window_seconds = vol_window_minutes * 60
        self.stop_seconds = stop_minutes * 60
        self.active_hours = active_hours
        self.skip_weekends = skip_weekends
        self.volume_fill_pct = volume_fill_pct
        self.tp_fill_rate = max(0.0, min(1.0, tp_fill_rate))
        self.halted = False

        self.open_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[dict] = []
        self._last_bar_ts: int = 0
        self._last_buy_fill_ts: Dict[str, int] = {}

        self.peak_bankroll = bankroll
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        self._market_state: Dict[str, dict] = {}

        # Fade-specific counters
        self.vol_gate_triggered = 0
        self.vol_gate_skipped = 0
        self.tp1_fills = 0
        self.tp2_fills = 0
        self.time_stops = 0
        self.weekend_skips = 0
        self.hour_skips = 0

    # ── Engine interface ──

    def _should_skip_market(self, open_ts: int) -> str:
        """Return skip reason or empty string if market is tradeable."""
        dt = _dt.datetime.utcfromtimestamp(open_ts)
        if self.skip_weekends and dt.weekday() >= 5:
            return 'weekend'
        h_start, h_end = self.active_hours
        if not (h_start <= dt.hour < h_end):
            return 'hour'
        return ''

    def on_market_open(self, market_ticker: str, open_ts: int):
        """Initialize per-market volatility tracking. No orders placed yet."""
        if self.halted:
            return

        skip = self._should_skip_market(open_ts)
        if skip == 'weekend':
            self.weekend_skips += 1
            return
        if skip == 'hour':
            self.hour_skips += 1
            return

        self._market_state[market_ticker] = {
            'open_price': None,
            'max_price': None,
            'min_price': None,
            'activated': False,
            'gate_checked': False,
            'entry_placed': False,
        }

    def on_candle(self, market_ticker: str, bar: CandleBar, open_ts: int, close_ts: int):
        if self.halted:
            return

        self._last_bar_ts = bar.ts
        elapsed = bar.ts - open_ts
        ms = self._market_state.get(market_ticker)
        if ms is None:
            return

        # 1. Track prices during volatility window
        if elapsed <= self.vol_window_seconds:
            if bar.price_open is not None and ms['open_price'] is None:
                ms['open_price'] = bar.price_open
            if bar.price_high is not None:
                if ms['max_price'] is None or bar.price_high > ms['max_price']:
                    ms['max_price'] = bar.price_high
            if bar.price_low is not None:
                if ms['min_price'] is None or bar.price_low < ms['min_price']:
                    ms['min_price'] = bar.price_low

        # 2. Evaluate volatility gate once at end of window
        if elapsed >= self.vol_window_seconds and not ms['gate_checked']:
            ms['gate_checked'] = True
            op, hi, lo = ms['open_price'], ms['max_price'], ms['min_price']
            if op and hi and lo and op > 0:
                price_range_pct = (hi - lo) / op
                if price_range_pct >= self.vol_threshold:
                    ms['activated'] = True
                    self.vol_gate_triggered += 1
                else:
                    self.vol_gate_skipped += 1
            else:
                self.vol_gate_skipped += 1

        # 3. Place entry orders once gate triggers
        if ms['activated'] and not ms['entry_placed']:
            ms['entry_placed'] = True
            risk_dollars = self._risk_dollars()
            qty = max(1, int(risk_dollars / self.entry_price))
            for side in [Side.YES, Side.NO]:
                oid = self._new_order_id()
                order = Order(
                    order_id=oid,
                    market_ticker=market_ticker,
                    side=side,
                    order_type=OrderType.BUY,
                    price=self.entry_price,
                    quantity=qty,
                    placed_at=bar.ts,
                )
                self.open_orders[oid] = order

        # 4. Cancel unfilled buy orders after expiry
        if elapsed > self.entry_expiry_seconds:
            self._cancel_buy_orders(market_ticker)

        # 5. Check buy fills (only when active and before expiry)
        if ms['activated'] and ms['entry_placed'] and elapsed <= self.entry_expiry_seconds:
            self._check_buy_fills(market_ticker, bar)

        # 6. Check sell (TP) fills throughout the market
        self._check_sell_fills(market_ticker, bar)

        # 7. Time stop: force-sell at market price if past stop_minutes
        if elapsed >= self.stop_seconds:
            self._force_close_positions(market_ticker, bar)

    def on_market_settle(self, market_ticker: str, result: str):
        self._cancel_all_orders(market_ticker)
        for side in [Side.YES, Side.NO]:
            pk = self._pos_key(market_ticker, side)
            pos = self.positions.get(pk)
            if pos and pos.quantity > 0:
                won = (side.value == result)
                qty = pos.quantity
                entry = pos.avg_entry
                payout = qty * 1.0 if won else 0.0
                self.bankroll += payout
                pnl = payout - qty * entry
                pos.realized_pnl += pnl
                pos.quantity = 0
                self._update_drawdown()
                if pnl < 0:
                    self.consecutive_losses += 1
                    self.max_consecutive_losses = max(
                        self.max_consecutive_losses, self.consecutive_losses)
                else:
                    self.consecutive_losses = 0
                self.trade_log.append({
                    'market': market_ticker,
                    'side': side.value,
                    'action': 'settlement',
                    'won': won,
                    'qty': qty,
                    'entry_price': entry,
                    'exit_price': 1.0 if won else 0.0,
                    'pnl': pnl,
                    'bankroll': self.bankroll,
                })
        self._market_state.pop(market_ticker, None)

    # ── Fill checks (same logic as existing strategy) ──

    def _check_buy_fills(self, market_ticker: str, bar: CandleBar):
        buy_orders = [o for o in self.open_orders.values()
                      if o.market_ticker == market_ticker
                      and o.order_type == OrderType.BUY
                      and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)]
        for order in buy_orders:
            fill_qty = self._compute_buy_fill(order, bar)
            if fill_qty <= 0:
                continue
            fee = kalshi_fee(fill_qty, order.price)
            order.fill(fill_qty, bar.ts)
            self.bankroll -= fill_qty * order.price + fee
            self._last_buy_fill_ts[market_ticker] = bar.ts

            pk = self._pos_key(market_ticker, order.side)
            if pk not in self.positions:
                self.positions[pk] = Position(market_ticker, order.side)
            self.positions[pk].add(fill_qty, order.price, fee)
            self._update_drawdown()

            self._cancel_opposite_buys(market_ticker, order.side)
            self._place_tp_orders(market_ticker, order.side, fill_qty)

            self.trade_log.append({
                'market': market_ticker,
                'side': order.side.value,
                'action': 'buy_fill',
                'qty': fill_qty,
                'entry_price': order.price,
                'exit_price': None,
                'pnl': -(fill_qty * order.price + fee),
                'bankroll': self.bankroll,
                'ts': bar.ts,
            })

    def _check_sell_fills(self, market_ticker: str, bar: CandleBar):
        sell_orders = [o for o in self.open_orders.values()
                       if o.market_ticker == market_ticker
                       and o.order_type == OrderType.SELL
                       and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)]
        for order in sell_orders:
            fill_qty = self._compute_sell_fill(order, bar)
            if fill_qty <= 0:
                continue
            fee = kalshi_fee(fill_qty, order.price)
            order.fill(fill_qty, bar.ts)
            pk = self._pos_key(market_ticker, order.side)
            pos = self.positions.get(pk)
            if pos:
                pnl = pos.sell(fill_qty, order.price, fee)
                self.bankroll += fill_qty * order.price - fee
                if pnl > 0:
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.max_consecutive_losses = max(
                        self.max_consecutive_losses, self.consecutive_losses)
                self._update_drawdown()

                is_tp1 = abs(order.price - self.tp1_price) < 0.01
                if is_tp1:
                    self.tp1_fills += 1
                else:
                    self.tp2_fills += 1

                self.trade_log.append({
                    'market': market_ticker,
                    'side': order.side.value,
                    'action': 'sell_fill',
                    'qty': fill_qty,
                    'entry_price': pos.avg_entry,
                    'exit_price': order.price,
                    'pnl': pnl,
                    'bankroll': self.bankroll,
                    'ts': bar.ts,
                    'tp_level': 1 if is_tp1 else 2,
                })

    def _compute_buy_fill(self, order: Order, bar: CandleBar) -> int:
        if order.remaining <= 0 or order.status == OrderStatus.CANCELLED:
            return 0
        matched = False
        if order.side == Side.YES:
            if bar.price_low is not None and bar.price_low <= order.price:
                matched = True
        else:
            yes_level = 1.0 - order.price
            if bar.price_high is not None and bar.price_high >= yes_level:
                matched = True
        if not matched:
            return 0
        return min(order.remaining, self._volume_cap(bar))

    def _compute_sell_fill(self, order: Order, bar: CandleBar) -> int:
        if order.remaining <= 0 or order.status == OrderStatus.CANCELLED:
            return 0
        if order.placed_at >= bar.ts:
            return 0
        last_buy_ts = self._last_buy_fill_ts.get(order.market_ticker, 0)
        if bar.ts <= last_buy_ts:
            return 0
        matched = False
        if order.side == Side.YES:
            if bar.price_high is not None and bar.price_high >= order.price:
                matched = True
        else:
            yes_level = 1.0 - order.price
            if bar.price_low is not None and bar.price_low <= yes_level:
                matched = True
        if not matched:
            return 0
        if self.tp_fill_rate < 1.0 and random.random() >= self.tp_fill_rate:
            return 0
        return min(order.remaining, self._volume_cap(bar))

    # ── Time stop ──

    def _force_close_positions(self, market_ticker: str, bar: CandleBar):
        """Force-sell any remaining position at current market price."""
        for side in [Side.YES, Side.NO]:
            pk = self._pos_key(market_ticker, side)
            pos = self.positions.get(pk)
            if not pos or pos.quantity <= 0:
                continue

            # Determine exit price from current bar
            if bar.price_close is None:
                continue
            if side == Side.YES:
                exit_price = bar.price_close
            else:
                exit_price = 1.0 - bar.price_close
            exit_price = max(exit_price, 0.01)

            qty = pos.quantity
            fee = kalshi_fee(qty, exit_price)
            pnl = qty * (exit_price - pos.avg_entry) - fee
            self.bankroll += qty * exit_price - fee
            pos.realized_pnl += pnl
            pos.quantity = 0

            self._cancel_all_orders(market_ticker)
            self._update_drawdown()
            self.time_stops += 1

            if pnl < 0:
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(
                    self.max_consecutive_losses, self.consecutive_losses)
            else:
                self.consecutive_losses = 0

            self.trade_log.append({
                'market': market_ticker,
                'side': side.value,
                'action': 'time_stop',
                'qty': qty,
                'entry_price': pos.avg_entry,
                'exit_price': exit_price,
                'pnl': pnl,
                'bankroll': self.bankroll,
                'ts': bar.ts,
            })

    # ── Order management ──

    def _place_tp_orders(self, market_ticker: str, side: Side, total_qty: int):
        """Place two take-profit sell orders: 50% at TP1, 50% at TP2."""
        tp1_qty = total_qty // 2
        tp2_qty = total_qty - tp1_qty
        if tp1_qty < 1:
            tp1_qty = 0
            tp2_qty = total_qty
        for price, qty in [(self.tp1_price, tp1_qty), (self.tp2_price, tp2_qty)]:
            if qty <= 0:
                continue
            oid = self._new_order_id()
            sell_order = Order(
                order_id=oid,
                market_ticker=market_ticker,
                side=side,
                order_type=OrderType.SELL,
                price=price,
                quantity=qty,
                placed_at=self._last_bar_ts,
            )
            self.open_orders[oid] = sell_order

    def _cancel_opposite_buys(self, market_ticker: str, filled_side: Side):
        opp = Side.NO if filled_side == Side.YES else Side.YES
        for o in self.open_orders.values():
            if (o.market_ticker == market_ticker
                    and o.side == opp
                    and o.order_type == OrderType.BUY
                    and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)):
                o.cancel()

    def _cancel_buy_orders(self, market_ticker: str):
        for o in self.open_orders.values():
            if (o.market_ticker == market_ticker
                    and o.order_type == OrderType.BUY
                    and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)):
                o.cancel()

    def _cancel_all_orders(self, market_ticker: str):
        for o in self.open_orders.values():
            if (o.market_ticker == market_ticker
                    and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)):
                o.cancel()

    # ── Helpers ──

    def _pos_key(self, ticker, side):
        return f"{ticker}:{side.value}"

    def _new_order_id(self):
        return str(uuid.uuid4())[:8]

    def _update_drawdown(self):
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        dd = self.peak_bankroll - self.bankroll
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def _risk_dollars(self) -> float:
        return self.bankroll * self.risk_pct

    def _volume_cap(self, bar: CandleBar) -> int:
        if self.volume_fill_pct <= 0 or bar.volume <= 0:
            return 0
        return max(1, int(bar.volume * self.volume_fill_pct))
