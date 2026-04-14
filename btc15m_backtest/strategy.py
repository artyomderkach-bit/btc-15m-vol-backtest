"""
Strategy: Manages order placement, fill handling, cancellation, and settlement
for the 10¢ buy / 40¢ sell Kalshi prediction market strategy.
"""
import math
import random
import uuid
from typing import Dict, List, Optional, Tuple

from models import Order, Position, Side, OrderType, OrderStatus, CandleBar

MAKER_RATE = 0.0175

# Risk sizing (matches live bot)
STATIC_RISK_DOL = 7.0
RISK_THRESHOLD_GAIN = 100


def kalshi_fee(contracts: int, price: float) -> float:
    """Kalshi maker fee: ceil(rate * contracts * price * (1 - price) * 100) / 100."""
    if contracts <= 0:
        return 0.0
    return math.ceil(MAKER_RATE * contracts * price * (1 - price) * 100) / 100


class Strategy:
    def __init__(self, bankroll: float, risk_pct: float = 0.01,
                 buy_price: float = 0.10, sell_price: float = 0.40,
                 entry_cutoff_seconds: int = 240,
                 circuit_breaker_pct: float = 0.0,
                 volume_fill_pct: float = 0.10,
                 tp_fill_rate: float = 1.0,
                 stop_loss_price: float = 0.0,
                 trailing_stop_trigger: float = 0.0,
                 trailing_stop_floor: float = 0.0,
                 vix_min: float = 0.0,
                 weekend_size_mult: float = 1.0,
                 single_side: str = 'both',
                 session_hours: tuple = (0, 24),
                 max_consecutive_losses_halt: int = 0,
                 mid_filter_min: Optional[float] = None,
                 mid_filter_max: Optional[float] = None,
                 mid_filter_bars: int = 4):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.risk_pct = risk_pct
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.entry_cutoff_seconds = entry_cutoff_seconds
        self.circuit_breaker_pct = circuit_breaker_pct
        self.volume_fill_pct = volume_fill_pct
        self.tp_fill_rate = max(0.0, min(1.0, tp_fill_rate))
        self.halted = False

        # New optimization parameters (backward-compatible defaults)
        self.stop_loss_price = stop_loss_price
        self.trailing_stop_trigger = trailing_stop_trigger
        self.trailing_stop_floor = trailing_stop_floor
        self.vix_min = vix_min
        self.weekend_size_mult = weekend_size_mult
        self.single_side = single_side
        self.session_hours = session_hours
        self.max_consecutive_losses_halt = max_consecutive_losses_halt
        self._consec_loss_halted = False

        self.mid_filter_min = mid_filter_min
        self.mid_filter_max = mid_filter_max
        self.mid_filter_bars = max(1, int(mid_filter_bars))
        self._mid_waiting: Dict[str, int] = {}

        self.open_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}  # key: "ticker:side"
        self.trade_log: List[dict] = []
        self._last_bar_ts: int = 0
        self._last_buy_fill_ts: Dict[str, int] = {}  # market -> ts of last buy fill

        self.peak_bankroll = bankroll
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # UTC calendar day (YYYYMMDD) for VIX lookup — set in on_market_open from open_ts
        self._current_day: Optional[int] = None

        # VIX data injected by the engine (date_int -> vix_level)
        self._vix_by_date: Dict[int, float] = {}
        self._trailing_triggered: Dict[str, bool] = {}

    def _pos_key(self, ticker, side):
        return f"{ticker}:{side.value}"

    def _new_order_id(self):
        return str(uuid.uuid4())[:8]

    def _check_circuit_breaker(self):
        if self.circuit_breaker_pct <= 0:
            return
        if self.bankroll <= self.initial_bankroll * (1 - self.circuit_breaker_pct):
            if not self.halted:
                print(f"  CIRCUIT BREAKER: Bankroll ${self.bankroll:.2f} below "
                      f"${self.initial_bankroll * (1 - self.circuit_breaker_pct):.2f} threshold. Halting.")
                self.halted = True

    def _update_drawdown(self):
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        dd = self.peak_bankroll - self.bankroll
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def _risk_dollars(self) -> float:
        """Risk budget per trade: bankroll * risk_pct."""
        return self.bankroll * self.risk_pct

    @property
    def mid_filter_enabled(self) -> bool:
        return (
            self.mid_filter_min is not None
            and self.mid_filter_max is not None
            and self.mid_filter_min < self.mid_filter_max
        )

    def _entry_cutoff_elapsed(self) -> int:
        """Seconds from market open after which resting buy orders are cancelled."""
        extra = self.mid_filter_bars * 60 if self.mid_filter_enabled else 0
        return self.entry_cutoff_seconds + extra

    def _mid_from_bar(self, bar: CandleBar) -> Optional[float]:
        """Midpoint of YES quote at bar close (matches microstructure feature)."""
        ac, bc = bar.ask_close, bar.bid_close
        if ac is not None and bc is not None and ac > bc:
            return (ac + bc) / 2.0
        if bar.price_close is not None:
            return bar.price_close
        return None

    def _place_buys(self, market_ticker: str, placed_at: int, open_ts: int):
        """Place resting buy orders (both sides unless filtered)."""
        import datetime as _dt
        dt = _dt.datetime.utcfromtimestamp(open_ts)

        risk_dollars = self._risk_dollars()
        if self.weekend_size_mult != 1.0 and dt.weekday() >= 5:
            risk_dollars *= self.weekend_size_mult

        qty = max(1, int(risk_dollars / self.buy_price))

        if self.single_side == 'yes':
            sides = [Side.YES]
        elif self.single_side == 'no':
            sides = [Side.NO]
        else:
            sides = [Side.YES, Side.NO]

        for side in sides:
            oid = self._new_order_id()
            order = Order(
                order_id=oid,
                market_ticker=market_ticker,
                side=side,
                order_type=OrderType.BUY,
                price=self.buy_price,
                quantity=qty,
                placed_at=placed_at,
            )
            self.open_orders[oid] = order

    # ── Market lifecycle ──

    def set_vix_data(self, vix_by_date: Dict[int, float]):
        self._vix_by_date = vix_by_date

    def on_market_open(self, market_ticker: str, open_ts: int):
        """Place buy orders at open, or defer until mid-price filter (N bars)."""
        if self.halted:
            return

        import datetime as _dt
        self._current_day = int(_dt.datetime.utcfromtimestamp(open_ts).strftime("%Y%m%d"))

        if (self.max_consecutive_losses_halt > 0
                and self.consecutive_losses >= self.max_consecutive_losses_halt):
            self._consec_loss_halted = True
            return
        if self._consec_loss_halted and self.consecutive_losses > 0:
            return
        self._consec_loss_halted = False

        if self.vix_min > 0 and self._current_day:
            vix_today = self._vix_by_date.get(self._current_day, 0)
            if vix_today < self.vix_min:
                return

        dt = _dt.datetime.utcfromtimestamp(open_ts)
        h_start, h_end = self.session_hours
        hour = dt.hour
        if h_start < h_end:
            if not (h_start <= hour < h_end):
                return
        elif h_start > h_end:
            if not (hour >= h_start or hour < h_end):
                return

        if self.mid_filter_enabled:
            self._mid_waiting[market_ticker] = 0
            return

        self._place_buys(market_ticker, placed_at=open_ts, open_ts=open_ts)

    def on_candle(self, market_ticker: str, bar: CandleBar, open_ts: int, close_ts: int):
        """Process one 1-min bar. Check fills, enforce cutoff, stop loss."""
        if self.halted:
            return

        if market_ticker in self._mid_waiting:
            self._mid_waiting[market_ticker] += 1
            n = self._mid_waiting[market_ticker]
            if n < self.mid_filter_bars:
                return
            del self._mid_waiting[market_ticker]
            mid = self._mid_from_bar(bar)
            lo, hi = self.mid_filter_min, self.mid_filter_max
            if mid is None or mid < lo or mid > hi:
                self.trade_log.append({
                    'market': market_ticker,
                    'side': '',
                    'action': 'skip_mid_filter',
                    'qty': 0,
                    'entry_price': None,
                    'exit_price': None,
                    'pnl': 0.0,
                    'bankroll': self.bankroll,
                    'ts': bar.ts,
                    'mid': mid,
                    'mid_lo': lo,
                    'mid_hi': hi,
                })
                return
            self._place_buys(market_ticker, placed_at=bar.ts, open_ts=open_ts)

        self._last_bar_ts = bar.ts
        elapsed = bar.ts - open_ts
        past_cutoff = elapsed > self._entry_cutoff_elapsed()

        if past_cutoff:
            self._cancel_buy_orders(market_ticker)

        if not past_cutoff:
            self._check_buy_fills(market_ticker, bar)

        self._check_sell_fills(market_ticker, bar)

        # Stop loss / trailing stop: force-sell open positions
        self._check_stop_loss(market_ticker, bar)

    def on_market_settle(self, market_ticker: str, result: str):
        """
        Settle any remaining position. result is 'yes' or 'no' (winning side).
        Cancel any remaining open orders first.
        """
        self._mid_waiting.pop(market_ticker, None)
        self._cancel_all_orders(market_ticker)

        for side in [Side.YES, Side.NO]:
            pk = self._pos_key(market_ticker, side)
            pos = self.positions.get(pk)
            if pos and pos.quantity > 0:
                won = (side.value == result)
                qty = pos.quantity
                entry = pos.avg_entry

                # Bankroll already had cost deducted on buy.
                # Settlement just adds back the payout: $1/contract if won, $0 if lost.
                payout = qty * 1.0 if won else 0.0
                self.bankroll += payout

                # P&L for logging: net profit/loss including entry cost
                pnl = payout - qty * entry
                pos.realized_pnl += pnl
                pos.quantity = 0

                self._update_drawdown()
                self._check_circuit_breaker()

                if pnl < 0:
                    self.consecutive_losses += 1
                    self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
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

    # ── Fill checks ──

    def _check_buy_fills(self, market_ticker: str, bar: CandleBar):
        """Check if any buy orders fill on this bar."""
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
            self._check_circuit_breaker()

            # Cancel opposite side buy
            self._cancel_opposite_buys(market_ticker, order.side)

            # Place/update sell order for total filled quantity
            self._update_sell_order(market_ticker, order.side)

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
        """Check if any sell orders fill on this bar."""
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
                    self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

                self._update_drawdown()

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
                })

    def _volume_cap(self, bar: CandleBar) -> int:
        """Max contracts we can realistically fill on this bar."""
        if self.volume_fill_pct <= 0 or bar.volume <= 0:
            return 0
        return max(1, int(bar.volume * self.volume_fill_pct))

    def _compute_buy_fill(self, order: Order, bar: CandleBar) -> int:
        """
        Only use TRADE prices (not bid/ask offers) to confirm fills.
        YES buy at 10¢: a trade happened at 10¢ or below (price_low <= 10¢).
        NO buy at 10¢: a trade happened at 90¢ or above (price_high >= 90¢).
        Fill capped at volume_fill_pct of bar volume.
        """
        if order.remaining <= 0 or order.status == OrderStatus.CANCELLED:
            return 0

        matched = False
        if order.side == Side.YES:
            if bar.price_low is not None and bar.price_low <= order.price:
                matched = True
        else:  # NO
            yes_level = 1.0 - order.price  # 0.90
            if bar.price_high is not None and bar.price_high >= yes_level:
                matched = True

        if not matched:
            return 0
        return min(order.remaining, self._volume_cap(bar))

    def _compute_sell_fill(self, order: Order, bar: CandleBar) -> int:
        """
        Sell fill rules:
        1. Cannot fill on the same bar the sell order was placed.
        2. Cannot fill on ANY bar where a buy also filled (same candle guard).
        3. YES sell at TP: price_high >= sell_price (YES traded at that level+).
        4. NO sell at TP: price_low <= (1 - sell_price) in YES terms (NO at TP+).
        Fill capped at volume_fill_pct of bar volume.
        """
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
        else:  # NO
            yes_level = 1.0 - order.price  # 0.67
            if bar.price_low is not None and bar.price_low <= yes_level:
                matched = True

        if not matched:
            return 0
        # Model queue position: when price touches TP, we only get filled
        # tp_fill_rate of the time (e.g. 0.85 = 85% fill rate).
        if self.tp_fill_rate < 1.0 and random.random() >= self.tp_fill_rate:
            return 0
        return min(order.remaining, self._volume_cap(bar))

    # ── Order management ──

    def _cancel_opposite_buys(self, market_ticker: str, filled_side: Side):
        """Cancel buy orders on the opposite side after a fill."""
        opp = Side.NO if filled_side == Side.YES else Side.YES
        to_cancel = [o for o in self.open_orders.values()
                     if o.market_ticker == market_ticker
                     and o.side == opp
                     and o.order_type == OrderType.BUY
                     and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)]
        for o in to_cancel:
            o.cancel()

    def _update_sell_order(self, market_ticker: str, side: Side):
        """Cancel existing sell, place new one for total position quantity."""
        # Cancel existing sells for this side
        for o in list(self.open_orders.values()):
            if (o.market_ticker == market_ticker and o.side == side
                    and o.order_type == OrderType.SELL
                    and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)):
                o.cancel()

        pk = self._pos_key(market_ticker, side)
        pos = self.positions.get(pk)
        if not pos or pos.quantity <= 0:
            return

        oid = self._new_order_id()
        sell_order = Order(
            order_id=oid,
            market_ticker=market_ticker,
            side=side,
            order_type=OrderType.SELL,
            price=self.sell_price,
            quantity=pos.quantity,
            placed_at=self._last_bar_ts,  # track when placed to prevent same-bar fills
        )
        self.open_orders[oid] = sell_order

    def _cancel_buy_orders(self, market_ticker: str):
        """Cancel only resting buy orders for a market (leave sells active)."""
        for o in self.open_orders.values():
            if (o.market_ticker == market_ticker
                    and o.order_type == OrderType.BUY
                    and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)):
                o.cancel()

    def _cancel_all_orders(self, market_ticker: str):
        """Cancel all open orders for a market."""
        for o in self.open_orders.values():
            if o.market_ticker == market_ticker and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL):
                o.cancel()

    def _check_stop_loss(self, market_ticker: str, bar: CandleBar):
        """Force-sell positions that hit stop loss or trailing stop."""
        for side in [Side.YES, Side.NO]:
            pk = self._pos_key(market_ticker, side)
            pos = self.positions.get(pk)
            if not pos or pos.quantity <= 0:
                continue

            # Current YES price approximation from the bar
            if side == Side.YES:
                current_price = bar.price_close
                worst_price = bar.price_low
            else:
                current_price = (1.0 - bar.price_close) if bar.price_close is not None else None
                worst_price = (1.0 - bar.price_high) if bar.price_high is not None else None

            if current_price is None or worst_price is None:
                continue

            should_stop = False
            exit_price = worst_price

            # Hard stop loss: if position value dropped below threshold
            if self.stop_loss_price > 0 and worst_price <= self.stop_loss_price:
                should_stop = True
                exit_price = self.stop_loss_price

            # Trailing stop: once price hits trigger, lock floor
            trail_key = f"{market_ticker}:{side.value}"
            if self.trailing_stop_trigger > 0 and self.trailing_stop_floor > 0:
                if current_price >= self.trailing_stop_trigger:
                    self._trailing_triggered[trail_key] = True
                if self._trailing_triggered.get(trail_key) and worst_price <= self.trailing_stop_floor:
                    should_stop = True
                    exit_price = self.trailing_stop_floor

            if not should_stop:
                continue

            # Cancel any resting sell orders for this position
            for o in list(self.open_orders.values()):
                if (o.market_ticker == market_ticker and o.side == side
                        and o.order_type == OrderType.SELL
                        and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)):
                    o.cancel()

            qty = pos.quantity
            fee = kalshi_fee(qty, exit_price)
            pnl = pos.sell(qty, exit_price, fee)
            self.bankroll += qty * exit_price - fee

            if pnl > 0:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

            self._update_drawdown()
            self._check_circuit_breaker()

            self.trade_log.append({
                'market': market_ticker,
                'side': side.value,
                'action': 'stop_loss',
                'qty': qty,
                'entry_price': pos.avg_entry,
                'exit_price': exit_price,
                'pnl': pnl,
                'bankroll': self.bankroll,
                'ts': bar.ts,
            })
