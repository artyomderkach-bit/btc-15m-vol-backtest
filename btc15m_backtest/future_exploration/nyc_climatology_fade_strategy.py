"""
NYCClimatologyFadeStrategy: Trade NYC daily temperature prediction markets
using a climatology mean-reversion signal.

Root cause of original "0 fills" bug:
    The old version placed limit buy orders at a fixed 30¢.  For extreme
    markets (z > 1.5) the YES price is already 1–10¢ because the market
    correctly reflects climatology — the 30¢ fill level is never reached.

New design:
    1. At market open: compute the climatology probability using a proper
       normal CDF (not just z-score direction).
    2. At the FIRST candle: read the actual YES opening price from the market.
    3. Compute edge = market_price - climo_probability.
    4. If |edge| >= MIN_EDGE, place a limit buy at a small discount to the
       current market price.  This will fill within the first few candles.
    5. Take profit at entry + TP_PROFIT.  Hold remainder to settlement.

Entry:
    BUY NO  if market overprices YES  (market_yes > climo_prob + min_edge)
    BUY YES if market underprices YES (market_yes < climo_prob - min_edge)

Fill:
    BUY NO  at P:  fills when YES >= 1 - P  (YES rises slightly from open)
    BUY YES at P:  fills when YES <= P       (YES dips slightly from open)

    We set entry = current market price ± entry_spread, so fills happen
    quickly (typically within the first few 1-minute candles).
"""
import math
import re
import random
import uuid
import datetime as _dt
from typing import Dict, List, Optional

from models import Order, Position, Side, OrderType, OrderStatus, CandleBar

MAKER_RATE = 0.0175

# 1991-2020 NWS 30-year daily high temperature normals (Central Park).
NYC_MONTHLY_MEAN = {
    1: 39, 2: 42, 3: 50, 4: 62, 5: 72, 6: 80,
    7: 85, 8: 83, 9: 76, 10: 65, 11: 54, 12: 44,
}

# Per-month standard deviation of daily high temp (°F) — Central Park.
# Winter is more variable (polar vortex); summer is more stable.
NYC_MONTHLY_STD = {
    1: 9,  2: 9,  3: 9,  4: 8,  5: 7,  6: 6,
    7: 5,  8: 5,  9: 6, 10: 7, 11: 8, 12: 8,
}


def kalshi_fee(contracts: int, price: float) -> float:
    if contracts <= 0:
        return 0.0
    return math.ceil(MAKER_RATE * contracts * price * (1 - price) * 100) / 100


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc — no scipy needed."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _climo_prob(threshold_temp: int, month: int, strike_type: str) -> float:
    """
    Climatology probability that the market resolves YES.
      strike_type='greater' → P(temp > threshold)
      strike_type='less'    → P(temp < threshold)
    """
    mean = NYC_MONTHLY_MEAN.get(month, 65)
    std  = NYC_MONTHLY_STD.get(month, 7)
    z = (threshold_temp - mean) / std
    if strike_type == 'less':
        return _norm_cdf(z)        # P(temp < threshold)
    return 1.0 - _norm_cdf(z)     # P(temp > threshold)  [default]


class NYCClimatologyFadeStrategy:

    def __init__(self, bankroll: float, risk_pct: float = 0.02,
                 min_edge: float = 0.08,
                 entry_spread: float = 0.02,
                 tp_profit: float = 0.20,
                 entry_expiry_minutes: int = 60,
                 volume_fill_pct: float = 0.10,
                 tp_fill_rate: float = 1.0):
        """
        min_edge       : minimum |market_prob - climo_prob| to take a trade.
                         0.10 = market must misprice by at least 10 percentage
                         points vs climatology.
        entry_spread   : how many cents better than current market price we
                         bid.  0.02 → buy 2¢ below current YES/NO ask.
                         Small spread → fast fills; large spread → better
                         price but fewer fills.
        tp_profit      : take-profit size in dollars per contract above entry.
                         0.20 → sell at entry + 20¢.
        """
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.risk_pct = risk_pct
        self.min_edge = min_edge
        self.entry_spread = entry_spread
        self.tp_profit = tp_profit
        self.entry_expiry_minutes = entry_expiry_minutes
        self.volume_fill_pct = volume_fill_pct
        self.tp_fill_rate = max(0.0, min(1.0, tp_fill_rate))
        self.halted = False

        # Engine compatibility shims
        self.buy_price  = 0.30
        self.sell_price = 0.50

        self.open_orders: Dict[str, Order] = {}
        self.positions:   Dict[str, Position] = {}
        self.trade_log:   List[dict] = []
        self._last_bar_ts: int = 0
        self._last_buy_fill_ts: Dict[str, int] = {}

        self.peak_bankroll       = bankroll
        self.max_drawdown        = 0.0
        self.consecutive_losses  = 0
        self.max_consecutive_losses = 0

        # Per-market state: stores eligibility and, after first candle, entry
        self._market_signals: Dict[str, dict] = {}
        # Pre-injected data from runner (threshold + strike_type)
        self._injected_info: Dict[str, dict] = {}

        # Strategy counters
        self.entry_fills      = 0
        self.tp_fills         = 0
        self.settlement_wins  = 0
        self.settlement_losses = 0
        self.signals_fired    = 0   # signals generated at first-candle check
        self.no_edge_count    = 0   # markets where |edge| < min_edge
        self.markets_no_threshold = 0

    # ── Runner injects market metadata before on_market_open ──

    def inject_market_info(self, market_ticker: str,
                           threshold_temp: int,
                           strike_type: str = 'greater'):
        self._injected_info[market_ticker] = {
            'threshold_temp': threshold_temp,
            'strike_type': strike_type,
        }

    def inject_threshold(self, market_ticker: str, threshold_temp: int):
        """Backward-compatible shim."""
        info = self._injected_info.get(market_ticker, {})
        info['threshold_temp'] = threshold_temp
        if 'strike_type' not in info:
            info['strike_type'] = 'greater'
        self._injected_info[market_ticker] = info

    # ── Engine interface ──

    def on_market_open(self, market_ticker: str, open_ts: int):
        """
        Record market eligibility.  No order is placed yet — we wait for
        the first candle to read the actual opening price.
        """
        if self.halted:
            return

        info = self._injected_info.get(market_ticker)
        if not info:
            threshold_temp = self._parse_threshold(market_ticker)
            if threshold_temp is None:
                self.markets_no_threshold += 1
                return
            strike_type = 'greater'
        else:
            threshold_temp = info.get('threshold_temp')
            strike_type    = info.get('strike_type', 'greater')
            if threshold_temp is None:
                self.markets_no_threshold += 1
                return

        dt_obj = _dt.datetime.utcfromtimestamp(open_ts)
        month  = dt_obj.month
        climo_p = _climo_prob(threshold_temp, month, strike_type)

        self._market_signals[market_ticker] = {
            'threshold_temp':  threshold_temp,
            'strike_type':     strike_type,
            'climo_prob':      round(climo_p, 4),
            'climo_mean':      NYC_MONTHLY_MEAN.get(month, 65),
            'climo_std':       NYC_MONTHLY_STD.get(month, 7),
            'month':           month,
            'date':            dt_obj.strftime('%Y-%m-%d'),
            'first_bar_done':  False,
            'entry_price':     None,
            'tp_price':        None,
            'signal_side':     None,
        }

    def on_candle(self, market_ticker: str, bar: CandleBar,
                  open_ts: int, close_ts: int):
        if self.halted:
            return
        self._last_bar_ts = bar.ts
        ms = self._market_signals.get(market_ticker)
        if ms is None:
            return

        # ── First candle: compare market opening price to climo probability ──
        first_price = bar.price_open
        if first_price is None:
            # Fall back to midpoint or close for thinly traded candles
            if bar.price_high is not None and bar.price_low is not None:
                first_price = (bar.price_high + bar.price_low) / 2.0
            elif bar.price_close is not None:
                first_price = bar.price_close

        if not ms['first_bar_done'] and first_price is not None:
            ms['first_bar_done'] = True
            yes_open  = first_price
            climo_p   = ms['climo_prob']
            edge      = yes_open - climo_p  # positive → market overprices YES

            if edge >= self.min_edge:
                # Market thinks YES is more likely than climatology → fade: BUY NO
                side    = Side.NO
                # Buy NO at a slight discount to current NO price (= 1 - yes_open)
                entry_p = max(0.02, min(0.95, 1.0 - yes_open - self.entry_spread))
                # Fill when YES price >= 1 - entry_p (≈ yes_open + spread)
            elif edge <= -self.min_edge:
                # Market underprices YES relative to climatology → BUY YES
                side    = Side.YES
                # Buy YES at a slight discount to current YES price
                entry_p = max(0.02, min(0.95, yes_open - self.entry_spread))
                # Fill when YES price <= entry_p (≈ yes_open - spread)
            else:
                ms['no_edge'] = True
                self.no_edge_count += 1
                return

            tp_price = min(0.97, entry_p + self.tp_profit)
            ms['entry_price']  = round(entry_p, 4)
            ms['tp_price']     = round(tp_price, 4)
            ms['signal_side']  = side
            ms['market_prob']  = round(yes_open, 4)
            ms['edge']         = round(edge, 4)
            self.signals_fired += 1

            risk_dollars = self._risk_dollars()
            qty = max(1, int(risk_dollars / entry_p))
            oid = self._new_order_id()
            order = Order(
                order_id=oid,
                market_ticker=market_ticker,
                side=side,
                order_type=OrderType.BUY,
                price=entry_p,
                quantity=qty,
                placed_at=bar.ts,
            )
            self.open_orders[oid] = order

        self._check_buy_fills(market_ticker, bar)
        self._check_sell_fills(market_ticker, bar)

    def on_market_settle(self, market_ticker: str, result: str):
        self._cancel_all_orders(market_ticker)
        sig = self._market_signals.get(market_ticker, {})

        for side in [Side.YES, Side.NO]:
            pk  = self._pos_key(market_ticker, side)
            pos = self.positions.get(pk)
            if not pos or pos.quantity <= 0:
                continue

            won    = (side.value == result)
            qty    = pos.quantity
            entry  = pos.avg_entry
            payout = qty * 1.0 if won else 0.0
            self.bankroll += payout
            pnl = payout - qty * entry
            pos.realized_pnl += pnl
            pos.quantity = 0
            self._update_drawdown()

            if won:
                self.settlement_wins += 1
            else:
                self.settlement_losses += 1

            if pnl < 0:
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(
                    self.max_consecutive_losses, self.consecutive_losses)
            else:
                self.consecutive_losses = 0

            log_entry = {
                'market':      market_ticker,
                'side':        side.value,
                'action':      'settlement',
                'won':         won,
                'qty':         qty,
                'entry_price': entry,
                'exit_price':  1.0 if won else 0.0,
                'pnl':         pnl,
                'bankroll':    self.bankroll,
            }
            log_entry.update(self._sig_fields(sig))
            self.trade_log.append(log_entry)

        self._market_signals.pop(market_ticker, None)

    # ── Fill checks ──

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
            self.entry_fills += 1

            self._place_tp_order(market_ticker, order.side)

            sig = self._market_signals.get(market_ticker, {})
            log_entry = {
                'market':      market_ticker,
                'side':        order.side.value,
                'action':      'buy_fill',
                'qty':         fill_qty,
                'entry_price': order.price,
                'exit_price':  None,
                'pnl':         -(fill_qty * order.price + fee),
                'bankroll':    self.bankroll,
                'ts':          bar.ts,
            }
            log_entry.update(self._sig_fields(sig))
            self.trade_log.append(log_entry)

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
            pk  = self._pos_key(market_ticker, order.side)
            pos = self.positions.get(pk)
            if not pos:
                continue
            pnl = pos.sell(fill_qty, order.price, fee)
            self.bankroll += fill_qty * order.price - fee
            self.tp_fills += 1

            if pnl > 0:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(
                    self.max_consecutive_losses, self.consecutive_losses)
            self._update_drawdown()

            sig = self._market_signals.get(market_ticker, {})
            log_entry = {
                'market':      market_ticker,
                'side':        order.side.value,
                'action':      'sell_fill',
                'qty':         fill_qty,
                'entry_price': pos.avg_entry,
                'exit_price':  order.price,
                'pnl':         pnl,
                'bankroll':    self.bankroll,
                'ts':          bar.ts,
            }
            log_entry.update(self._sig_fields(sig))
            self.trade_log.append(log_entry)

    def _compute_buy_fill(self, order: Order, bar: CandleBar) -> int:
        if order.remaining <= 0 or order.status == OrderStatus.CANCELLED:
            return 0
        if order.placed_at >= bar.ts:
            return 0
        # Entry expiry: cancel entry orders that have been open too long.
        if (self.entry_expiry_minutes > 0
                and bar.ts > order.placed_at + self.entry_expiry_minutes * 60):
            order.cancel()
            return 0
        matched = False
        if order.side == Side.YES:
            # Buy YES at P: fills when YES price dips to P
            if bar.price_low is not None and bar.price_low <= order.price:
                matched = True
        else:
            # Buy NO at P: fills when YES price rises to (1 - P)
            yes_level = 1.0 - order.price
            if bar.price_high is not None and bar.price_high >= yes_level:
                matched = True
        if not matched:
            return 0
        return min(order.remaining, self._volume_cap(bar, order))

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
        return min(order.remaining, self._volume_cap(bar, order))

    # ── Order management ──

    def _place_tp_order(self, market_ticker: str, side: Side):
        pk  = self._pos_key(market_ticker, side)
        pos = self.positions.get(pk)
        if not pos or pos.quantity <= 0:
            return

        # Only one TP order per (market, side) at a time — update quantity if
        # it already exists rather than spawning a new order on every partial fill.
        existing_tp = next(
            (o for o in self.open_orders.values()
             if o.market_ticker == market_ticker
             and o.side == side
             and o.order_type == OrderType.SELL
             and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)),
            None,
        )
        if existing_tp is not None:
            # Keep the price; refresh quantity to match current position size.
            existing_tp.quantity = pos.quantity + existing_tp.filled_qty
            return

        ms = self._market_signals.get(market_ticker, {})
        tp_price = ms.get('tp_price')
        if tp_price is None:
            tp_price = min(0.97, pos.avg_entry + self.tp_profit)

        oid = self._new_order_id()
        sell_order = Order(
            order_id=oid,
            market_ticker=market_ticker,
            side=side,
            order_type=OrderType.SELL,
            price=tp_price,
            quantity=pos.quantity,
            placed_at=self._last_bar_ts,
        )
        self.open_orders[oid] = sell_order

    def _cancel_all_orders(self, market_ticker: str):
        for o in self.open_orders.values():
            if (o.market_ticker == market_ticker
                    and o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)):
                o.cancel()
        # Purge terminal orders (filled/cancelled) to keep the dict lean.
        done = [oid for oid, o in self.open_orders.items()
                if o.status in (OrderStatus.FILLED, OrderStatus.CANCELLED)]
        for oid in done:
            del self.open_orders[oid]

    # ── Helpers ──

    def _sig_fields(self, sig: dict) -> dict:
        if not sig:
            return {}
        return {
            'date':           sig.get('date'),
            'threshold_temp': sig.get('threshold_temp'),
            'climo_mean':     sig.get('climo_mean'),
            'climo_std':      sig.get('climo_std'),
            'climo_prob':     sig.get('climo_prob'),
            'market_prob':    sig.get('market_prob'),
            'edge':           sig.get('edge'),
            'z_score':        round(
                (sig.get('threshold_temp', 0) - sig.get('climo_mean', 65))
                / max(1, sig.get('climo_std', 7)), 3
            ) if sig.get('threshold_temp') is not None else None,
        }

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

    def _volume_cap(self, bar: CandleBar, order: 'Order' = None) -> int:
        if self.volume_fill_pct <= 0:
            return 0
        if bar.volume <= 0:
            # Thin market (no actual trades in this 1-minute bar).  Fill the
            # entire remaining order at once — our small position is trivial
            # relative to a daily market with thousands of contracts of OI.
            return order.remaining if order is not None else 1
        return max(1, int(bar.volume * self.volume_fill_pct))

    @staticmethod
    def _parse_threshold(market_ticker: str) -> Optional[int]:
        m = re.search(r'[-_]T(\d{1,3})(?=[^:\d]|$)', market_ticker)
        if m:
            val = int(m.group(1))
            if -20 <= val <= 130:
                return val
        return None
