"""
Core models: Order, Position, MarketState.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(Enum):
    YES = 'yes'
    NO = 'no'


class OrderType(Enum):
    BUY = 'buy'
    SELL = 'sell'


class OrderStatus(Enum):
    OPEN = 'open'
    FILLED = 'filled'
    PARTIAL = 'partial'
    CANCELLED = 'cancelled'


@dataclass
class Order:
    order_id: str
    market_ticker: str
    side: Side
    order_type: OrderType
    price: float          # 0-1 fraction (e.g. 0.10 = 10¢)
    quantity: int          # contracts requested
    filled_qty: int = 0
    status: OrderStatus = OrderStatus.OPEN
    placed_at: int = 0    # unix timestamp (seconds)
    filled_at: Optional[int] = None

    @property
    def remaining(self):
        return self.quantity - self.filled_qty

    def fill(self, qty, ts):
        self.filled_qty += qty
        self.filled_at = ts
        if self.filled_qty >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL

    def cancel(self):
        self.status = OrderStatus.CANCELLED


@dataclass
class Position:
    market_ticker: str
    side: Side
    quantity: int = 0
    avg_entry: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    def add(self, qty, price, fee):
        total_cost = self.avg_entry * self.quantity + price * qty
        self.quantity += qty
        self.avg_entry = total_cost / self.quantity if self.quantity else 0
        self.fees_paid += fee

    def sell(self, qty, price, fee):
        pnl = qty * (price - self.avg_entry) - fee
        self.realized_pnl += pnl
        self.quantity -= qty
        self.fees_paid += fee
        return pnl

    def settle(self, won: bool, fee: float = 0.0):
        if won:
            pnl = self.quantity * (1.0 - self.avg_entry) - fee
        else:
            pnl = -self.quantity * self.avg_entry - fee
        self.realized_pnl += pnl
        self.fees_paid += fee
        settled_qty = self.quantity
        self.quantity = 0
        return pnl, settled_qty


@dataclass
class CandleBar:
    """One 1-minute candlestick bar, all prices as 0-1 fractions."""
    ts: int               # end_period_ts (unix seconds)
    price_open: Optional[float] = None
    price_high: Optional[float] = None
    price_low: Optional[float] = None
    price_close: Optional[float] = None
    ask_low: Optional[float] = None
    ask_high: Optional[float] = None
    ask_close: Optional[float] = None
    bid_low: Optional[float] = None
    bid_high: Optional[float] = None
    bid_close: Optional[float] = None
    volume: int = 0
