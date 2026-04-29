from stink_bid_config import StinkBidConfig
from stink_bid_strategy import StinkBidStrategy


def test_snap_to_tick_rejects_out_of_bounds():
    p = StinkBidStrategy.snap_to_tick(0.004, 0.01, 0.01, 0.99)
    assert p is None
    p2 = StinkBidStrategy.snap_to_tick(0.106, 0.01, 0.01, 0.99)
    assert p2 == 0.11


def test_side_selection_yes_only():
    cfg = StinkBidConfig(sides=("YES",), num_levels=3, bad_prices_yes=(0.01, 0.02, 0.03))
    strat = StinkBidStrategy(bankroll=1000.0, config=cfg)
    orders = strat.submit_open_ladder("M1", open_ts=1000, close_ts=1900)
    assert len(orders) == 3
    assert all(o.side == "YES" for o in orders)


def test_cancel_timeout_marks_unfilled_orders():
    cfg = StinkBidConfig(
        sides=("YES",),
        num_levels=1,
        bad_prices_yes=(0.01,),
        cancel_timeout_ms=250,
        max_contracts_per_level=1,
    )
    strat = StinkBidStrategy(bankroll=1000.0, config=cfg)
    strat.submit_open_ladder("M1", open_ts=1000, close_ts=1900)
    strat.process_trade_window("M1", open_ts=1000, close_ts=1900, trades=[], used_trade_data=True)
    cancels = [r for r in strat.trade_log if r.get("action") == "cancel"]
    assert len(cancels) == 1
    assert cancels[0]["note"] == "timeout_cancel"


def test_replace_canceled_resubmits_same_level():
    cfg = StinkBidConfig(
        sides=("YES",),
        num_levels=1,
        bad_prices_yes=(0.01,),
        cancel_timeout_ms=250,
        max_contracts_per_level=1,
        replace_canceled=True,
    )
    strat = StinkBidStrategy(bankroll=1000.0, config=cfg)
    strat.submit_open_ladder("M1", open_ts=1000, close_ts=1900)
    strat.process_trade_window("M1", open_ts=1000, close_ts=1900, trades=[], used_trade_data=True)
    submits = [r for r in strat.trade_log if r.get("action") == "submit"]
    replacements = [r for r in submits if r.get("note") == "replacement"]
    cancels = [r for r in strat.trade_log if r.get("action") == "cancel"]
    assert len(submits) >= 2
    assert len(replacements) >= 1
    assert len(cancels) >= 2


def test_qty_cap_by_notional_per_market():
    cfg = StinkBidConfig(
        sides=("YES",),
        num_levels=5,
        bad_prices_yes=(0.10, 0.20, 0.30, 0.40, 0.50),
        max_contracts_per_level=5,
        max_notional_per_market=0.25,
    )
    strat = StinkBidStrategy(bankroll=1000.0, config=cfg)
    orders = strat.submit_open_ladder("M1", open_ts=1000, close_ts=1900)
    notional = sum(o.qty_total * o.price for o in orders)
    assert notional <= cfg.max_notional_per_market + 1e-9
