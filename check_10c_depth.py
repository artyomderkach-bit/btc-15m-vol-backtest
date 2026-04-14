#!/usr/bin/env python3
"""Wrapper: run orderbook depth check from repo root (implementation in btc15m_backtest/scripts/)."""
import os
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_HERE, "btc15m_backtest", "scripts", "check_10c_depth.py"), run_name="__main__")
