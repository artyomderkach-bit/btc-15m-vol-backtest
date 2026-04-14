#!/bin/bash
# Run backtest - uses .venv from project root
cd "$(dirname "$0")"
../.venv/bin/python run_backtest.py "$@"
