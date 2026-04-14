# Overlay Format for Equity Plot

To overlay an external time series (e.g. from TradingView, Yahoo Finance) over the equity curve:

## CSV format

```csv
date,value
2025-12-20,1.85
2025-12-21,1.90
...
```

- **date**: `YYYY-MM-DD` or any format pandas can parse
- **value**: numeric (e.g. price, index, ratio)

Alternative column names: `Date`/`Close`, `datetime`/`price`, etc. The script auto-detects.

## Usage

```bash
python plot_equity.py --overlay overlay_sample.csv --overlay-label "Your Series"
```

## Exporting from TradingView

1. Add your indicator/asset to the chart
2. Right-click chart → Export chart data
3. Save as CSV, ensure date + value columns
4. Use as `--overlay your_file.csv`

## Correlation

The plot shows Pearson correlation between daily equity and overlay values over the overlapping date range. Positive = equity tends to rise when overlay rises.
