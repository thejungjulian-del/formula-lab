# Formula Lab

A stock-formula lab that turned into a small automated trading desk. Every afternoon it ranks S&P 500 stocks and picks ten to hold from 15:59 ET to the next morning's open. A simulator trades every plan on its own and logs the result, and one click can send the same plan to an Alpaca paper account. There was a daytime leg too, but at the prices you can actually get it kept only 17% of its backtested edge, so I paused it. The app also has a formula builder with about twenty baseline variables per ticker (momentum, reversal, value, quality, volatility, spread and so on) and a forecast page.

## Run it

Python 3.12 with numpy, pandas, yfinance, requests and lxml.

```
python3 -m venv .venv
.venv/bin/pip install numpy pandas yfinance requests lxml
.venv/bin/python app.py
```

Then open http://localhost:8802. Price data comes from Yahoo and is cached in `data/`, so the first scan is slow.

Sending orders is optional. Copy `.env.example` to `.env` and add Alpaca paper keys. `broker.py` places orders for a followed plan, and `sim.py` simulates it without any keys.

Tests: `python test_vars.py` and `python test_broker.py`.

## Status

Paper trading and simulation only. The expected +7.7% a year (51% winning days, about a 72% chance of being ahead after a year) is a backtest estimate on 2023 onward with real fill times and costs. It's probably optimistic, because that period was used more than once while I was building it. This isn't investment advice.

The research files (`*_results.json`, `search_ledger.json`, the holdout files) are the pre-registered tests behind those numbers, including the ones that failed: minute-level trading, a wider stock list, news and analyst signals, and skipping earnings nights.

Portfolio: https://thejungjulian-del.github.io/portfolio/
