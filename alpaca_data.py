"""Historical intraday data from Alpaca (keys in .env). Read-only: never places orders.

  bars      1-minute bars (SIP feed), 2019 -> today, for SPY, QQQ and the 50 most-traded S&P 500 stocks
            -> data/alpaca_1m/<SYMBOL>.pkl  (ET index = bar START; open/high/low/close/volume/vwap)
  spreads   bid/ask spread samples: for each symbol, the quote at 60 random moments across the years
            -> data/alpaca_spreads.pkl  (used as the trading cost: a market order pays the spread per round trip)

Run: .venv/bin/python alpaca_data.py bars | spreads
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import broker
import search_data as sd

DATA = Path(__file__).with_name("data")
BASE = "https://data.alpaca.markets/v2/stocks"


_pace = {"next": 0.0}
_lock = __import__("threading").Lock()


def get(path, params):
    e = broker.env()
    h = {"APCA-API-KEY-ID": e["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": e["ALPACA_SECRET_KEY"]}
    for wait in (0, 5, 30, 60):
        time.sleep(wait)
        with _lock:  # all threads together stay under the free plan's 200 requests a minute
            delay = _pace["next"] - time.time()
            _pace["next"] = max(_pace["next"], time.time()) + 0.32
        if delay > 0:
            time.sleep(delay)
        r = requests.get(BASE + path, headers=h, params=params, timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
    raise RuntimeError("Alpaca kept rate-limiting")


def universe(n=50):
    prices = sd.load("prices")
    dv = {t: float((df["Close"] * df["Volume"]).tail(60).mean()) for t, df in prices.items()}
    return ["SPY", "QQQ"] + [t.replace("-", ".") for t in sorted(dv, key=dv.get, reverse=True)[:n]]  # Alpaca writes BRK.B


def bars(start="2019-01-01", timeframe="1Min"):
    """Saves each symbol to data/alpaca_1m/<SYMBOL>.pkl as soon as it is complete (resumable: done symbols are skipped)."""
    folder = DATA / "alpaca_1m"
    folder.mkdir(exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    todo = [s for s in universe() if not (folder / f"{s}.pkl").exists()]
    def safe(s):
        try:
            one(s, folder / f"{s}.pkl", start, timeframe)
        except Exception as e:  # one bad symbol must not stop the rest
            print(f"{s}: FAILED {str(e)[:150]}", flush=True)
    with ThreadPoolExecutor(5) as ex:  # network waits overlap; the shared pacer keeps the request rate legal
        list(ex.map(safe, todo))


def one(s, f, start, timeframe):
        rows, token, pages = [], None, 0
        while True:
            params = {"symbols": s, "timeframe": timeframe, "start": start, "limit": 10000, "adjustment": "all", "feed": "sip"}
            if token:
                params["page_token"] = token
            d = get("/bars", params)
            rows += (d.get("bars") or {}).get(s, [])
            token, pages = d.get("next_page_token"), pages + 1
            if not token:
                break
        df = pd.DataFrame(rows).rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap"})
        df.index = pd.to_datetime(df.pop("time")).dt.tz_convert("America/New_York")
        df[["open", "high", "low", "close", "volume", "vwap"]].astype("float32").to_pickle(f)
        print(f"{s}: {len(df):,} bars, {pages} pages", flush=True)


def load_bars():
    return {f.stem: pd.read_pickle(f) for f in sorted((DATA / "alpaca_1m").glob("*.pkl"))}


def spreads(samples=60, seed=3):
    """Quoted spread (ask − bid) / mid at random regular-hours moments, per symbol."""
    B = load_bars()
    rng = np.random.default_rng(seed)
    rows = []
    for s, df in B.items():
        hm = df.index.hour * 60 + df.index.minute
        times = df.index[(hm >= 585) & (hm <= 945)]
        for i in rng.choice(len(times), size=min(samples, len(times)), replace=False):
            t = times[i]                                   # keep the ET time zone
            q = get(f"/{s}/quotes", {"start": t.tz_convert("UTC").isoformat(), "end": (t + pd.Timedelta(seconds=30)).tz_convert("UTC").isoformat(),
                                     "limit": 5, "feed": "sip"}).get("quotes") or []
            q = [x for x in q if x.get("bp", 0) > 0 and x.get("ap", 0) >= x.get("bp", 0)]
            if q:
                bp, ap = q[0]["bp"], q[0]["ap"]
                rows.append({"symbol": s, "time": t, "spread": (ap - bp) / ((ap + bp) / 2)})
        print(f"spreads {s}: {sum(1 for r in rows if r['symbol'] == s)} samples", flush=True)
    pd.to_pickle(pd.DataFrame(rows), DATA / "alpaca_spreads.pkl")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
