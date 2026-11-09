"""Short-term setups: do any "confluences" give an edge on a 1–2 day trade?

PRE-REGISTERED (written before the first run; do not add or tune after seeing results):
  Trade: signal known at the CLOSE of day t  ->  buy at the OPEN of day D = t+1  ->  sell at the close of
         D (same day), D+1 (next day) or D+2 (two days later).
  Costs: 5 bps round trip (liquid ETFs and mega-caps), subtracted from every trade.
  Universe: 4 index ETFs + 11 sector ETFs + 20 mega-caps. In-sample 2010–2018, out-of-sample 2019–today.
  Same-day trades are averaged per date first, so 20 stocks signaling on one crash day count as one result.
  VERIFIED = beats buying the same stocks on random days with t >= 2 in BOTH periods, and is profitable
             after costs out-of-sample. With 48 tests, ~2 would pass t >= 2 in one period by luck alone,
             which is why both periods are required.

SHORT SIDE (pre-registered 2026-10-16, before its first run): mirror-image setups for a falling stock.
  Short trade: sell short at the open of D, buy back at the close of D / D+1 / D+2. Return = −(price change)
  minus 5 bps costs going in and out (borrow fees on these liquid names are ~0 over 2 days and are ignored).
  Same universe, periods, date clustering and VERIFIED rule; the benchmark is shorting on random days.

Run: .venv/bin/python shortterm.py   -> prints every result and writes shortterm_results.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
yf.set_tz_cache_location(__import__("tempfile").mkdtemp(prefix="yf-tz-"))  # own cache per process: a shared one fails when two run at once

UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLC", "XLV", "XLF", "XLE", "XLB", "XLI", "XLY", "XLP", "XLU", "XLRE",
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH",
            "JNJ", "V", "PG", "HD", "KO", "PEP", "COST", "WMT", "BAC", "AVGO"]
COST = 0.0005
SPLIT = "2019-01-01"
HOLDS = {0: "sell same day", 1: "sell next day", 2: "sell 2 days later"}

# name -> (plain description, needs to be checked at the open?, side)
SETUPS = {
    "rsi2_low": ("RSI(2) below 10: sharply oversold over the last 2 days", False, "long"),
    "down3": ("Closed lower 3 days in a row", False, "long"),
    "big_drop": ("Fell more than 2× its normal daily move", False, "long"),
    "close_near_low": ("Closed in the bottom 10% of the day's range", False, "long"),
    "above_200": ("Price above its 200-day average (uptrend)", False, "long"),
    "vix_high": ("VIX (fear index) 10%+ above its 20-day average", False, "long"),
    "turn_of_month": ("Buy day is the last trading day of the month or one of the first 3", False, "long"),
    "gap_down": ("Opens 1%+ below the previous close", True, "long"),
    "vol_spike": ("Volume more than 2× its 20-day average", False, "long"),
    "high_52w": ("Closed at a 52-week high", False, "long"),
    "rsi2_low+above_200": ("Oversold (RSI 2 < 10) in an uptrend", False, "long"),
    "close_near_low+above_200": ("Closed near the low, in an uptrend", False, "long"),
    "rsi2_low+above_200+vix_high": ("Oversold in an uptrend while fear is elevated", False, "long"),
    "down3+above_200": ("3 down days in an uptrend", False, "long"),
    "turn_of_month+above_200": ("Turn of the month, in an uptrend", False, "long"),
    "gap_down+above_200": ("Gaps down 1%+ at the open, in an uptrend", True, "long"),
    # short side: the stock is stretched up, or falling in a downtrend
    "rsi2_high": ("RSI(2) above 90: sharply overbought over the last 2 days", False, "short"),
    "up3": ("Closed higher 3 days in a row", False, "short"),
    "big_rise": ("Rose more than 2× its normal daily move", False, "short"),
    "close_near_high": ("Closed in the top 10% of the day's range", False, "short"),
    "below_200": ("Price below its 200-day average (downtrend)", False, "short"),
    "gap_up": ("Opens 1%+ above the previous close", True, "short"),
    "low_52w": ("Closed at a 52-week low", False, "short"),
    "rsi2_high+below_200": ("Overbought (RSI 2 > 90) in a downtrend", False, "short"),
    "up3+below_200": ("3 up days in a downtrend", False, "short"),
    "close_near_high+below_200": ("Closed near the high, in a downtrend", False, "short"),
    "gap_up+below_200": ("Gaps up 1%+ at the open, in a downtrend", True, "short"),
    "big_drop+below_200": ("Big drop in a downtrend (keeps falling?)", False, "short"),
    "down3+below_200": ("3 down days in a downtrend (keeps falling?)", False, "short"),
}


def rsi(c, n=2):
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def signals(df, vix):
    """Boolean setups indexed by signal day t (known at t's close; gap_down needs D's open)."""
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    r = c.pct_change()
    idx = c.index
    month = pd.Series(idx.to_period("M"), index=idx)
    pos = month.groupby(month).cumcount()
    last = month != month.shift(-1)
    tom_d = (pos < 3) | last                       # is this date a turn-of-month day?
    vix = vix.reindex(idx).ffill()
    s = {
        "rsi2_low": rsi(c) < 10,
        "down3": (r < 0) & (r.shift(1) < 0) & (r.shift(2) < 0),
        "big_drop": r < -2 * r.rolling(20).std().shift(1),
        "close_near_low": ((c - l) / (h - l)) < 0.1,
        "above_200": c > c.rolling(200).mean(),
        "vix_high": vix > 1.1 * vix.rolling(20).mean(),
        "turn_of_month": tom_d.shift(-1, fill_value=False),   # entry day D = t+1
        "gap_down": o.shift(-1) < 0.99 * c,
        "vol_spike": v > 2 * v.rolling(20).mean(),
        "high_52w": c >= c.rolling(252).max(),
        "rsi2_high": rsi(c) > 90,
        "up3": (r > 0) & (r.shift(1) > 0) & (r.shift(2) > 0),
        "big_rise": r > 2 * r.rolling(20).std().shift(1),
        "close_near_high": ((c - l) / (h - l)) > 0.9,
        "below_200": c < c.rolling(200).mean(),
        "gap_up": o.shift(-1) > 1.01 * c,
        "low_52w": c <= c.rolling(252).min(),
    }
    for combo in SETUPS:
        if "+" in combo:
            parts = combo.split("+")
            s[combo] = np.logical_and.reduce([s[p] for p in parts])
    return pd.DataFrame(s).fillna(False).astype(bool)


def stats(x, base_mean):
    """x = trade returns (net), index = date. Average per date first, then t-stats across dates."""
    if len(x) < 20:
        return None
    d = x.groupby(level=0).mean()
    se = d.std() / np.sqrt(len(d))
    return {"trades": int(len(x)), "days": int(len(d)), "win": float((x > 0).mean()), "avg": float(d.mean()),
            "t": float(d.mean() / se), "edge": float(d.mean() - base_mean), "t_edge": float((d.mean() - base_mean) / se),
            "worst": float(x.min())}


def run():
    raw = yf.download(UNIVERSE + ["^VIX"], start="2009-01-01", auto_adjust=True, group_by="ticker", progress=False)
    vix = raw["^VIX"]["Close"]
    rows = []
    coverage = {}
    for t in UNIVERSE:
        df = raw[t].dropna(subset=["Open", "Close"])
        coverage[t] = (str(df.index[0].date()), str(df.index[-1].date()), len(df))
        sig = signals(df, vix)
        fwd = {h: df["Close"].shift(-(1 + h)) / df["Open"].shift(-1) - 1 - COST for h in HOLDS}
        rows.append(pd.concat([sig, pd.DataFrame({f"fwd{h}": f for h, f in fwd.items()})], axis=1).assign(ticker=t))
    data = pd.concat(rows)
    data = data[data.index >= "2010-01-01"]
    periods = {"in_sample": data.index < SPLIT, "out_of_sample": data.index >= SPLIT}

    results = []
    for h, hname in HOLDS.items():
        col = f"fwd{h}"
        for name, (desc, at_open, side) in SETUPS.items():
            res = {"setup": name, "desc": desc, "at_open": at_open, "side": side, "hold": h, "hold_name": hname}
            ret = data[col] if side == "long" else -(data[col] + COST) - COST  # short: gain when the price falls
            for pname, pmask in periods.items():
                all_ = ret[pmask].dropna()
                base = all_.groupby(level=0).mean().mean()
                res[pname] = stats(ret[pmask & data[name]].dropna(), base)
                res[pname + "_baseline"] = float(base)
            i, o = res["in_sample"], res["out_of_sample"]
            res["verified"] = bool(i and o and i["t_edge"] >= 2 and o["t_edge"] >= 2 and o["avg"] > 0)
            results.append(res)
    return {"generated": str(pd.Timestamp.now().date()), "cost": COST, "split": SPLIT, "coverage": coverage,
            "results": results}


def show(out):
    print(f"coverage: {len(out['coverage'])} tickers, from", min(v[0] for v in out["coverage"].values()),
          "to", max(v[1] for v in out["coverage"].values()))
    fmt = lambda s: "      —" if not s else f"{s['days']:5d}d {s['win']*100:4.0f}% {s['avg']*100:+6.3f}% t{s['t_edge']:+5.1f}"
    print(f"{'setup':32s} {'side':5s} {'hold':18s} {'IN-SAMPLE days win avg edge-t':35s} {'OUT-OF-SAMPLE days win avg edge-t':35s} verified")
    for r in out["results"]:
        print(f"{r['setup']:32s} {r['side']:5s} {r['hold_name']:18s} {fmt(r['in_sample']):35s} {fmt(r['out_of_sample']):35s} {'YES' if r['verified'] else ''}")
    for (h, hname), side in [(x, sd) for x in HOLDS.items() for sd in ("long", "short")]:
        r = next(x for x in out["results"] if x["hold"] == h and x["side"] == side)
        print(f"baseline ({side} every day), {hname}: in {r['in_sample_baseline']*100:+.3f}%  out {r['out_of_sample_baseline']*100:+.3f}%")


if __name__ == "__main__":
    out = run()
    Path(__file__).with_name("shortterm_results.json").write_text(json.dumps(out, indent=2))
    show(out)
