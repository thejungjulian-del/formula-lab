"""X1: how much of the daily plan's edge survives real fill times? 7 years of Alpaca 1-minute prices for the
plan's actual picks (2019 -> today), instead of 59 days of Yahoo 5-minute bars.

For every session D and the plan's top-5 picks:
  day leg     buy at the official open vs at 9:31 / 9:33 / 9:35 ET (close of that minute); sell at the official
              close vs at 15:59 (close of the 15:58 bar = last full minute before the auction)
  night leg   buy at the official close vs at 15:59; sell at the next official open vs at 9:31 / 9:33 / 9:35
"kept" = realistic average return / official average return (top-5 portfolio, before costs).

Run: .venv/bin/python exec7.py   -> exec7.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import alpaca_data
import combo
import signal_search as ss

HERE = Path(__file__).parent
ET = "America/New_York"
TIMES = ("0931", "0933", "0935")      # minute-bar END times, same "%H%M" format as the downloaded keys
CACHE = HERE / "data" / "exec7_minutes.pkl"


def picks():
    p = ss.build()
    C = json.loads((HERE / "combo.json").read_text())["trades"]
    p = p[p.index >= "2019-01-01"]
    out = {}
    for leg, trade, y in (("day", "h0", "yh0"), ("night", "ovn", "yovn")):
        s = combo.score(p, C[trade]["components"])
        d = pd.DataFrame({"s": s, "t": p["ticker"], "y": p[y]}).dropna().assign(dt=lambda x: x.index)
        out[leg] = {dt: list(g["t"]) for dt, g in d.sort_values(["dt", "s"], ascending=[True, False]).groupby("dt").head(5).groupby("dt")}
    return out


def window(symbols, day, lo, hi):
    """{(ticker, "HHMM" of bar END): close} for 1-minute bars of `day` between lo and hi (ET clock times)."""
    d = pd.Timestamp(day).tz_localize(ET)
    r = alpaca_data.get("/bars", {"symbols": ",".join(sorted({s.replace("-", ".") for s in symbols})), "timeframe": "1Min",
                                  "start": (d + pd.Timedelta(lo)).isoformat(), "end": (d + pd.Timedelta(hi)).isoformat(),
                                  "limit": 10000, "adjustment": "all", "feed": "sip"})
    out = {}
    for sym, rows in (r.get("bars") or {}).items():
        for b in rows:
            end = pd.Timestamp(b["t"]).tz_convert(ET) + pd.Timedelta(minutes=1)
            out[(sym.replace(".", "-"), end.strftime("%H%M"))] = b["c"]
    return out


def download(P, days):
    """Per session D: morning window (day picks for D + last night's picks) and afternoon window (day picks for D +
    tonight's picks). Cached to disk; resumable."""
    from concurrent.futures import ThreadPoolExecutor
    cache = pd.read_pickle(CACHE) if CACHE.exists() else {}
    prev = {D: t for t, D in zip(days[:-1], days[1:])}
    todo = [D for D in days[1:] if D not in cache]

    def one(D):
        t = prev[D]
        morning = P["day"].get(t, []) + P["night"].get(t, [])        # row t trades on D / was bought at t's close
        afternoon = P["day"].get(t, []) + P["night"].get(D, [])      # sell day picks at 15:59 / buy tonight's at 15:59
        return D, {**window(morning, D.date(), "9h30min", "9h36min"), **window(afternoon, D.date(), "15h57min", "16h")}
    with ThreadPoolExecutor(5) as ex:
        for i, (D, m) in enumerate(ex.map(one, todo), 1):
            cache[D] = m
            if i % 100 == 0:
                pd.to_pickle(cache, CACHE)
                print(f"{i}/{len(todo)} sessions", flush=True)
    pd.to_pickle(cache, CACHE)
    return cache


def measure(P, days, cache):
    prices = __import__("search_data").load("prices")
    recs = {"day": [], "night": []}
    for t, D in zip(days[:-1], days[1:]):
        mD, mt = cache.get(D, {}), cache.get(t, {})
        for s in P["day"].get(t, []):
            px = prices.get(s)
            if px is None or D not in px.index:
                continue
            rec = {"official": float(px.at[D, "Close"] / px.at[D, "Open"] - 1)}
            out_ = mD.get((s, "1559"))
            for k in TIMES:
                inn = mD.get((s, k))
                rec[k] = out_ / inn - 1 if inn and out_ else np.nan
            recs["day"].append(rec)
        for s in P["night"].get(t, []):
            px = prices.get(s)
            if px is None or D not in px.index or t not in px.index:
                continue
            rec = {"official": float(px.at[D, "Open"] / px.at[t, "Close"] - 1)}
            buy = mt.get((s, "1559"))
            for k in TIMES:
                sell = mD.get((s, k))
                rec[k] = sell / buy - 1 if buy and sell else np.nan
            recs["night"].append(rec)
    out = {}
    for leg in ("day", "night"):
        d = pd.DataFrame(recs[leg])
        cov = float(d[list(TIMES)].notna().all(axis=1).mean()) if len(d) else 0.0
        d = d.dropna()
        out[leg] = {"trades": len(d), "coverage": cov, "official": float(d["official"].mean()),
                    **{k: {"avg": float(d[k].mean()), "kept": float(d[k].mean() / d["official"].mean())} for k in TIMES}}
        print(f"{leg}: {len(d)} trades ({cov * 100:.0f}% had minute prices) · official {d['official'].mean() * 100:+.3f}% · " +
              " · ".join(f"{k[1]}:{k[2:]} {d[k].mean() * 100:+.3f}% (kept {d[k].mean() / d['official'].mean() * 100:.0f}%)" for k in TIMES))
    return out


def main(limit=None):
    P = picks()
    days = sorted(set(P["day"]) | set(P["night"]))
    if limit:
        days = days[-limit:]
    cache = download(P, days)
    out = measure(P, days, cache)
    if not limit:
        (HERE / "exec7.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
