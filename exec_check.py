"""Execution check: the tests assume the OFFICIAL open/close price; the plan buys a few minutes after the open and
a few minutes before the close. How much of the edge survives at the prices you can actually get?

Uses Yahoo 5-minute bars (last ~60 days). For each day and the plan's top-10 picks:
  day trade    official: open -> close        realistic: 9:40 ET price -> 3:55 ET price
  overnight    official: close -> next open   realistic: 3:55 ET price -> 9:40 ET next morning
Same comparison for the average stock (no skill), so the generic open/close effect is separated from the picks.
Paired by stock and day; t-stats clustered by day. Only ~40 days: a rough check, not a precise number.

Run: .venv/bin/python exec_check.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
yf.set_tz_cache_location(__import__("tempfile").mkdtemp(prefix="yf-tz-"))  # own cache per process: a shared one fails when two run at once

import combo
import search_data as sd
import signal_search as ss

HERE = Path(__file__).parent
import sys
IN_ET, OUT_ET = (sys.argv[1:3] if len(sys.argv) > 2 else ("09:40", "15:55"))  # default: inside the plan's windows (PT 6:40 / 12:55)
WIDE = "--wide" in sys.argv   # liquid US stocks (wide_check.py) instead of the S&P 500
CACHE = HERE / "data" / ("bars5m_wide.pkl" if WIDE else "bars5m.pkl")


def bars(tickers):
    out = {}
    for i in range(0, len(tickers), 50):
        raw = yf.download(tickers[i:i + 50], period="60d", interval="5m", group_by="ticker", progress=False, threads=True, prepost=False)
        for t in tickers[i:i + 50]:
            try:
                b = raw[t]["Close"].dropna()
            except KeyError:
                continue
            if len(b):
                b.index = b.index.tz_convert("America/New_York")
                out[t] = b
    return out


def at(b, day, hhmm):
    """Close of the 5-minute bar that ENDS at hhmm (the price at hhmm)."""
    ts = pd.Timestamp(f"{day} {hhmm}", tz="America/New_York") - pd.Timedelta(minutes=5)
    return b.get(ts, np.nan)


def main():
    if WIDE:
        import wide_check
        p, C = wide_check.panel()
        prices = sd.load("prices_wide")
    else:
        p = ss.build()
        C = json.loads((HERE / "combo.json").read_text())["trades"]
        prices = sd.load("prices")
    p = p.assign(day=combo.score(p, C["h0"]["components"]), night=combo.score(p, C["ovn"]["components"]))
    if CACHE.exists() and (pd.Timestamp.now() - pd.Timestamp(CACHE.stat().st_mtime, unit="s")).days < 1:
        B = pd.read_pickle(CACHE)
    else:
        if WIDE:  # only what matters: every day's top-10 picks (last ~60 days) + 300 random liquid stocks as the baseline
            r = p[p.index >= p.index.max() - pd.Timedelta(days=90)]
            picks = set()
            for sc in ("day", "night"):
                picks |= set(r.dropna(subset=[sc]).sort_values(sc).groupby(level=0).tail(10)["ticker"])
            rest = sorted(set(r["ticker"]) - picks)
            need = sorted(picks) + list(np.random.default_rng(1).choice(rest, size=min(300, len(rest)), replace=False))
        else:
            need = sorted(p["ticker"].unique())
        B = bars(need)
        pd.to_pickle(B, CACHE)
    first = min(b.index.min() for b in B.values()).normalize().tz_localize(None)
    cal = sd.load("macro")["spy"].index
    rows = []
    for t_day in [d for d in sorted(p.index.unique()) if d >= first]:
        later = cal[cal > t_day]
        if not len(later):
            continue
        D = later[0]
        g = p.loc[[t_day]]
        for trade, sc in (("day", "day"), ("night", "night")):
            ranked = g.dropna(subset=[sc]).sort_values(sc)
            top = set(ranked["ticker"].tail(10))
            for _, r in ranked.iterrows():
                tk, b, px = r["ticker"], B.get(r["ticker"]), prices.get(r["ticker"])
                if b is None or px is None or t_day not in px.index or D not in px.index:
                    continue
                o_D, c_D, c_t = px.at[D, "Open"], px.at[D, "Close"], px.at[t_day, "Close"]
                if trade == "day":
                    off, real = c_D / o_D - 1, at(b, D.date(), OUT_ET) / at(b, D.date(), IN_ET) - 1
                else:
                    off, real = o_D / c_t - 1, at(b, D.date(), IN_ET) / at(b, t_day.date(), OUT_ET) - 1
                if np.isfinite(off) and np.isfinite(real):
                    rows.append({"date": t_day, "trade": trade, "ticker": tk, "top": tk in top, "official": off, "real": real})
    d = pd.DataFrame(rows)
    print(f"buy/sell at {IN_ET} / {OUT_ET} ET · {d['date'].nunique()} days, {d['date'].min().date()} to {d['date'].max().date()}")
    out = {}
    for trade in ("day", "night"):
        x = d[d["trade"] == trade]
        per_day = x.groupby(["date", "top"])[["official", "real"]].mean().unstack("top")
        res = {}
        for label, col in (("top10", True), ("average stock", False)):
            off, real = per_day[("official", col)], per_day[("real", col)]
            gap = real - off
            res[label] = {"official": float(off.mean()), "real": float(real.mean()), "drag": float(gap.mean()),
                          "drag_t": float(gap.mean() / (gap.std() / np.sqrt(len(gap))))}
        edge_off = per_day[("official", True)] - per_day[("official", False)]
        edge_real = per_day[("real", True)] - per_day[("real", False)]
        res["edge_vs_avg"] = {"official": float(edge_off.mean()), "official_t": float(edge_off.mean() / (edge_off.std() / np.sqrt(len(edge_off)))),
                              "real": float(edge_real.mean()), "real_t": float(edge_real.mean() / (edge_real.std() / np.sqrt(len(edge_real))))}
        out[trade] = res
        pc = lambda v: f"{v * 100:+.3f}%"
        print(f"\n{trade}: top 10 official {pc(res['top10']['official'])} -> realistic {pc(res['top10']['real'])} "
              f"(drag {pc(res['top10']['drag'])}, t {res['top10']['drag_t']:+.1f}) | average stock official {pc(res['average stock']['official'])} "
              f"-> realistic {pc(res['average stock']['real'])}")
        print(f"   edge vs average stock: official {pc(res['edge_vs_avg']['official'])} (t {res['edge_vs_avg']['official_t']:+.1f}) "
              f"-> realistic {pc(res['edge_vs_avg']['real'])} (t {res['edge_vs_avg']['real_t']:+.1f})")
    out["days"] = int(d["date"].nunique())
    out["in_et"], out["out_et"] = IN_ET, OUT_ET
    (HERE / f"exec_check{'_wide' if WIDE else ''}_{IN_ET.replace(':', '')}_{OUT_ET.replace(':', '')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
