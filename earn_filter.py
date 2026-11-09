"""E1: should the night leg skip stocks that report earnings while we hold them (after today's close, before the next open)?

Why: the live sim (2026-10-19 → 10-30, 70 night trades) lost most of its money on 5 trades held through earnings
(GOOGL -12%, HOOD -5.7%); those 5 averaged -2.6% vs -0.16% for the other 65.

PRE-REGISTERED (written before the first run):
  A  today's plan: night combo score, top 10 per day, buy official close, sell next official open
  B  same, but stocks with an earnings report between the close and the next open are removed first (next-best fill in)
  Adopt B if  B's average night (top-10 portfolio, after 0.05% costs) >= A's in at least 2 of 3 periods
              (2010–18, 2019–22, 2023+) AND B's worst-5% night is no worse than A's overall.
  The earnings dates are scheduled weeks ahead, so B uses nothing we couldn't know at 15:59.

Run: .venv/bin/python earn_filter.py   -> earn_filter.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import combo
import search_data
import signal_search as ss

HERE = Path(__file__).parent
COST = 0.0005
PERIODS = {"2010-18": ("2010-01-01", "2019-01-01"), "2019-22": ("2019-01-01", "2023-01-01"), "2023+": ("2023-01-01", "2100-01-01")}


def earnings_tonight(p):
    """True where the stock's earnings land after close t and at/before the next session's 9:30 open."""
    ev = search_data.load("events")
    cal = p.index.unique().sort_values()
    nxt = pd.Series(cal[1:], index=cal[:-1])
    flag = pd.Series(False, index=pd.MultiIndex.from_arrays([p.index, p["ticker"]]))
    keys = set()
    for t, (_, ed) in ev.items():
        if ed is None or not len(ed):
            continue
        when = pd.DatetimeIndex(ed.index).tz_convert("America/New_York")
        mins = when.hour * 60 + when.minute
        day = when.tz_localize(None).normalize()
        for d, m in zip(day, mins):
            if m >= 16 * 60 - 1:  # after the close on d -> held from d's close
                keys.add((d, t))
            elif m <= 9 * 60 + 30:  # before the open on d -> held from the previous session's close
                i = cal.searchsorted(d)
                if i > 0:
                    keys.add((cal[i - 1], t))
            # during market hours: already in the price at 15:59, not an overnight surprise
    flag[:] = [(d, t) in keys for d, t in flag.index]
    return flag.to_numpy()


def nightly(d):
    top = d.sort_values("s").groupby("d").tail(10)
    return top.groupby("d")["y"].mean() - COST, top["y"] - COST


def stats(night, trade):
    w, l = trade[trade > 0], trade[trade <= 0]
    return {"nights": int(len(night)), "avg_night": float(night.mean()), "night_win": float((night > 0).mean()),
            "worst5_night": float(night.quantile(0.05)), "trade_win": float((trade > 0).mean()),
            "avg_win": float(w.mean()), "avg_loss": float(l.mean()), "worst_trade": float(trade.min()),
            "worst1pct_trade": float(trade.quantile(0.01))}


def main():
    p = ss.build()
    C = json.loads((HERE / "combo.json").read_text())["trades"]["ovn"]
    s = combo.score(p, C["components"])
    d = pd.DataFrame({"s": s.to_numpy(), "y": p["yovn"].to_numpy(), "d": p.index, "earn": earnings_tonight(p)}).dropna(subset=["s", "y"])
    out = {}
    for name, (a, b) in PERIODS.items():
        x = d[(d["d"] >= a) & (d["d"] < b)]
        A = stats(*nightly(x))
        B = stats(*nightly(x[~x["earn"]]))
        top = x.sort_values("s").groupby("d").tail(10)
        out[name] = {"A": A, "B": B, "earn_share_of_A_picks": float(top["earn"].mean()),
                     "earn_picks_avg": float(top.loc[top["earn"], "y"].mean() - COST)}
    allA, allB = stats(*nightly(d)), stats(*nightly(d[~d["earn"]]))
    wins = sum(out[k]["B"]["avg_night"] >= out[k]["A"]["avg_night"] for k in PERIODS)
    out["all"] = {"A": allA, "B": allB}
    out["verdict"] = {"periods_B_wins": int(wins), "tail_ok": allB["worst5_night"] >= allA["worst5_night"],
                      "adopt": bool(wins >= 2 and allB["worst5_night"] >= allA["worst5_night"])}
    (HERE / "earn_filter.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if k == "verdict":
            print(k, v)
            continue
        print(f"== {k}" + (f"  (earnings picks = {v['earn_share_of_A_picks']:.1%} of A, avg {v['earn_picks_avg']:+.3%})" if "earn_share_of_A_picks" in v else ""))
        for lab in ("A", "B"):
            z = v[lab]
            print(f"  {lab}: avg night {z['avg_night']:+.4%}  night win {z['night_win']:.0%}  worst5% night {z['worst5_night']:+.2%} | "
                  f"trade win {z['trade_win']:.0%}  avg win {z['avg_win']:+.2%}  avg loss {z['avg_loss']:+.2%}  worst1% {z['worst1pct_trade']:+.1%}")


if __name__ == "__main__":
    main()
