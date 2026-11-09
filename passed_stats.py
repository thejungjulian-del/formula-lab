"""Plain numbers for the signals that passed signal_search.py's locked test, for the dashboard.

Descriptive only: the pass/fail verdict is already fixed in holdout_results.json. For each passed signal it
records, on 2023–today trades: the scaling used in the test, the out-of-sample slope, and the spread of
single-trade returns per fifth of stocks (so the page can show the chance a trade makes money at ANY cost).

Run: .venv/bin/python passed_stats.py   -> passed_stats.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import signal_search as ss

HERE = Path(__file__).parent
QS = np.linspace(0, 1, 201)  # return quantiles, every 0.5%


def summary(y):
    return {"trades": int(len(y)), "avg": float(y.mean()), "worst": float(y.min()), "quantiles": [float(x) for x in np.quantile(y, QS)]}


def main():
    passed = [r for r in json.loads((HERE / "holdout_results.json").read_text())["results"] if r.get("helps")]
    p = ss.build()
    out = {"generated": str(pd.Timestamp.now().date()), "period": f"{ss.HOLDOUT} to today", "test_cost": ss.COST, "signals": {}}
    for r in passed:
        n, h, sign = r["feature"], r["h"], float(np.sign(r["search_t"]))
        ref = p.loc[p.index < ss.CONFIRM, n]
        m, s = float(ref.mean()), float(ref.std())
        ho = p.index >= ss.HOLDOUT
        d = pd.DataFrame({"z": ((p[n] - m) / s).clip(-5, 5)[ho], "y": p[f"y{h}"][ho], "d": p.index[ho]}).dropna()
        fifth = np.ceil((d["z"] * sign).groupby(d["d"]).rank(pct=True) * 5).clip(1, 5)  # 5 = strongest buy
        out["signals"][n] = {"trade": h, "desc": ss.DESC[n], "mean": m, "std": s, "sign": sign, "slope": r["holdout_b"],
                             "holdout_t": r["holdout_t"], "all": summary(d["y"]),
                             "fifths": {int(k): summary(g["y"]) for k, g in d.groupby(fifth)}}  # 5 = strongest buy
        # what the plan actually does: equal $ in today's top 10, one portfolio per day
        top10 = d.assign(k=d["z"] * sign).sort_values("k").groupby("d").tail(10).groupby("d")["y"].mean()
        out["signals"][n]["top10_daily"] = summary(top10)
    (HERE / "passed_stats.json").write_text(json.dumps(out))
    for n, v in out["signals"].items():
        print(n, v["trade"], "all avg %.4f%%" % (v["all"]["avg"] * 100),
              {k: (f["trades"], "%.4f%%" % (f["avg"] * 100)) for k, f in v["fifths"].items()},
              "top10 daily avg %.4f%%, worst %.2f%%" % (v["top10_daily"]["avg"] * 100, v["top10_daily"]["worst"] * 100))



if __name__ == "__main__":
    main()
