"""Do the frozen combined scores work on stocks they were never built on? (out-of-universe check)

PRE-REGISTERED (written before the first run):
  Universe  liquid US common stocks (search_data.wide): on each day t a stock counts only if its close was > $5
            and its 20-day average dollar volume was >= $20M ON THAT DAY (point in time). Delisted names are missing
            (survivorship), so results are judged against the other stocks on the same day.
  Scores    exactly combo.json's components and signs. No new selection, no refitting.
  Report    2019–22 and 2023–today, for (a) every liquid stock and (b) only stocks that are NOT in today's
            S&P 500 (the part of the universe the search never saw).
  Verdict   "generalizes" for a trade if, on 2023–today, group (b) has top-fifth excess > 0 with t >= 2.

Run: .venv/bin/python wide_check.py   -> wide_check.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import combo
import search_data as sd
import signal_search as ss

HERE = Path(__file__).parent
MIN_PX, MIN_DV = 5.0, 20e6


def panel():
    prices, macro = sd.load("prices_wide"), sd.load("macro")
    cal = macro["spy"].index[macro["spy"].index >= "2009-01-01"]
    spy_gap = macro["spy"]["Open"].shift(-1) / macro["spy"]["Close"] - 1
    comps = set()
    C = json.loads((HERE / "combo.json").read_text())["trades"]
    for t in C.values():
        comps |= set(t["components"])
    keep = sorted(comps) + ["yh0", "yovn"]
    parts = []
    for tk, df in prices.items():
        df = df[df.index.isin(cal)]
        if len(df) < 60:
            continue
        f = ss.stock_features(df, spy_gap, cal)
        c, v = df["Close"].astype("float64"), df["Volume"].astype("float64")
        liquid = (c > MIN_PX) & ((c * v).rolling(20).mean() >= MIN_DV)
        f = f.loc[liquid, keep].astype("float32")
        f["ticker"] = tk
        parts.append(f)
    p = pd.concat(parts)
    return p[p.index >= ss.SEARCH], C


def main():
    p, C = panel()
    sp = set(sd.load("universe")["ticker"])
    out = {"stocks": int(p["ticker"].nunique()), "per_day_median": float(p.groupby(level=0).size().median()), "trades": {}}
    for trade, y in (("h0", "yh0"), ("ovn", "yovn")):
        s = combo.score(p, C[trade]["components"])
        res = {}
        for per, m in (("2019-22", (p.index >= ss.CONFIRM) & (p.index < ss.HOLDOUT)), ("2023-now", p.index >= ss.HOLDOUT)):
            for grp, g in (("all liquid", np.ones(len(p), bool)), ("not in S&P 500", ~p["ticker"].isin(sp).to_numpy())):
                mm = m & g
                # rank within the group on each day, so (b) is a fair test on its own
                sub = p[mm]
                ssub = combo.score(sub, C[trade]["components"]) if grp != "all liquid" else s[mm]
                res[f"{per} | {grp}"] = combo.evaluate(ssub, sub[y], sub.index)
        h = res["2023-now | not in S&P 500"]
        res["generalizes"] = bool(h["fifth_excess"] > 0 and h["fifth_t"] >= 2)
        out["trades"][trade] = res
    (HERE / "wide_check.json").write_text(json.dumps(out, indent=1))
    print(f"{out['stocks']} liquid stocks ever, median {out['per_day_median']:.0f} per day")
    for trade, r in out["trades"].items():
        print(f"\n=== {trade} ===")
        for k, e in r.items():
            if k == "generalizes":
                continue
            print(f"{k:32s} top10 {e['top10_avg'] * 100:+.3f}%/trade (after 0.05%: {e['top10_net'] * 100:+.3f}%, win {e['top10_win'] * 100:.1f}%) | "
                  f"top-fifth excess {e['fifth_excess'] * 100:+.3f}% (t {e['fifth_t']:+.2f}) | {e['days']} days")
        print("GENERALIZES to stocks outside the S&P 500" if r["generalizes"] else "does NOT generalize")


if __name__ == "__main__":
    main()
