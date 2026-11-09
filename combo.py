"""Combine the promoted signals into one score per trade (the quant way: z-score each across stocks, average).

PRE-REGISTERED (written before the first run):
  Inputs    every signal in promoted.json for that trade (it passed 2010–18 AND 2019–22; frozen before 2023+ was
            scored), with the sign it had in 2010–18. Size/liquidity signals stay out (survivorship, as before).
  Score     each day: cross-sectional z-score of each input (signed), clipped at ±3, then the plain average.
            No weights are fitted.
  Trades    h0 (buy at the open, sell at the close) and ovn (buy at the close, sell at the next open).
  Use it    in the plan only if, on 2023–today, the combo's top-10 portfolio beats the single tested signal's
            top-10 portfolio on average after 0.05% costs AND the combo's top-fifth excess has t >= 2.
  Caveat    2023–today was already used once to confirm the single signals, so this is a second look at it.
            Paper-trading going forward is the clean test.

Run: .venv/bin/python combo.py   -> combo.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import signal_search as ss

HERE = Path(__file__).parent
SINGLE = {"h0": "gap_D_rel", "ovn": "vol_surprise"}
COST = 0.0005


def components(trade):
    pro = json.loads((HERE / "promoted.json").read_text())
    skip = set(pro.get("not_eligible_to_help", {}).get("features", []))
    return {x["feature"]: float(np.sign(x["search_t"])) for x in pro["promoted"]
            if x["h"] == trade and x["measure"] == "xs" and x["feature"] not in skip}


def score(p, comps):
    """Average of signed, per-day z-scores."""
    zs = []
    for f, sign in comps.items():
        g = p.groupby(level=0)[f]
        zs.append((sign * (p[f] - g.transform("mean")) / g.transform("std")).clip(-3, 3))
    return pd.concat(zs, axis=1).mean(axis=1)


def evaluate(s, y, dates):
    d = pd.DataFrame({"s": s.to_numpy(), "y": y.to_numpy(), "d": dates}).dropna()
    rk = d.groupby("d")["s"].rank(pct=True)
    ex = (d["y"] - d.groupby("d")["y"].transform("mean"))[rk >= 0.8].groupby(d["d"][rk >= 0.8]).mean()
    top10 = d.sort_values("s").groupby("d").tail(10).groupby("d")["y"].mean()
    top5 = d.sort_values("s").groupby("d").tail(5).groupby("d")["y"].mean()  # what the $100 plan buys per leg
    fm = ss.fm(d["s"], d["y"], d["d"])
    return {"slope_t": fm[1] if fm else None, "fifth_excess": float(ex.mean()), "fifth_t": float(ss.nw_t(ex.sort_index().to_numpy(), 0)),
            "top10_avg": float(top10.mean()), "top10_net": float(top10.mean() - COST), "top10_win": float((top10 > COST).mean()),
            "top10_bad_day": float(top10.quantile(0.05)), "days": int(len(top10)),
            "avg": float(top10.mean()), "quantiles": [float(x) for x in np.quantile(top10, np.linspace(0, 1, 201))],
            "top5": {"avg": float(top5.mean()), "quantiles": [float(x) for x in np.quantile(top5, np.linspace(0, 1, 201))],
                     "win": float((top5 > COST).mean()), "bad_day": float(top5.quantile(0.05))}}


def main():
    p = ss.build()
    out = {"generated": str(pd.Timestamp.now().date()), "cost": COST, "trades": {}}
    for trade, single in SINGLE.items():
        comps = components(trade)
        combo = score(p, comps)
        res = {"components": comps}
        for name, mask in (("2019-22", (p.index >= ss.CONFIRM) & (p.index < ss.HOLDOUT)), ("2023-now", p.index >= ss.HOLDOUT)):
            y = p.loc[mask, f"y{trade}"]
            res[name] = {"combo": evaluate(combo[mask], y, p.index[mask]),
                         "single": evaluate(p.loc[mask, single] * np.sign(json.loads((HERE / "passed_stats.json").read_text())["signals"][single]["sign"]), y, p.index[mask])}
        # descriptive: single-stock trades by fifth of the combo ranking, 2023–today (5 = top)
        m = p.index >= ss.HOLDOUT
        d = pd.DataFrame({"s": combo[m].to_numpy(), "y": p.loc[m, f"y{trade}"].to_numpy(), "d": p.index[m]}).dropna()
        fifth = np.ceil(d.groupby("d")["s"].rank(pct=True) * 5).clip(1, 5)
        res["fifths_2023"] = {int(k): {"avg": float(g["y"].mean()), "win": float((g["y"] > COST).mean()), "trades": int(len(g))} for k, g in d.groupby(fifth)}
        res["all_2023"] = {"avg": float(d["y"].mean()), "win": float((d["y"] > COST).mean())}
        h = res["2023-now"]
        res["use_combo"] = bool(h["combo"]["top10_net"] > h["single"]["top10_net"] and h["combo"]["fifth_t"] >= 2)
        out["trades"][trade] = res
    (HERE / "combo.json").write_text(json.dumps(out, indent=1))
    for trade, r in out["trades"].items():
        print(f"\n=== {trade}: combo of {list(r['components'])} ===")
        for per in ("2019-22", "2023-now"):
            for k in ("single", "combo"):
                e = r[per][k]
                print(f"{per} {k:6s} top10 avg {e['top10_avg'] * 100:+.3f}%/trade, after 0.05% {e['top10_net'] * 100:+.3f}%, win {e['top10_win'] * 100:.1f}%, "
                      f"bad day {e['top10_bad_day'] * 100:+.2f}% | top-fifth excess {e['fifth_excess'] * 100:+.3f}% (t {e['fifth_t']:+.2f})")
        print("USE COMBO" if r["use_combo"] else "keep the single signal")


if __name__ == "__main__":
    main()
