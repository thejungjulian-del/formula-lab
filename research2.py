"""Ways to make the daily plan earn more / win more often. Each idea vs the current plan, on two periods.

PRE-REGISTERED (written 2026-10-17 before any of these were computed):
  Current plan (baseline)  day leg = top 5 by the day combo, buy open / sell close; night leg = top 5 by the night
                           combo, buy close / sell next open; 0.05% per round trip.
  Ideas
    D1  day:   at most 1 pick per sector
    D2  day:   add "gap vs the stock's sector" (gap minus the sector's average gap, sign −) to the day combo
    D3  day:   only stocks with NO news overnight (no 8-K filed, no earnings released between t's close and D's open)
    D4  day:   trade only when the VIX closed above its own 1-year median the day before (else sit in cash)
    D5  day:   only the 250 most-traded stocks that day (by 20-day $ volume)
    N1  night: at most 1 pick per sector
    N2  night: hold QQQ overnight instead of 5 stocks (QQQ round trip cost 0.004%)
    N3  night: hold QQQ overnight only when QQQ closed above its 200-day average, else cash
    N6  night: only the 250 most-traded stocks
    N4  cost:  a night pick that is also the next morning's day pick is kept instead of sold and re-bought
               (measured: how often, and the cost saved)
  Periods   DEV 2010–2022 (the plan was built here) · FRESH 2000–2009 (never used for the daily signals) ·
            2023–today is shown for information only (already used several times).
  Measure   per day: idea's leg return minus the current leg's return, after costs (days an idea sits out = 0).
  Passes    improvement > 0 with t >= 2 in DEV AND t >= 2.58 in FRESH (Bonferroni for ~10 ideas), same sign.
  Caveat    FRESH uses today's S&P 500 members, so survivorship bias is stronger there; ideas are judged against the
            current plan on the same days, where most of that bias cancels.

Run: .venv/bin/python research2.py   -> research2.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import combo
import qualitative
import search_data as sd
import signal_search as ss

HERE = Path(__file__).parent
COST, QQQ_COST, TOP = 0.0005, 0.00004, 5
PERIODS = {"DEV 2010-22": ("2010-01-01", "2023-01-01"), "FRESH 2000-09": ("2000-01-01", "2010-01-01"), "2023-now (info)": ("2023-01-01", "2030-01-01")}


def build(prices, spy, vix, sec, events, start, end):
    cal = spy.index[(spy.index >= pd.Timestamp(start) - pd.Timedelta(days=420)) & (spy.index < end)]
    gap = spy["Open"].shift(-1) / spy["Close"] - 1
    U = sd.load("universe")
    sector = dict(zip(U["ticker"], U["sector"]))
    parts = []
    for t, df in prices.items():
        df = df[df.index.isin(cal)]
        if len(df) < 300 or t not in sector:
            continue
        f = ss.stock_features(df, gap, cal)
        c, v = df["Close"].astype(float), df["Volume"].astype(float)
        f["adv"] = (c * v).rolling(20).mean()
        f["raw_gap"] = df["Open"].shift(-1) / c - 1
        f["sd"] = c.pct_change().rolling(20).std()
        s8 = ss.sec_features(sec.get(t), cal).reindex(f.index)["sec_8k_overnight"] if sec.get(t) is not None else 0.0
        q = qualitative.q_features(f.index, *events.get(t, (None, None)))
        f["news"] = ((s8 > 0) | (q["earn_new"] != 0)).astype(float)
        f["ticker"], f["sector"] = t, sector[t]
        parts.append(f)
    p = pd.concat(parts)
    p = p[(p.index >= start) & (p.index < end)]
    g = p.groupby([p.index, p["sector"]])["raw_gap"]
    p["gap_vs_sector"] = (p["raw_gap"] - (g.transform("sum") - p["raw_gap"]) / (g.transform("count") - 1).clip(lower=1)) / p["sd"]
    p["adv_rank"] = p.groupby(level=0)["adv"].rank(ascending=False)
    vx = vix["Close"].astype(float)
    p["vix_high"] = (vx > vx.rolling(252, min_periods=120).median()).reindex(p.index).to_numpy()
    return p


def top_legs(p, score, y, top=TOP, cap=None):
    """Each day: the `top` highest scores (NaN scores are left out), at most `cap` per sector; return − cost."""
    d = pd.DataFrame({"s": score, "y": p[y], "sector": p["sector"], "t": p["ticker"]}).dropna(subset=["s", "y"])
    d = d.assign(dt=d.index).sort_values(["dt", "s"], ascending=[True, False])
    if cap:
        d = d[d.groupby(["dt", "sector"]).cumcount() < cap]
    picks = d.groupby("dt").head(top)
    return picks.groupby("dt")["y"].mean() - COST, picks


def compare(a, b):
    """Per-day improvement a − b (missing days = 0 for the idea: it sat out)."""
    x = (a.reindex(b.index).fillna(0) - b).dropna()
    return {"improve": float(x.mean()), "t": float(x.mean() / (x.std() / np.sqrt(len(x)))) if x.std() > 0 else 0.0, "days": int(len(x))}


def qqq_night(qqq, idx, filt=False):
    o, c = qqq["Open"].astype(float), qqq["Close"].astype(float)
    r = (o.shift(-1) / c - 1 - QQQ_COST)
    if filt:
        r = r.where(c > c.rolling(200).mean(), 0.0)
    return r.reindex(idx)


def evaluate(p, qqq):
    C = json.loads((HERE / "combo.json").read_text())["trades"]
    day = combo.score(p, C["h0"]["components"])
    night = combo.score(p, C["ovn"]["components"])
    base_d, bd_picks = top_legs(p, day, "yh0")
    base_n, bn_picks = top_legs(p, night, "yovn")
    out = {"baseline_day": float(base_d.mean()), "baseline_night": float(base_n.mean()), "days": int(len(base_d))}
    z = lambda col: ((p[col] - p.groupby(level=0)[col].transform("mean")) / p.groupby(level=0)[col].transform("std")).clip(-5, 5)
    ideas = {
        "D1 sector cap": top_legs(p, day, "yh0", cap=1)[0],
        "D2 gap vs sector": top_legs(p, (day * len(C["h0"]["components"]) - z("gap_vs_sector")) / (len(C["h0"]["components"]) + 1), "yh0")[0],
        "D3 no-news only": top_legs(p, day.where(p["news"] == 0), "yh0")[0],
        "D4 high-VIX days only": base_d.where(pd.Series(p["vix_high"].to_numpy(), index=p.index).groupby(level=0).first().reindex(base_d.index) == 1),
        "D5 top-250 liquid": top_legs(p, day.where(p["adv_rank"] <= 250), "yh0")[0],
        "N1 sector cap": top_legs(p, night, "yovn", cap=1)[0],
        "N2 QQQ overnight": qqq_night(qqq, base_n.index),
        "N3 QQQ overnight >200d": qqq_night(qqq, base_n.index, filt=True),
        "N6 top-250 liquid": top_legs(p, night.where(p["adv_rank"] <= 250), "yovn")[0],
    }
    base = {k: (base_d if k.startswith("D") else base_n) for k in ideas}
    out["ideas"] = {k: {**compare(v, base[k]), "level": float(v.mean())} for k, v in ideas.items()}
    # N4: night picks at close t that are also the day picks for D (same row t): keep instead of sell + re-buy
    both = pd.merge(bn_picks.reset_index()[["dt", "t"]], bd_picks.reset_index()[["dt", "t"]], on=["dt", "t"])
    out["N4 overlap"] = {"share_of_day_picks": float(len(both) / len(bd_picks)), "cost_saved_per_day": float(len(both) / len(bd_picks) * COST)}
    return out


def main():
    res = {}
    events = sd.load("events")
    secL = sd.load("sec_long")
    for name, (a, b) in PERIODS.items():
        if name.startswith("FRESH"):
            L = sd.load("prices_long")
            prices = {t: df for t, df in L.items() if t not in ("SPY", "QQQ", "^VIX")}
            spy, vix, qqq = L["SPY"], L["^VIX"], L["QQQ"]
        else:
            prices, M = sd.load("prices"), sd.load("macro")
            spy, vix = M["spy"], M["vix"]
            import yfinance as yf
            qqq = yf.download("QQQ", start="2008-01-01", auto_adjust=True, progress=False)
            qqq.columns = [c[0] if isinstance(c, tuple) else c for c in qqq.columns]
        p = build(prices, spy, vix, secL, events, a, b)
        res[name] = evaluate(p, qqq)
        print(f"\n=== {name}: {res[name]['days']} days · current day leg {res[name]['baseline_day'] * 100:+.3f}%/day, night leg {res[name]['baseline_night'] * 100:+.3f}%/day")
        for k, v in res[name]["ideas"].items():
            print(f"  {k:24s} improvement {v['improve'] * 100:+.4f}%/day (t {v['t']:+.2f})  -> leg {v['level'] * 100:+.3f}%/day")
        o = res[name]["N4 overlap"]
        print(f"  N4 overlap: {o['share_of_day_picks'] * 100:.1f}% of day picks were already held overnight -> saves {o['cost_saved_per_day'] * 100:.4f}%/day")
    verdict = {}
    for k in res["DEV 2010-22"]["ideas"]:
        d, f = res["DEV 2010-22"]["ideas"][k], res["FRESH 2000-09"]["ideas"][k]
        verdict[k] = bool(d["improve"] > 0 and d["t"] >= 2 and f["improve"] > 0 and f["t"] >= 2.58)
    res["passes"] = verdict
    (HERE / "research2.json").write_text(json.dumps(res, indent=1))
    print("\nPASSES (DEV t>=2 and FRESH t>=2.58):", [k for k, v in verdict.items() if v] or "none")


def plan_stats():
    """Measured results of the plan as it now trades (day leg = no-news top 5; night leg = top 5), 2023–today,
    for the Plan page's numbers. Official open/close prices; real-fill losses are applied on the page."""
    prices, M = sd.load("prices"), sd.load("macro")
    p = build(prices, M["spy"], M["vix"], sd.load("sec_long"), sd.load("events"), "2023-01-01", "2030-01-01")
    C = json.loads((HERE / "combo.json").read_text())["trades"]
    q = lambda x: {"avg": float(x.mean()), "quantiles": [float(v) for v in np.quantile(x, np.linspace(0, 1, 201))], "days": int(len(x))}
    day = top_legs(p, combo.score(p, C["h0"]["components"]).where(p["news"] == 0), "yh0")[0] + COST   # gross (costs applied on the page)
    night = top_legs(p, combo.score(p, C["ovn"]["components"]), "yovn", top=10)[0] + COST      # the plan: 10 overnight picks
    out = {"generated": str(pd.Timestamp.now().date()), "period": "2023-now", "day": q(day), "night": q(night)}
    # the $100 account as it trades: half the money per leg, real-fill losses (exec check), 0.05% per trade
    import confidence as cf
    ex = json.loads((HERE / "exec7.json").read_text())                 # 7 years of real fills at the broker's times
    kept = {"day": ex["day"]["0935"]["kept"], "night": ex["night"]["0931"]["kept"]}
    acct = (night - (1 - kept["night"]) * night.mean() - COST).dropna().to_numpy()   # day leg paused: whole $100 overnight
    rng, n = np.random.default_rng(7), len(acct)
    boot = acct[cf.block_bootstrap_idx(n, 10000, rng)]
    six = 100 * np.prod(1 + acct[cf.block_bootstrap_idx(n, 10000, rng)[:, :126]], axis=1)
    yr = 100 * np.prod(1 + acct[cf.block_bootstrap_idx(n, 10000, rng)[:, :252]], axis=1)
    out["account"] = {"kept": kept, "win_days": float((acct > 0).mean()), "per_day_usd": float(acct.mean() * 100),
                      "per_year": float(np.prod(1 + acct) ** (252 / n) - 1), "p_mean_above_0": float((boot.mean(1) > 0).mean()),
                      "six_months_5_50_95": [float(x) for x in np.quantile(six, [.05, .5, .95])], "six_months_p_loss": float((six < 100).mean()),
                      "year_p_ahead": float((yr > 100).mean())}
    (HERE / "plan_stats.json").write_text(json.dumps(out))
    print("day (no-news, paused) top5 avg %.3f%%, night top10 avg %.3f%% per trade, before costs" % (out["day"]["avg"] * 100, out["night"]["avg"] * 100))
    print("account:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["account"].items()})


if __name__ == "__main__":
    import sys
    plan_stats() if sys.argv[1:] == ["stats"] else main()
