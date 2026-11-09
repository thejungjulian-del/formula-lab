"""Qualitative reasons: do analyst opinion changes and earnings news predict the day-trade move?

PRE-REGISTERED (written before the first run; do not add, drop or tune features after seeing results):
  "Qualitative" = human judgments and news events that have a dated history long enough to test:
    analyst upgrades/downgrades and price-target changes (Yahoo, since 2012) and earnings beats/misses
    (Yahoo, report time + surprise %). News headlines and insider trades only go back ~2 years: not testable.
  Each value counts only if it was public BEFORE the open of the trade day D (the buy). Analyst actions dated
  day d are counted at the close of the first trading day >= d (late, never early). An earnings report counts
  from the first market open after its timestamp.
  Same trade, universe, costs and out-of-sample period as daily_model.py. In-sample starts 2013 (analyst data
  begins 2012), so the comparison model (the 18 numeric features alone) is refit on 2013–2018 too.
  Step 1  Each new feature alone, both periods (t-stats clustered by date).
  Step 2  Combined = numeric features chosen by daily_model.py + new features with in-sample |t| >= 2;
          fit 2013–2018. Comparison = numeric features alone, fit 2013–2018.
  Step 3  Score both ONCE on 2019–today. The combined model is used only if it passes daily_model's rule
          AND its edge over always-buying is larger than the comparison's. Otherwise nothing changes.

Run: .venv/bin/python qualitative.py   -> prints every result and writes qualitative.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
yf.set_tz_cache_location(__import__("tempfile").mkdtemp(prefix="yf-tz-"))  # own cache per process: a shared one fails when two run at once

import daily_model as dm
from shortterm import SPLIT, UNIVERSE

START = "2013-01-01"
QFEATURES = {
    "analyst_net_1d": "Analyst upgrades minus downgrades, latest day",
    "analyst_net_20d": "Analyst upgrades minus downgrades, last 20 trading days",
    "target_net_5d": "Price-target raises minus cuts, last 5 trading days",
    "earn_new": "Earnings surprise % released since the last close (before the buy)",
    "earn_recent": "Earnings surprise % from the 10 trading days before that",
    "earn_ahead": "Earnings come out during a 2-day hold",
}


def events(ticker):
    """(analyst actions, earnings reports) from Yahoo; empty for ETFs."""
    tk = yf.Ticker(ticker)
    try:
        ud = tk.upgrades_downgrades
    except Exception:
        ud = None
    try:
        ed = tk.get_earnings_dates(limit=100)
    except Exception:
        ed = None
    return ud, ed


def q_features(idx, ud, ed):
    """Rows = closes t in idx (trading days). Every value was public before the open of the next trading day D."""
    out = pd.DataFrame(0.0, index=idx, columns=list(QFEATURES))
    ext = idx.append(pd.DatetimeIndex([idx[-1] + pd.offsets.BDay(1), idx[-1] + pd.offsets.BDay(2)]))  # + next two days
    n = len(idx)
    if ud is not None and len(ud):
        when = pd.DatetimeIndex(ud.index).tz_localize(None).normalize()
        at = ext.searchsorted(when)  # first trading day >= action date
        a = pd.DataFrame({"at": at, "net": (ud["Action"] == "up").astype(int) - (ud["Action"] == "down").astype(int),
                          "tgt": (ud["priceTargetAction"] == "Raises").astype(int) - (ud["priceTargetAction"] == "Lowers").astype(int)})
        daily = a[a["at"] < n].groupby("at")[["net", "tgt"]].sum().reindex(range(n), fill_value=0)
        out["analyst_net_1d"] = daily["net"].to_numpy()
        out["analyst_net_20d"] = daily["net"].rolling(20, min_periods=1).sum().to_numpy()
        out["target_net_5d"] = daily["tgt"].rolling(5, min_periods=1).sum().to_numpy()
    if ed is not None and len(ed):
        t_et = pd.DatetimeIndex(ed.index).tz_convert("America/New_York")
        day = t_et.tz_localize(None).normalize()
        before_open = (t_et.hour * 60 + t_et.minute) < 570
        # first trading day whose 9:30 open comes after the report
        known = np.where(before_open, ext.searchsorted(day), ext.searchsorted(day, side="right"))
        surprise = ed["Surprise(%)"].clip(-30, 30).to_numpy()
        for k, s in zip(known, surprise):
            # row i trades on D = ext[i + 1]
            if np.isfinite(s):
                if 1 <= k <= n:
                    out.iloc[k - 1, out.columns.get_loc("earn_new")] = s
                lo, hi = max(k, 0), min(k + 10, n)  # rows whose D comes 1–10 trading days after the report
                if lo < hi:
                    out.iloc[lo:hi, out.columns.get_loc("earn_recent")] = s
            if 2 <= k <= n + 1:
                out.iloc[k - 2, out.columns.get_loc("earn_ahead")] = 1.0  # report lands between D's open and D+1's open
    return out


def run():
    raw = yf.download(UNIVERSE + ["^VIX"], start="2009-01-01", auto_adjust=True, group_by="ticker", progress=False)
    vix, spy = raw["^VIX"]["Close"].dropna(), raw["SPY"]["Close"].dropna()
    base_json = json.loads(Path(__file__).with_name("daily_model.json").read_text())
    rows = []
    for t in UNIVERSE:
        df = raw[t].dropna(subset=["Open", "Close"])
        f = pd.concat([dm.features(df, spy, vix), q_features(df.index, *events(t))], axis=1)
        for hh in dm.HOLDS:
            f[f"y{hh}"] = df["Close"].shift(-(1 + hh)) / df["Open"].shift(-1) - 1
        rows.append(f.assign(ticker=t))
    data = pd.concat(rows)
    data = data[data.index >= START].dropna()
    ins, oos = data[data.index < SPLIT], data[data.index >= SPLIT]
    names = list(dm.FEATURES) + list(QFEATURES)
    mean, std = ins[names].mean(), ins[names].std().replace(0, 1)
    z = lambda d: ((d[names] - mean) / std).clip(-5, 5)
    Zi, Zo = z(ins), z(oos)
    cover = {q: {"in": float((ins[q] != 0).mean()), "out": float((oos[q] != 0).mean())} for q in QFEATURES}

    out = {"generated": str(pd.Timestamp.now().date()), "start": START, "split": SPLIT, "rows": len(data),
           "mean": mean.to_dict(), "std": std.to_dict(), "features": {**dm.FEATURES, **QFEATURES},
           "qualitative": QFEATURES, "coverage": cover, "horizons": {}}
    for hh, hname in dm.HOLDS.items():
        yi, yo = ins[f"y{hh}"].to_numpy(), oos[f"y{hh}"].to_numpy()
        single = {}
        for q in QFEATURES:  # step 1
            bi, ti = dm.ols_cl(Zi[[q]].to_numpy(), yi, ins.index)
            bo, to = dm.ols_cl(Zo[[q]].to_numpy(), yo, oos.index)
            single[q] = {"in_b": float(bi[1]), "in_t": float(ti[1]), "out_b": float(bo[1]), "out_t": float(to[1]),
                         "both": bool(abs(ti[1]) >= 2 and abs(to[1]) >= 2 and np.sign(ti[1]) == np.sign(to[1]))}
        numeric = base_json["horizons"][str(hh)]["chosen"]
        new = [q for q in QFEATURES if abs(single[q]["in_t"]) >= 2]
        comp = dm.fit_and_score(Zi, yi, ins.index, Zo, yo, oos.index, numeric)  # steps 2–3
        comb = dm.fit_and_score(Zi, yi, ins.index, Zo, yo, oos.index, numeric + new)
        use = bool(new and comb["passed"] and comb["oos"]["edge"] > comp["oos"]["edge"])
        out["horizons"][hh] = {"name": hname, "single": single, "new": new, "comparison": comp, "combined": comb, "use": use}
    return out


def show(out):
    print(f"{out['rows']} stock-days from {out['start']}")
    for q, c in out["coverage"].items():
        print(f"  {q:16s} non-zero on {c['in'] * 100:5.1f}% of in-sample rows, {c['out'] * 100:5.1f}% out-of-sample")
    for hh, r in out["horizons"].items():
        print(f"\n=== {r['name']} ===")
        for q, s in r["single"].items():
            print(f"{q:16s} in t {s['in_t']:+6.2f}  out t {s['out_t']:+6.2f}  {'BOTH' if s['both'] else ''}")
        for label, m in (("numeric only", r["comparison"]), ("numeric + qualitative", r["combined"])):
            o = m.get("oos")
            print(f"{label:22s} uses {m['chosen']}")
            if o:
                print(f"   OOS: right {o['hit'] * 100:.1f}% vs {o['hit_always_up'] * 100:.1f}% always-up, "
                      f"edge {o['edge'] * 100:+.3f}%/day (t {o['edge_t']:+.2f}), slope t {o['slope_t']:+.2f}, passed={m['passed']}")
        print("USE QUALITATIVE MODEL" if r["use"] else "keep the numeric model (qualitative did not add a tested edge)")


if __name__ == "__main__":
    out = run()
    Path(__file__).with_name("qualitative.json").write_text(json.dumps(out, indent=2))
    show(out)
