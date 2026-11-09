"""Day-trading model: which numbers known at today's close actually predict tomorrow's move?

PRE-REGISTERED (written before the first run; do not add, drop or tune features after seeing results):
  Trade: features known at the CLOSE of day t -> buy (or short) at the OPEN of D = t+1 -> sell at the close of
         D (h=0, "same day") or D+1 (h=1, "next day"). Dashboard "1 day" uses h=0, "2 days" uses h=1.
  Universe, costs and periods: same as shortterm.py (35 liquid names, 5 bps round trip, in-sample 2010–2018,
         out-of-sample 2019–today). Every feature is scaled using in-sample numbers only.
  Step 1  Each feature alone: pooled regression of the trade return on the feature, t-stats clustered by date
          (all stocks move together on a given day, so a day counts once). Reported for both periods.
  Step 2  Combined model = one regression on every feature with in-sample |t| >= 2, fitted on 2010–2018 only.
  Step 3  Score it ONCE on 2019–today. PASSES if (a) predicted vs actual slope > 0 with t >= 2, and
          (b) trading its direction (long if forecast > cost, short if < −cost, else no trade) beats always
          buying with t >= 2 and makes money after costs.
  The dashboard adds the model's forecast only for a horizon whose model PASSES.

Run: .venv/bin/python daily_model.py   -> prints every result and writes daily_model.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
yf.set_tz_cache_location(__import__("tempfile").mkdtemp(prefix="yf-tz-"))  # own cache per process: a shared one fails when two run at once

from shortterm import COST, SPLIT, UNIVERSE, rsi

HOLDS = {0: "same day", 1: "next day"}
FEATURES = {
    "ret_1d": "Today's move, in units of its normal daily move",
    "ret_5d": "Last 5 days' move, in units of its normal move",
    "mom_12_1": "12-month trend, skipping the last month",
    "dist_ma20": "Distance above its 20-day average",
    "dist_ma200": "Distance above its 200-day average",
    "rsi2": "RSI(2): overbought (+) or oversold (−)",
    "close_loc": "Where it closed in today's range (top + / bottom −)",
    "overnight": "Today's gap at the open",
    "intraday": "Today's move from open to close",
    "vol_surprise": "Today's volume vs its 20-day average",
    "volatility": "How jumpy it has been (20-day volatility)",
    "near_high": "Distance below its 52-week high",
    "spy_1d": "The S&P 500's move today",
    "vix_jump": "VIX (fear index) vs its 20-day average",
    "vix_level": "VIX level",
    "rel_spy_5d": "Last 5 days vs the S&P 500",
    "monday": "Trade day is a Monday",
    "turn_of_month": "Trade day is at the turn of the month",
}


def features(df, spy, vix):
    """One row per close t. Only uses data up to t (plus the calendar date of the next trading day)."""
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    idx = c.index
    spy, vix = spy.reindex(idx).ffill(), vix.reindex(idx).ffill()
    r, sr = c.pct_change(), spy.pct_change()
    sd = r.rolling(20).std()
    # Entry day D = next trading day; for the last row it is the next weekday (ponytail: holidays ignored).
    entry = pd.DatetimeIndex(list(idx[1:]) + [idx[-1] + pd.offsets.BDay(1)])
    day = entry.values.astype("datetime64[D]")
    pos = np.busday_count(entry.values.astype("datetime64[M]").astype("datetime64[D]"), day)
    last = (entry + pd.offsets.BDay(1)).month != entry.month
    return pd.DataFrame({
        "ret_1d": r / sd,
        "ret_5d": c.pct_change(5) / (sd * np.sqrt(5)),
        "mom_12_1": (c.shift(21) / c.shift(252) - 1) / (r.rolling(252).std() * np.sqrt(231)),
        "dist_ma20": (c / c.rolling(20).mean() - 1) / sd,
        "dist_ma200": c / c.rolling(200).mean() - 1,
        "rsi2": (rsi(c) - 50) / 50,
        "close_loc": (c - l) / (h - l) - 0.5,
        "overnight": (o / c.shift() - 1) / sd,
        "intraday": (c / o - 1) / sd,
        "vol_surprise": np.log(v / v.rolling(20).mean()),
        "volatility": np.log(sd),
        "near_high": c / c.rolling(252).max() - 1,
        "spy_1d": sr / sr.rolling(20).std(),
        "vix_jump": vix / vix.rolling(20).mean() - 1,
        "vix_level": np.log(vix),
        "rel_spy_5d": (c.pct_change(5) - spy.pct_change(5)) / (sd * np.sqrt(5)),
        "monday": (entry.dayofweek == 0).astype(float),
        "turn_of_month": ((pos < 3) | last).astype(float),
    }, index=idx).replace([np.inf, -np.inf], np.nan)


def ols_cl(X, y, dates):
    """OLS with an intercept; t-stats clustered by date. Returns (coefs, t-stats), intercept first."""
    X = np.column_stack([np.ones(len(y)), X])
    inv = np.linalg.inv(X.T @ X)
    b = inv @ X.T @ y
    g = pd.DataFrame(X * (y - X @ b)[:, None]).groupby(np.asarray(dates)).sum().to_numpy()
    return b, b / np.sqrt(np.diag(inv @ (g.T @ g) @ inv))


def per_date_t(x, dates):
    d = pd.Series(x).groupby(np.asarray(dates)).mean()
    return float(d.mean()), float(d.mean() / (d.std() / np.sqrt(len(d))))


def run():
    raw = yf.download(UNIVERSE + ["^VIX"], start="2009-01-01", auto_adjust=True, group_by="ticker", progress=False)
    vix, spy = raw["^VIX"]["Close"].dropna(), raw["SPY"]["Close"].dropna()
    rows = []
    for t in UNIVERSE:
        df = raw[t].dropna(subset=["Open", "Close"])
        f = features(df, spy, vix)
        for hh in HOLDS:
            f[f"y{hh}"] = df["Close"].shift(-(1 + hh)) / df["Open"].shift(-1) - 1
        rows.append(f.assign(ticker=t))
    data = pd.concat(rows)
    data = data[data.index >= "2010-01-01"].dropna()
    ins, oos = data[data.index < SPLIT], data[data.index >= SPLIT]
    names = list(FEATURES)
    mean, std = ins[names].mean(), ins[names].std()
    z = lambda d: ((d[names] - mean) / std).clip(-5, 5)  # scaled with in-sample numbers only; outliers capped
    Zi, Zo = z(ins), z(oos)

    out = {"generated": str(pd.Timestamp.now().date()), "cost": COST, "split": SPLIT, "rows": len(data),
           "days_in": int(ins.index.nunique()), "days_out": int(oos.index.nunique()),
           "mean": mean.to_dict(), "std": std.to_dict(), "features": FEATURES, "horizons": {}}
    for hh, hname in HOLDS.items():
        yi, yo = ins[f"y{hh}"].to_numpy(), oos[f"y{hh}"].to_numpy()
        single = {}
        for n in names:  # step 1
            bi, ti = ols_cl(Zi[[n]].to_numpy(), yi, ins.index)
            bo, to = ols_cl(Zo[[n]].to_numpy(), yo, oos.index)
            single[n] = {"in_b": float(bi[1]), "in_t": float(ti[1]), "out_b": float(bo[1]), "out_t": float(to[1]),
                         "both": bool(abs(ti[1]) >= 2 and abs(to[1]) >= 2 and np.sign(ti[1]) == np.sign(to[1]))}
        chosen = [n for n in names if abs(single[n]["in_t"]) >= 2]  # step 2
        out["horizons"][hh] = {"name": hname, "single": single, **fit_and_score(Zi, yi, ins.index, Zo, yo, oos.index, chosen)}
    return out


def fit_and_score(Zi, yi, di, Zo, yo, do, chosen):
    """Steps 2–3: fit on in-sample rows, then score once on out-of-sample rows."""
    res = {"chosen": chosen, "weights": {}, "passed": False}
    if not chosen:
        return res
    b, tb = ols_cl(Zi[chosen].to_numpy(), yi, di)
    res["intercept"] = float(b[0])
    res["weights"] = {n: {"w": float(w), "t": float(tt)} for n, w, tt in zip(chosen, b[1:], tb[1:])}
    pred = b[0] + Zo[chosen].to_numpy() @ b[1:]
    sb, st = ols_cl(pred[:, None], yo, do)
    pos = np.where(pred > COST, 1, np.where(pred < -COST, -1, 0))
    strat = np.where(pos != 0, pos * yo - COST, 0.0)
    base = yo - COST
    avg, t_avg = per_date_t(strat, do)
    edge, t_edge = per_date_t(strat - base, do)
    traded = pos != 0
    res["oos"] = {"slope": float(sb[1]), "slope_t": float(st[1]), "corr": float(np.corrcoef(pred, yo)[0, 1]),
                  "hit": float((np.sign(pred[traded]) == np.sign(yo[traded])).mean()) if traded.any() else None,
                  "hit_always_up": float((yo > 0).mean()), "traded": float(traded.mean()),
                  "short_share": float((pos == -1).sum() / max(traded.sum(), 1)),
                  "avg": avg, "avg_t": t_avg, "base_avg": float(per_date_t(base, do)[0]),
                  "edge": edge, "edge_t": t_edge}
    o = res["oos"]
    res["passed"] = bool(o["slope"] > 0 and o["slope_t"] >= 2 and o["edge_t"] >= 2 and o["avg"] > 0)
    return res


def show(out):
    print(f"{out['rows']} stock-days, {out['days_in']} in-sample days, {out['days_out']} out-of-sample days")
    for hh, r in out["horizons"].items():
        print(f"\n=== {r['name']} (buy at the open, sell at the close {'that day' if hh == 0 else 'the next day'}) ===")
        print(f"{'feature':15s} {'in t':>7s} {'out t':>7s}  both")
        for n, s in r["single"].items():
            print(f"{n:15s} {s['in_t']:+7.2f} {s['out_t']:+7.2f}  {'YES' if s['both'] else ''}")
        print("combined model uses:", r["chosen"])
        if "oos" in r:
            o = r["oos"]
            print(f"OUT-OF-SAMPLE: slope {o['slope']:+.2f} (t {o['slope_t']:+.2f}), corr {o['corr']:+.4f}, "
                  f"direction right {o['hit'] * 100:.1f}% of trades vs {o['hit_always_up'] * 100:.1f}% always-up, "
                  f"trades {o['traded'] * 100:.0f}% of days ({o['short_share'] * 100:.0f}% shorts)")
            print(f"  strategy avg {o['avg'] * 100:+.3f}%/trade-day (t {o['avg_t']:+.2f}) vs always buy {o['base_avg'] * 100:+.3f}% "
                  f"-> edge {o['edge'] * 100:+.3f}% (t {o['edge_t']:+.2f})")
        print("PASSED" if r["passed"] else "did not pass")


if __name__ == "__main__":
    out = run()
    Path(__file__).with_name("daily_model.json").write_text(json.dumps(out, indent=2))
    show(out)
