"""Minute-market test, done properly: years of Alpaca 1-minute bars, real measured spreads, locked holdout.

PRE-REGISTERED (revised 2026-10-17 BEFORE any Alpaca bar was looked at; the download was still running):
  Universe  SPY, QQQ and the 50 most-traded S&P 500 stocks (cheapest to trade; alpaca_data.universe()).
  Periods   SEARCH 2019–2021 · CONFIRM 2022–2023 · HOLDOUT 2024-01-01 to 2026-07-23 (locked; scored once).
            2026-07-24 onward is EXCLUDED: intraday_test.py already looked at it with Yahoo bars.
  Trade     buy at the end of a 1-minute bar T, sell h minutes later the same day. Entries every 5 minutes from
            9:45 ET (until 16:00 − h). h = 1, 5, 10, 15, 30, 60.
  Cost      the stock's measured bid/ask spread (median of alpaca_spreads.pkl samples) per round trip: market orders
            in and out, no commission. Nothing cheaper is assumed.
  Signals   cross-sectional, known at T, scaled by the stock's 1-minute volatility over the previous 20 days:
              rev_1, rev_5, rev_15, mom_30   moves over the last 1/5/15/30 minutes
              rel_5, rel_15                  the same minus SPY's move (the part the market doesn't explain)
              day_ret, vwap_dev, range_pos   since the open / vs today's VWAP / place in today's range
              bar_pos                        where the last 1-minute bar closed within its high–low
              vol_spike                      last 5 minutes' volume vs the same 5 minutes over 20 days
            combo = average signed per-time z of every signal with |SEARCH t| >= 2 (signs from SEARCH)
            model = least-squares fit of the per-time-demeaned return on all z-signals, fitted on SEARCH only
            extreme = combo but only stocks with combo z >= 2
  Market    SPY intraday momentum (Gao, Han, Li & Zhou 2018): sign of 9:30–10:00 -> SPY 15:30–16:00. Daily.
  Promote   |t| >= 3 in SEARCH and same sign |t| >= 2 in CONFIRM (Newey-West lags = max(1, h/5)).
  Best      among promoted, the (signal, h) with the highest CONFIRM net profit per trade after spreads.
  Holdout   promoted items scored once. "Works" = same sign with t >= Bonferroni level AND the top-5 trades make
            money after measured spreads with t >= 2.
  Added 2026-10-17 after CONFIRM, BEFORE the holdout was opened (cost-minimizing variant, Julian asked to minimize
  costs): "cheap_h60" = the SEARCH-fitted model, only stocks with measured spread <= 0.01% (SPY, QQQ, AAPL, MSFT),
  buy the single best each entry time, hold 60 min. Chosen on CONFIRM (+0.0015%/trade net), so it is counted in the
  Bonferroni level and must make money after spreads with t >= 2 on the holdout.

Run: .venv/bin/python intraday_v2.py            (search + confirm -> intraday_v2_ledger.json, frozen list)
     .venv/bin/python intraday_v2.py holdout    (once -> intraday_v2_holdout.json)
"""
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import alpaca_data
import signal_search as ss

HERE = Path(__file__).parent
ET = "America/New_York"
SEARCH_END, CONFIRM_END, HOLD_END = (pd.Timestamp(x, tz=ET) for x in ("2022-01-01", "2024-01-01", "2026-07-24"))
HOLDS = (1, 5, 10, 15, 30, 60)
NAMES = ["rev_1", "rev_5", "rev_15", "mom_30", "rel_5", "rel_15", "day_ret", "vwap_dev", "range_pos", "bar_pos", "vol_spike"]


def rth(df):
    hm = df.index.hour * 60 + df.index.minute
    df = df[(hm >= 570) & (hm < 960)].copy()
    df.index = df.index + pd.Timedelta(minutes=1)        # label by bar END: the price is known then
    return df


def stock_frame(df, spy_r):
    c, v = df["close"].astype(float), df["volume"].astype(float)
    day = c.index.normalize()
    g = c.groupby(day)
    r1 = (c / g.shift(1) - 1).fillna(c / df["open"] - 1)
    s1 = r1.groupby(day).std().shift(1).rolling(20, min_periods=10).mean().reindex(day).to_numpy()   # previous 20 days
    rk = {k: c / g.shift(k) - 1 for k in (1, 5, 15, 30)}
    first = df["open"].groupby(day).transform("first")
    pv = (df["vwap"] * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    hi, lo = df["high"].groupby(day).cummax(), df["low"].groupby(day).cummin()
    slot = c.index.hour * 60 + c.index.minute
    v5 = v.groupby(day).rolling(5, min_periods=5).sum().droplevel(0)
    vs = pd.Series(v5.to_numpy(), index=pd.MultiIndex.from_arrays([day, slot])).unstack()
    vbase = vs.rolling(20, min_periods=10).mean().shift(1).stack().reindex(pd.MultiIndex.from_arrays([day, slot])).to_numpy()
    spy = spy_r.reindex(c.index)
    sc = lambda k: s1 * np.sqrt(k)
    f = pd.DataFrame({
        "rev_1": rk[1] / sc(1), "rev_5": rk[5] / sc(5), "rev_15": rk[15] / sc(15), "mom_30": rk[30] / sc(30),
        "rel_5": (rk[5] - spy["r5"]) / sc(5), "rel_15": (rk[15] - spy["r15"]) / sc(15),
        "day_ret": (c / first - 1) / sc(30), "vwap_dev": (c / pv - 1) / sc(15), "range_pos": (c - lo) / (hi - lo) - 0.5,
        "bar_pos": (c - df["low"]) / (df["high"] - df["low"]) - 0.5, "vol_spike": np.log(v5.to_numpy() / vbase),
        **{f"y{h}": g.shift(-h) / c - 1 for h in HOLDS},
    }, index=c.index)
    keep = (slot >= 585) & (slot % 5 == 0)                  # entries every 5 minutes from 9:45
    return f[keep].replace([np.inf, -np.inf], np.nan)


def panel(end=HOLD_END):
    B = alpaca_data.load_bars()
    sp = pd.read_pickle(HERE / "data" / "alpaca_spreads.pkl").groupby("symbol")["spread"].median()
    spy = rth(B["SPY"])["close"].astype(float)
    sg = spy.groupby(spy.index.normalize())
    spy_r = pd.DataFrame({"r5": spy / sg.shift(5) - 1, "r15": spy / sg.shift(15) - 1})
    parts = []
    for s, df in B.items():
        f = stock_frame(rth(df), spy_r)
        f["symbol"], f["cost"] = s, float(sp.get(s, sp.median()))
        parts.append(f[f.index < end])
    p = pd.concat(parts)
    p = p[p.index >= pd.Timestamp("2019-02-01", tz=ET)]     # first month only warms up the 20-day baselines
    for h in HOLDS:  # an entry must exit by 16:00
        p.loc[(p.index.hour * 60 + p.index.minute) > 960 - h, f"y{h}"] = np.nan
    return p, B


def zcs(p, col):
    g = p.groupby(level=0)[col]
    return ((p[col] - g.transform("mean")) / g.transform("std")).clip(-5, 5)


def money(p, score, h, top=5, min_z=None):
    """Top `top` stocks by score at each entry time (only z >= min_z if set). net = return − measured spread."""
    d = pd.DataFrame({"s": score, "y": p[f"y{h}"], "c": p["cost"]}).dropna()
    if min_z is not None:
        d = d[d["s"] >= min_z]
    if d.empty:
        return {"net": None, "gross": None, "cost": None, "t_net": None, "win": None, "times": 0}
    picks = d.sort_values("s").groupby(level=0).tail(top)
    net = (picks["y"] - picks["c"]).groupby(level=0).mean()
    lags = max(1, h // 5)
    return {"net": float(net.mean()), "gross": float(picks["y"].groupby(level=0).mean().mean()), "cost": float(picks["c"].mean()),
            "t_net": float(ss.nw_t(net.sort_index().to_numpy(), lags)) if len(net) > 50 else None,
            "win": float((net > 0).mean()), "times": int(len(net))}


def spy_momentum(B, start, end):
    s = rth(B["SPY"])
    s = s[(s.index >= start) & (s.index < end)]
    out = []
    for d, x in s.groupby(s.index.normalize()):
        t = x["close"]
        try:
            first = t.at[d + pd.Timedelta(hours=10)] / x["open"].iloc[0] - 1
            last = t.at[d + pd.Timedelta(hours=16)] / t.at[d + pd.Timedelta(hours=15, minutes=30)] - 1
        except KeyError:
            continue
        out.append(np.sign(first) * last)
    return np.array(out)


def scores(p, Z, h, cut):
    """All candidate scores for horizon h, with every choice (signs, model weights) made on SEARCH rows only."""
    y = p[f"y{h}"]
    st = {n: ss.fm(Z[n][cut["search"]], y[cut["search"]], p.index[cut["search"]], lags=max(1, h // 5)) for n in NAMES}
    signs = {n: float(np.sign(r[1])) for n, r in st.items() if r and abs(r[1]) >= 2}
    items = {n: Z[n] for n in NAMES}
    meta = {}
    if signs:
        items["combo"] = pd.concat([signs[n] * Z[n] for n in signs], axis=1).mean(axis=1)
        meta["combo"] = signs
    S = pd.concat([Z, y.rename("y")], axis=1)[cut["search"]].dropna()
    yd = S["y"] - S.groupby(level=0)["y"].transform("mean")       # per-time excess return
    b = np.linalg.lstsq(S[NAMES].to_numpy(), yd.to_numpy(), rcond=None)[0]
    items["model"] = Z[NAMES].fillna(0) @ b
    meta["model"] = dict(zip(NAMES, map(float, b)))
    return items, meta


def run():
    p, B = panel()
    cut = {"search": p.index < SEARCH_END, "confirm": (p.index >= SEARCH_END) & (p.index < CONFIRM_END)}
    Z = pd.DataFrame({n: zcs(p, n) for n in NAMES}, index=p.index)
    ledger = []
    for h in HOLDS:
        y = p[f"y{h}"]
        items, meta = scores(p, Z, h, cut)
        for name, sc in items.items():
            r = {per: ss.fm(sc[m], y[m], p.index[m], lags=max(1, h // 5)) for per, m in cut.items()}
            sign = float(np.sign(r["search"][1])) if r["search"] else 1.0
            row = {"signal": name, "h": h, "search_t": r["search"][1] if r["search"] else None,
                   "confirm_t": r["confirm"][1] if r["confirm"] else None, "sign": sign, "meta": meta.get(name),
                   "money_confirm": money(p[cut["confirm"]], sign * sc[cut["confirm"]], h),
                   "money_confirm_extreme": money(p[cut["confirm"]], sign * sc[cut["confirm"]], h, min_z=2.0) if name == "combo" else None}
            row["promoted"] = bool(row["search_t"] and row["confirm_t"] and abs(row["search_t"]) >= 3 and abs(row["confirm_t"]) >= 2
                                   and np.sign(row["search_t"]) == np.sign(row["confirm_t"]))
            ledger.append(row)
            print(f"h={h:2d} {name:9s} search t {row['search_t'] or 0:+7.2f} confirm t {row['confirm_t'] or 0:+7.2f} "
                  f"net/trade {row['money_confirm']['net'] * 100 if row['money_confirm']['net'] is not None else float('nan'):+.4f}% "
                  f"{'PROMOTED' if row['promoted'] else ''}", flush=True)
    spy = {per: spy_momentum(B, a, b) for per, (a, b) in {"search": (pd.Timestamp("2019-01-01", tz=ET), SEARCH_END), "confirm": (SEARCH_END, CONFIRM_END)}.items()}
    spy_row = {per: {"mean": float(x.mean()), "t": float(x.mean() / (x.std() / np.sqrt(len(x)))), "days": len(x)} for per, x in spy.items()}
    spy_row["promoted"] = bool(spy_row["search"]["t"] >= 3 and spy_row["confirm"]["t"] >= 2)
    promoted = [r for r in ledger if r["promoted"]]
    best = max(promoted, key=lambda r: r["money_confirm"]["net"] or -1) if promoted else None
    out = {"generated": str(pd.Timestamp.now()), "symbols": int(p["symbol"].nunique()),
           "median_spread": float(p.groupby("symbol")["cost"].first().median()), "tests": ledger, "spy_momentum": spy_row,
           "best": {"signal": best["signal"], "h": best["h"]} if best else None}
    (HERE / "intraday_v2_ledger.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def holdout():
    """Scores the frozen promoted list ONCE on 2024-01-01 .. 2026-07-23."""
    if (HERE / "intraday_v2_holdout.json").exists():
        raise SystemExit("Already scored once; scoring again would turn the holdout into a search period.")
    L = json.loads((HERE / "intraday_v2_ledger.json").read_text())
    frozen = [r for r in L["tests"] if r["promoted"]]
    p, B = panel()
    cut = {"search": p.index < SEARCH_END, "holdout": p.index >= CONFIRM_END}
    Z = pd.DataFrame({n: zcs(p, n) for n in NAMES}, index=p.index)
    k = len(frozen) + int(L["spy_momentum"]["promoted"]) + 1   # + the cost-minimizing variant
    thr = NormalDist().inv_cdf(1 - 0.05 / max(k, 1))
    res = []
    for r in frozen:
        h = r["h"]
        items, _ = scores(p, Z, h, cut)                     # rebuilt from SEARCH only, exactly as frozen
        sc = r["sign"] * items[r["signal"]]
        m = cut["holdout"]
        fm = ss.fm(sc[m], p[f"y{h}"][m], p.index[m], lags=max(1, h // 5))
        mo = money(p[m], sc[m], h)
        ok = bool(fm and fm[1] >= thr and mo["net"] is not None and mo["net"] > 0 and (mo["t_net"] or 0) >= 2)
        res.append({**{x: r[x] for x in ("signal", "h")}, "holdout_t": fm[1] if fm else None, "threshold": thr, "money": mo, "works": ok})
    items, _ = scores(p, Z, 60, cut)                       # cost-minimizing variant (pre-declared above)
    m = cut["holdout"] & (p["cost"] <= 0.0001).to_numpy()
    d = pd.DataFrame({"s": items["model"][m], "y": p["y60"][m], "c": p["cost"][m]}).dropna()
    top = d.sort_values("s").groupby(level=0).tail(1)
    net = (top["y"] - top["c"]).sort_index()
    t_net = float(ss.nw_t(net.to_numpy(), 12))
    res.append({"signal": "cheap_h60", "h": 60, "holdout_t": t_net, "threshold": thr,
                "money": {"net": float(net.mean()), "gross": float(top["y"].mean()), "cost": float(top["c"].mean()), "t_net": t_net,
                          "win": float((net > 0).mean()), "times": int(len(net))},
                "works": bool(net.mean() > 0 and t_net >= max(thr, 2))})
    if L["spy_momentum"]["promoted"]:
        x = spy_momentum(B, CONFIRM_END, HOLD_END)
        t = float(x.mean() / (x.std() / np.sqrt(len(x))))
        res.append({"signal": "spy_momentum", "h": 30, "holdout_t": t, "threshold": thr, "money": {"net": float(x.mean() - 0.00003)},
                    "works": bool(t >= thr and x.mean() > 0.00003)})
    (HERE / "intraday_v2_holdout.json").write_text(json.dumps({"scored": str(pd.Timestamp.now()), "results": res}, indent=1, default=float))
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["holdout"]:
        for r in holdout():
            print(r)
    else:
        out = run()
        print("best:", out["best"], "| SPY momentum:", out["spy_momentum"])
