"""Broad signal search: price action, market/macro, news attention, SEC filings, analysts, earnings.

PRE-REGISTERED (written before any result was seen):
  Universe  ~500 S&P 500 stocks (today's members: survivorship bias, so judge signals RELATIVE to the
            other stocks on the same day, where that bias mostly cancels).
  Trades    h0  buy at D's open, sell at D's close (day trade)      h1  ... sell at D+1's close (2-day hold)
            ovn buy at the CLOSE of t, sell at D's open (overnight; only signals known at t's close)
            h4 / h20  ... sell 5 / 21 trading days later (for comparison: research says durable signals live here)
  Revised 2026-10-17 BEFORE the first run, after reading quant-research.md: added ovn/h4/h20, the VIX and
  bear-market interactions, industry momentum and one fixed-sign composite. Still no result had been seen.
  Splits    SEARCH 2010–2018 · CONFIRM 2019–2022 · HOLDOUT 2023–today. The holdout is not loaded during the
            search: run() drops it, and holdout() only scores the frozen list in promoted.json.
  Measures  xs  = does the signal pick which stocks beat the others that day? (daily cross-sectional slope,
                 Fama-MacBeth; Newey-West t over days, lags = hold length, because multi-day holds overlap).
                 Stock-level signals, all five trades.
            abs = does it predict the stock's own direction? (pooled regression, t clustered by day with the same
                 Newey-West lags). ovn/h0/h1 only.
  Composite the average of each day's cross-sectional z-scores of the research-backed signals, with signs FIXED
            from the literature (not fitted): +12-month momentum, +industry momentum, −last month, −last week,
            −volatility, −lottery (max daily jump). One test per trade.
  Promote   |t| >= 3 in SEARCH and same sign with |t| >= 2 in CONFIRM. Every test is logged in search_ledger.json,
            so the number of tries is visible (with ~250 tests, ~1 false pass at t>=3 is expected in SEARCH alone;
            needing CONFIRM too makes a false promotion rare).
  Holdout   each promoted test is scored ONCE. Confirmed if same sign with t >= the Bonferroni level for the
            number promoted (one-sided 5% overall). "Helps" = confirmed AND trading it beats the alternative
            after costs: xs -> buying its top-fifth of stocks beats buying a random stock; abs -> same rule as
            daily_model.py.

Run: .venv/bin/python signal_search.py            (search + confirm, writes search_ledger.json, promoted.json)
     .venv/bin/python signal_search.py holdout    (scores promoted.json once on 2023–today)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import search_data as sd
from shortterm import rsi

HERE = Path(__file__).parent
SEARCH, CONFIRM, HOLDOUT = "2010-01-01", "2019-01-01", "2023-01-01"
COST = 0.0010  # 10 bps round trip: S&P 500 names are a bit less liquid than the mega-caps
TARGETS = {"ovn": 0, "h0": 0, "h1": 1, "h4": 4, "h20": 20}  # trade -> Newey-West lags (days of overlap)
ABS_TARGETS = ("ovn", "h0", "h1")
COMPOSITE = {"mom_12_1": 1, "industry_mom": 1, "ret_21d": -1, "ret_5d": -1, "volatility": -1, "max_ret_21": -1}
SEC_ITEMS = {"801": "8.01", "701": "7.01", "101": "1.01", "502": "5.02", "205": "2.05", "202": "2.02"}

DESC = {
    # price action (stock level)
    "ret_1d": "Yesterday's move", "ret_5d": "Last 5 days' move", "ret_21d": "Last month's move",
    "mom_12_1": "12-month trend (skip last month)", "intraday": "Yesterday open→close", "overnight": "Yesterday's gap",
    "gap_D": "Today's opening gap", "gap_D_rel": "Today's gap vs the S&P 500's gap", "close_loc": "Close position in yesterday's range",
    "range": "Yesterday's high–low range (vs normal)", "vol_surprise": "Yesterday's volume vs normal",
    "volatility": "20-day volatility", "max_ret_21": "Biggest 1-day jump in the last month (lottery)",
    "near_high": "Distance below 52-week high", "dist_ma20": "Distance from 20-day average",
    "dist_ma200": "Distance from 200-day average", "rsi2": "RSI(2)", "ovn_mom_21": "Last month's overnight gains",
    "intra_mom_21": "Last month's daytime gains", "streak": "Up/down days in a row", "dollar_vol": "Trading volume in $ (size)",
    "amihud": "Illiquidity (move per $ traded)", "price_level": "Share price level", "ret_x_volume": "Yesterday's move × volume surprise",
    "sector_ret_1d": "Its sector's move yesterday", "rel_sector_1d": "Yesterday's move vs its sector",
    "industry_mom": "Its sector's 12-month trend", "composite": "Research-backed combo (fixed signs)",
    "rev_x_vix": "Last week's move × fear level (VIX)", "mom_x_bear": "12-month trend × market down over 2 years",
    # market / calendar / macro (same for every stock on a day)
    "spy_gap_D": "S&P 500 opening gap today", "spy_ret_1d": "S&P 500 move yesterday", "vix_chg": "VIX change yesterday",
    "vix_level": "VIX level", "spy_bear": "S&P 500 down over the last 2 years",
    "btc_ret": "Bitcoin move overnight (to 8 PM ET)", "nikkei_D": "Japan's market today (closes before US open)",
    "tnx_chg": "10-year yield change", "dollar_chg": "Dollar change", "oil_chg": "Oil change", "gold_chg": "Gold change",
    "monday": "Trade day is Monday", "friday": "Trade day is Friday", "turn_of_month": "Turn of the month",
    "pre_holiday": "Day before a market holiday", "post_holiday": "Day after a market holiday",
    "fomc_day": "Fed decision day", "pre_fomc": "Day before a Fed decision",
    # news attention (Wikipedia)
    "wiki_spike": "Wikipedia views spike (day before trade day)", "wiki_spike_safe": "Wikipedia views spike (2 days before)",
    "wiki_x_ret": "Views spike × yesterday's move", "wiki_trend": "Wikipedia views trend (week vs 3 months)",
    # SEC filings (official news)
    "sec_8k_overnight": "8-K news filed overnight (after close, before open)", "sec_8k_day": "8-K news filed during yesterday's session",
    "sec_8k_recent5": "8-K filings in the last 5 days",
    **{f"sec_{k}_overnight": f"8-K item {v} filed overnight" for k, v in SEC_ITEMS.items()},
    "sec_13d_recent5": "Activist stake (13D) filed, last 5 days", "sec_13g_recent5": "Large passive stake (13G), last 5 days",
    "sec_424b_recent5": "Share/debt sale prospectus, last 5 days", "sec_form4_recent5": "Insider trade filings, last 5 days",
    "sec_nt_recent5": "Late-filing notice, last 5 days",
    # analysts / earnings
    "analyst_net_1d": "Analyst upgrades − downgrades (day)", "analyst_net_20d": "Analyst upgrades − downgrades (20 days)",
    "target_net_5d": "Price-target raises − cuts (5 days)", "earn_new": "Earnings surprise released overnight",
    "earn_recent": "Earnings surprise, previous 10 days", "earn_ahead": "Earnings during a 2-day hold",
}
MARKET = ["spy_gap_D", "spy_ret_1d", "vix_chg", "vix_level", "spy_bear", "btc_ret", "nikkei_D", "tnx_chg", "dollar_chg", "oil_chg", "gold_chg",
          "monday", "friday", "turn_of_month", "pre_holiday", "post_holiday", "fomc_day", "pre_fomc"]
WIKI_START = "2015-08-01"
# Known at t's close (the overnight trade buys then): nothing from after 4 PM, D's open, or later news.
NOT_AT_CLOSE = {"gap_D", "gap_D_rel", "spy_gap_D", "btc_ret", "nikkei_D", "dollar_chg", "wiki_spike", "wiki_spike_safe",
                "wiki_trend", "wiki_x_ret", "sec_8k_overnight", "sec_8k_recent5", "sec_13d_recent5", "sec_13g_recent5",
                "sec_424b_recent5", "sec_form4_recent5", "sec_nt_recent5", "analyst_net_1d", "analyst_net_20d",
                "target_net_5d", "earn_new"} | {f"sec_{k}_overnight" for k in SEC_ITEMS}


def market_features(cal, macro, fomc):
    """One row per trading day t (cal = SPY's trading days). Values known before the open of D = next trading day."""
    spy = macro["spy"].reindex(cal)
    entry = pd.DatetimeIndex(list(cal[1:]) + [cal[-1] + pd.offsets.BDay(1)])
    after = pd.DatetimeIndex(list(entry[1:]) + [entry[-1] + pd.offsets.BDay(1)])  # trading day after D
    sr = spy["Close"].pct_change()
    ssd = sr.rolling(20).std()
    lvl = lambda k: macro[k]["Close"].reindex(cal, method="ffill")
    btc = macro["btc"]["Close"].pct_change()
    nk = macro["nikkei"]["Close"].pct_change()
    fomc = pd.DatetimeIndex(fomc)
    mstart = entry.values.astype("datetime64[M]").astype("datetime64[D]")
    pos = np.busday_count(mstart, entry.values.astype("datetime64[D]"))
    return pd.DataFrame({
        "spy_gap_D": (spy["Open"].shift(-1) / spy["Close"] - 1) / ssd,
        "spy_ret_1d": sr / ssd,
        "vix_chg": lvl("vix").pct_change(),
        "vix_level": np.log(lvl("vix")),
        "spy_bear": (spy["Close"] / spy["Close"].shift(504) - 1 < 0).astype(float).where(spy["Close"].shift(504).notna()),
        "btc_ret": btc.reindex(entry - pd.Timedelta(days=1)).to_numpy(),  # day ending 8 PM ET the evening before D
        "nikkei_D": nk.reindex(entry).fillna(0).to_numpy(),
        "tnx_chg": lvl("tnx").diff(),
        "dollar_chg": lvl("dollar").pct_change(),
        "oil_chg": lvl("oil").pct_change().clip(-0.5, 0.5),  # 2020 negative oil price
        "gold_chg": lvl("gold").pct_change(),
        "monday": (entry.dayofweek == 0).astype(float),
        "friday": (entry.dayofweek == 4).astype(float),
        "turn_of_month": ((pos < 3) | ((entry + pd.offsets.BDay(1)).month != entry.month)).astype(float),
        "pre_holiday": (after > entry + pd.offsets.BDay(1)).astype(float),
        "post_holiday": (entry > cal + pd.offsets.BDay(1)).astype(float),
        "fomc_day": entry.isin(fomc).astype(float),
        "pre_fomc": after.isin(fomc).astype(float),
    }, index=cal)


def stock_features(df, spy_gap, cal):
    o, h, l, c, v = (df[k].astype("float64") for k in ("Open", "High", "Low", "Close", "Volume"))
    r = c.pct_change()
    s = r.rolling(20).std()
    ovn, intra = o / c.shift() - 1, c / o - 1
    gap = o.shift(-1) / c - 1  # today's open vs yesterday's close: known at the open we buy at
    vs = np.log(v / v.rolling(20).mean())
    sgn = np.sign(r)
    streak = sgn * (sgn.groupby((sgn != sgn.shift()).cumsum()).cumcount() + 1)
    f = pd.DataFrame({
        "ret_1d": r / s, "ret_5d": c.pct_change(5) / (s * np.sqrt(5)), "ret_21d": c.pct_change(21) / (s * np.sqrt(21)),
        "mom_12_1": (c.shift(21) / c.shift(252) - 1) / (r.rolling(252).std() * np.sqrt(231)),
        "intraday": intra / s, "overnight": ovn / s, "gap_D": gap / s,
        "gap_D_rel": (gap - spy_gap.reindex(c.index)) / s,
        "close_loc": (c - l) / (h - l) - 0.5, "range": (h - l) / c / s, "vol_surprise": vs,
        "volatility": np.log(s), "max_ret_21": r.rolling(21).max() / s,
        "near_high": c / c.rolling(252).max() - 1, "dist_ma20": (c / c.rolling(20).mean() - 1) / s,
        "dist_ma200": c / c.rolling(200).mean() - 1, "rsi2": (rsi(c) - 50) / 50,
        "ovn_mom_21": ovn.rolling(21).sum() / s, "intra_mom_21": intra.rolling(21).sum() / s, "streak": streak,
        "dollar_vol": np.log((c * v).rolling(20).mean()), "amihud": np.log((r.abs() / (c * v)).rolling(21).mean()),
        "price_level": np.log(c), "ret_x_volume": (r / s) * vs,
        "yh0": c.shift(-1) / o.shift(-1) - 1, "yh1": c.shift(-2) / o.shift(-1) - 1, "yovn": o.shift(-1) / c - 1,
        "yh4": c.shift(-5) / o.shift(-1) - 1, "yh20": c.shift(-21) / o.shift(-1) - 1,
    })
    # a missing day in this stock's history must not shift D: only keep rows whose next row is the next trading day
    nxt = pd.Series(c.index, index=c.index).shift(-1)
    cal_next = pd.Series(cal[1:], index=cal[:-1]).reindex(c.index)
    f.loc[nxt != cal_next, ["gap_D", "gap_D_rel", "yh0", "yh1", "yovn", "yh4", "yh20"]] = np.nan
    return f.replace([np.inf, -np.inf], np.nan)


def wiki_features(views, cal):
    """views: daily (UTC) page views. D-1 = the calendar day before the trade day (complete by ~8 PM ET)."""
    if views is None or len(views) < 120:
        return None
    v = np.log1p(views.asfreq("D").fillna(0))
    base = v.rolling(30).median().shift(1)
    spike = v - base
    trend = np.log1p(views.asfreq("D").fillna(0).rolling(7).mean()) - np.log1p(views.asfreq("D").fillna(0).rolling(90).mean())
    entry = pd.DatetimeIndex(list(cal[1:]) + [cal[-1] + pd.offsets.BDay(1)])
    d1, d2 = entry - pd.Timedelta(days=1), entry - pd.Timedelta(days=2)
    return pd.DataFrame({"wiki_spike": spike.reindex(d1).to_numpy(), "wiki_spike_safe": spike.reindex(d2).to_numpy(),
                         "wiki_trend": trend.reindex(d2).to_numpy()}, index=cal)


def sec_features(f, cal):
    """f: filings with acceptance time (UTC). Row t = filings that became public by the open of D (next day)."""
    cols = [k for k in DESC if k.startswith("sec_")]
    n = len(cal)
    if f is None or not len(f):
        return pd.DataFrame(0.0, index=cal, columns=cols)
    # The "Z" is real UTC: EDGAR's 10 PM Eastern filing-date cutoff lands at 02Z in July and 03Z in January.
    t = pd.to_datetime(f["acceptanceDateTime"].str[:19]).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).to_numpy()
    ext = cal.append(pd.DatetimeIndex([cal[-1] + pd.offsets.BDay(1)]))
    k = (ext + pd.Timedelta(hours=9, minutes=30)).searchsorted(t, side="right")  # D = ext[k]: first open after filing
    row = k - 1                                                                     # row t = the trading day before D
    ok = (row >= 0) & (row < n)
    overnight = np.zeros(len(t), bool)
    overnight[ok] = t[ok] >= (ext[row[ok]] + pd.Timedelta(hours=16)).to_numpy()   # after t's close
    day = ok & ~overnight                                                           # during t's session
    form, items = f["form"].to_numpy(dtype=str), f["items"].fillna("").to_numpy(dtype=str)
    count = lambda m: np.bincount(row[m & ok], minlength=n)[:n].astype(float)
    is8k = form == "8-K"
    out = {"sec_8k_overnight": count(is8k & overnight), "sec_8k_day": count(is8k & day)}
    for key, item in SEC_ITEMS.items():
        out[f"sec_{key}_overnight"] = count(is8k & overnight & np.array([item in x for x in items]))
    recent = {"sec_8k_recent5": is8k, "sec_13d_recent5": np.isin(form, ["SC 13D", "SCHEDULE 13D"]),
              "sec_13g_recent5": np.isin(form, ["SC 13G", "SCHEDULE 13G"]), "sec_424b_recent5": np.char.startswith(form, "424B"),
              "sec_form4_recent5": form == "4", "sec_nt_recent5": np.char.startswith(form, "NT ")}
    for col, m in recent.items():
        out[col] = np.log1p(pd.Series(count(m)).rolling(5, min_periods=1).sum().to_numpy())
    return pd.DataFrame(out, index=cal)[cols]


def build(end=None):
    """The full panel (date, ticker) of features and targets. end: drop rows from this date on."""
    u, prices, macro = sd.load("universe"), sd.load("prices"), sd.load("macro")
    wiki, sec, events, fomc = sd.load("wiki"), sd.load("sec"), sd.load("events"), sd.load("fomc")
    import qualitative
    cal = macro["spy"].index[macro["spy"].index >= "2009-01-01"]
    if end:
        cal = cal[cal < end]  # prices from `end` on are never loaded, so no return can reach into them
    mk = market_features(cal, macro, fomc)
    spy_gap = macro["spy"]["Open"].shift(-1) / macro["spy"]["Close"] - 1
    sector = dict(zip(u["ticker"], u["sector"]))
    rows = []
    for t, df in prices.items():
        df = df[df.index.isin(cal)]
        if len(df) < 300:
            continue
        f = stock_features(df, spy_gap, cal)
        idx = f.index
        w = wiki_features(wiki.get(t), cal)
        f = f.join(w if w is not None else pd.DataFrame(np.nan, index=cal, columns=["wiki_spike", "wiki_spike_safe", "wiki_trend"]))
        f["wiki_x_ret"] = f["wiki_spike_safe"] * f["ret_1d"]
        f = f.join(sec_features(sec.get(t), cal).reindex(idx))
        ud, ed = events.get(t, (None, None))
        f = f.join(qualitative.q_features(idx, ud, ed))
        f["ticker"], f["sector"] = t, sector.get(t)
        rows.append(f.astype({c: "float32" for c in f.columns if c not in ("ticker", "sector")}))
    p = pd.concat(rows)
    p.index.name = "date"
    p = p[p.index >= SEARCH]
    if end:
        p = p[p.index < end]
    g = p.groupby([p.index, p["sector"]])["ret_1d"]
    tot, cnt = g.transform("sum"), g.transform("count")
    p["sector_ret_1d"] = ((tot - p["ret_1d"]) / (cnt - 1)).astype("float32")
    p["rel_sector_1d"] = (p["ret_1d"] - p["sector_ret_1d"]).astype("float32")
    p = p.merge(mk, left_index=True, right_index=True, how="left")
    g = p.groupby([p.index, p["sector"]])["mom_12_1"]
    p["industry_mom"] = ((g.transform("sum") - p["mom_12_1"].fillna(0)) / (g.transform("count") - p["mom_12_1"].notna())).astype("float32")
    p["rev_x_vix"] = (p["ret_5d"] * (p["vix_level"] - np.log(20))).astype("float32")  # 20 = a typical VIX
    p["mom_x_bear"] = (p["mom_12_1"] * p["spy_bear"]).astype("float32")
    zs = []
    for c, sign in COMPOSITE.items():
        g = p.groupby(level=0)[c]
        zs.append(sign * (p[c] - g.transform("mean")) / g.transform("std"))
    p["composite"] = pd.concat(zs, axis=1).clip(-5, 5).mean(axis=1).astype("float32")
    for c in ("wiki_spike", "wiki_spike_safe", "wiki_trend", "wiki_x_ret"):
        p.loc[p.index < WIKI_START, c] = np.nan
    return p


def nw_t(x, lags):
    """t-stat of the mean of a daily series, Newey-West corrected for `lags` days of overlap."""
    x = np.asarray(x, float)
    e = x - x.mean()
    v = e @ e / len(x)
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / len(x)
    return x.mean() / np.sqrt(v / len(x))


def fm(f, y, dates, lags=0):
    """Daily cross-sectional slope of y on f (Fama-MacBeth). Returns (mean slope, t over days, days)."""
    d = pd.DataFrame({"f": f, "y": y, "d": dates}).dropna()
    g = d.groupby("d")
    fd, yd = d["f"] - g["f"].transform("mean"), d["y"] - g["y"].transform("mean")
    num, den = (fd * yd).groupby(d["d"]).sum(), (fd * fd).groupby(d["d"]).sum()
    s = (num / den)[den > 1e-12]
    if len(s) < 100:
        return None
    return float(s.mean()), float(nw_t(s.sort_index().to_numpy(), lags)), int(len(s))


def pooled(f, y, dates, lags=0):
    """Pooled slope with t clustered by day (Driscoll-Kraay: Newey-West across days for overlapping holds)."""
    d = pd.DataFrame({"f": f, "y": y, "d": dates}).dropna()
    if d["d"].nunique() < 100 or d["f"].std() == 0:
        return None
    X = np.column_stack([np.ones(len(d)), d["f"].to_numpy()])
    inv = np.linalg.inv(X.T @ X)
    b = inv @ X.T @ d["y"].to_numpy()
    g = pd.DataFrame(X * (d["y"].to_numpy() - X @ b)[:, None]).groupby(d["d"].to_numpy()).sum().sort_index().to_numpy()
    meat = g.T @ g
    for l in range(1, lags + 1):
        c = g[l:].T @ g[:-l]
        meat += (1 - l / (lags + 1)) * (c + c.T)
    return float(b[1]), float(b[1] / np.sqrt((inv @ meat @ inv)[1, 1])), int(d["d"].nunique())


def zscore(p, names, ref):
    m, s = ref[names].mean(), ref[names].std().replace(0, 1)
    return ((p[names] - m) / s).clip(-5, 5), m, s


def tests(p, names, z):
    """Every (feature, measure, horizon) result for panel p."""
    out = []
    for n in names:
        for h, lags in TARGETS.items():
            if h == "ovn" and n in NOT_AT_CLOSE:
                continue
            y = p[f"y{h}"]
            for meas in ("xs", "abs"):
                if (meas == "xs" and n in MARKET) or (meas == "abs" and h not in ABS_TARGETS):
                    continue
                r = (fm if meas == "xs" else pooled)(z[n], y, p.index, lags)
                out.append({"feature": n, "measure": meas, "h": h, **({"b": r[0], "t": r[1], "days": r[2]} if r else {"t": None})})
    return out


def run():
    p = build(end=HOLDOUT)  # the holdout years are never loaded here
    names = [n for n in DESC if n in p.columns]
    z, _, _ = zscore(p, names, p[p.index < CONFIRM])
    srch, conf = p.index < CONFIRM, p.index >= CONFIRM
    a, b = tests(p[srch], names, z[srch]), tests(p[conf], names, z[conf])
    ledger = []
    for x, y in zip(a, b):
        ok = x["t"] is not None and y["t"] is not None and abs(x["t"]) >= 3 and abs(y["t"]) >= 2 and np.sign(x["t"]) == np.sign(y["t"])
        ledger.append({"feature": x["feature"], "desc": DESC[x["feature"]], "measure": x["measure"], "h": x["h"],
                       "search_t": x["t"], "confirm_t": y["t"], "search_b": x.get("b"), "confirm_b": y.get("b"),
                       "coverage": float(p[x["feature"]].notna().mean()), "promoted": bool(ok)})
    (HERE / "search_ledger.json").write_text(json.dumps({"generated": str(pd.Timestamp.now()), "rows": len(p),
                                                         "stocks": int(p["ticker"].nunique()), "tests": ledger}, indent=1))
    promoted = [{k: r[k] for k in ("feature", "measure", "h", "search_t", "confirm_t")} for r in ledger if r["promoted"]]
    (HERE / "promoted.json").write_text(json.dumps({"frozen": str(pd.Timestamp.now()), "promoted": promoted}, indent=1))
    return ledger


def show(ledger):
    print(f"{len(ledger)} tests, {sum(r['promoted'] for r in ledger)} promoted")
    for r in sorted(ledger, key=lambda r: -abs(r["search_t"] or 0)):
        st = "—" if r["search_t"] is None else f"{r['search_t']:+6.2f}"
        ct = "—" if r["confirm_t"] is None else f"{r['confirm_t']:+6.2f}"
        print(f"{r['feature']:20s} {r['measure']:3s} {r['h']:4s} search t {st}  confirm t {ct}  {'PROMOTED' if r['promoted'] else ''}")


def holdout():
    """Score the frozen promoted.json ONCE on 2023–today (written before any holdout number was seen).
    "Helps": xs -> each day buy the top fifth by the signal (bottom fifth if its sign is negative); its excess over
    the average stock must be > 0 with t >= 2. abs -> fit y = a + b·signal on 2010–2022, trade its direction when the
    forecast beats costs; must beat always-buying with t >= 2 and make money after costs."""
    from statistics import NormalDist
    frozen = json.loads((HERE / "promoted.json").read_text())["promoted"]
    if not frozen:
        return []
    p = build()
    names = sorted({x["feature"] for x in frozen})
    z, _, _ = zscore(p, names, p[p.index < CONFIRM])  # same scaling as the search
    pre, ho = p.index < HOLDOUT, p.index >= HOLDOUT
    thr = NormalDist().inv_cdf(1 - 0.05 / len(frozen))
    out = []
    for x in frozen:
        n, h, sign = x["feature"], x["h"], np.sign(x["search_t"])
        y = p[f"y{h}"]
        r = (fm if x["measure"] == "xs" else pooled)(z[n][ho], y[ho], p.index[ho], TARGETS[h])
        res = {**x, "holdout_t": r[1] if r else None, "holdout_b": r[0] if r else None, "threshold": thr}
        res["confirmed"] = bool(r and np.sign(r[1]) == sign and abs(r[1]) >= thr)
        if x["measure"] == "xs":
            d = pd.DataFrame({"f": z[n][ho] * sign, "y": y[ho], "d": p.index[ho]}).dropna()
            rk = d.groupby("d")["f"].rank(pct=True)
            ex = (d["y"] - d.groupby("d")["y"].transform("mean"))[rk >= 0.8].groupby(d["d"][rk >= 0.8]).mean()
            res["top_fifth_excess"], res["excess_t"] = float(ex.mean()), float(nw_t(ex.sort_index().to_numpy(), TARGETS[h]))
            res["top_fifth_after_cost"] = float(d["y"][rk >= 0.8].groupby(d["d"][rk >= 0.8]).mean().mean() - COST)
            res["helps"] = bool(res["confirmed"] and res["top_fifth_excess"] > 0 and res["excess_t"] >= 2)
        else:
            fit = pd.DataFrame({"f": z[n][pre], "y": y[pre]}).dropna()
            b1, b0 = np.polyfit(fit["f"], fit["y"], 1)
            d = pd.DataFrame({"f": z[n][ho], "y": y[ho], "d": p.index[ho]}).dropna()
            pred = b0 + b1 * d["f"]
            pos = np.where(pred > COST, 1, np.where(pred < -COST, -1, 0))
            strat = pd.Series(np.where(pos != 0, pos * d["y"] - COST, 0.0)).groupby(d["d"].to_numpy()).mean()
            base = (d["y"] - COST).groupby(d["d"]).mean().to_numpy()
            diff = strat.to_numpy() - base
            res["strategy_avg"], res["edge"] = float(strat.mean()), float(diff.mean())
            res["edge_t"] = float(nw_t(diff, TARGETS[h]))
            res["helps"] = bool(res["confirmed"] and res["edge_t"] >= 2 and res["strategy_avg"] > 0)
        if x["feature"] in json.loads((HERE / "promoted.json").read_text()).get("not_eligible_to_help", {}).get("features", []):
            res["helps"], res["note"] = False, "survivorship-biased (size/liquidity): cannot count"
        out.append(res)
    (HERE / "holdout_results.json").write_text(json.dumps({"scored": str(pd.Timestamp.now()), "results": out}, indent=1))
    for r in out:
        print(f"{r['feature']:20s} {r['measure']} {r['h']}  search {r['search_t']:+.2f} confirm {r['confirm_t']:+.2f} "
              f"holdout {r['holdout_t'] if r['holdout_t'] is None else round(r['holdout_t'], 2)} (need {r['threshold']:.2f})  "
              f"{'CONFIRMED' if r['confirmed'] else 'failed'}  {'HELPS' if r['helps'] else ''}")
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["holdout"]:
        if (HERE / "holdout_results.json").exists():
            raise SystemExit("The holdout was already scored once; scoring it again would make it a search period.")
        holdout()
    else:
        show(run())
