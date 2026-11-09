"""Daily research scan. Runs every weekday before the open (launchd job com.formulalab.scan).

For each stock in the tested universe it records: which tested short-term setups are active, their measured
odds since 2019, the model's 2-day odds, what analysts expect, and factual "variant" flags where the price
action and the analysts disagree. Every number comes from market data; nothing is written by an AI.
Each scan is saved to research/<buy day>.json and is scored automatically once the 2 days have passed.

Run by hand: .venv/bin/python daily_scan.py
"""
import json
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yfinance as yf
yf.set_tz_cache_location(__import__("tempfile").mkdtemp(prefix="yf-tz-"))  # own cache per process: a shared one fails when two run at once

import app
import shortterm

HERE = Path(__file__).parent
OUT = HERE / "research"
H = 0          # day trade: buy at the open, sell at the close the same day
HM = 1         # the forecast model's horizon in days (its shortest)
COST = shortterm.COST
G_MOM = 0.10   # same default momentum weight as the dashboard
OVERSOLD = {"rsi2_low", "down3", "big_drop", "close_near_low"}


def model_odds(d, k):
    """Dashboard math with default views (market at its average, drivers unchanged), 2-day horizon."""
    m, v = d["market"], d["vars"]
    mu = m["beta"] * m["avg"][HM] + (HM / 21) * G_MOM / 100 * (v.get("z_mom_12_1") or 0)
    s = v["vol_63d"] * np.sqrt(HM)
    mu_n = k * mu - COST
    return {"mu_net": mu_n, "sigma": s, "p_gain": NormalDist().cdf(mu_n / s),
            "p_short": NormalDist().cdf((-k * mu - COST) / s)}  # chance a short makes money


def variant_flags(v, active):
    """Factual disagreements worth a closer look. Analysts look ~12 months out; these are not predictions."""
    flags, px = [], v.get("price")
    rec, tgt, n = v.get("rec_mean"), v.get("target_mean"), int(v.get("n_analysts") or 0)
    up = tgt / px - 1 if tgt and px else None
    if n >= 5 and up is not None:
        if active & OVERSOLD and rec is not None and rec <= 2.5 and up > 0.10:
            flags.append(f"Price is falling while {n} analysts rate it a buy with a target {up * 100:.0f}% higher")
        if "high_52w" in active and up < 0:
            flags.append(f"At a 52-week high, already above the analysts' average target ({up * 100:+.0f}%)")
        if rec is not None and rec >= 3.5:
            flags.append(f"Analysts lean negative (rating {rec:.1f} on a 1 = strong buy to 5 = sell scale)")
    if (v.get("si_float") or 0) > 0.10:
        flags.append(f"Heavily shorted: {v['si_float'] * 100:.0f}% of tradable shares are sold short")
    return flags


def scan():
    import search_data
    ev = search_data.DATA / "events.pkl"
    if not ev.exists() or (datetime.now().timestamp() - ev.stat().st_mtime) > 3 * 86400:
        search_data.events()  # earnings dates/times for the news check; scheduled weeks ahead, so every 3 days is enough
    k = app.journal_summary()["calib"]["k"]
    picks, buy_day, asof = [], None, None
    for t in shortterm.UNIVERSE:
        try:
            st, fd = app.setups_today(t), app.forecast_data(t)
            buy_day, asof = st["next_day"], st["asof"]
            v = fd["vars"]
            active = []
            for s in st["setups"]:
                if s["active"] is True:
                    r = s["results"][H]
                    o = r["out_of_sample"]
                    active.append({"name": s["name"], "desc": s["desc"], "side": s["side"], "verified": r["verified"],
                                   # beat random days AND made money in 2010–18
                                   "worked_before": bool(r["in_sample"] and r["in_sample"]["t_edge"] >= 2 and r["in_sample"]["avg"] > 0),
                                   "win": o["win"] if o else None, "avg": o["avg"] if o else None,
                                   "worst": o["worst"] if o else None})
            names = {a["name"] for a in active}
            strong = [a for a in active if a["verified"] or a["worked_before"]]
            warnings = []
            if v.get("days_to_earn") is not None and 0 <= v["days_to_earn"] <= HM:  # past earnings (negative) don't count
                warnings.append(f"Earnings in {int(v['days_to_earn'])} trading day(s): expect a big move around them")
            best = max(strong, key=lambda a: a["win"] or 0) if strong else None
            picks.append({
                "ticker": t, "side": best["side"] if best else None, "name": v.get("name"), "sector": fd["sector_name"], "price": v.get("price"),
                "active": active, "at_open": [s["name"] for s in st["setups"] if s["active"] == "open"],
                "flagged": bool(strong), "best": best,
                "model": model_odds(fd, k),
                "analysts": {"rec": v.get("rec_mean"), "key": v.get("rec_key"), "target": v.get("target_mean"),
                             "upside": v["target_mean"] / v["price"] - 1 if v.get("target_mean") else None,
                             "n": v.get("n_analysts")},
                "variant": variant_flags(v, names), "warnings": warnings,
            })
        except Exception as e:
            picks.append({"ticker": t, "error": str(e)})
    picks.sort(key=lambda p: (not p.get("flagged"), -((p.get("best") or {}).get("win") or 0), -(p.get("model") or {}).get("p_gain", 0)))
    # Stale if the buy day has already opened: the data source is missing the latest session.
    now_et = pd.Timestamp.now(tz="America/New_York")
    opened = pd.Timestamp(f"{buy_day} 09:30", tz="America/New_York") <= now_et if buy_day else True
    # Preliminary if today's session is still running: signals use today's price so far, not its close.
    live = asof == str(now_et.date()) and market_open(now_et)
    try:
        passed = passed_signals(now_et)
    except Exception as e:
        passed = {"error": str(e)}
    out = {"scan_time": datetime.now().isoformat(timespec="seconds"), "data_asof": asof, "buy_day": buy_day, "stale": opened,
           "preliminary": live, "hold_days": H, "k": k, "universe": len(shortterm.UNIVERSE), "picks": picks, "passed": passed}
    OUT.mkdir(exist_ok=True)
    f = OUT / f"{buy_day}.json"
    old = json.loads(f.read_text()) if f.exists() else None
    if not (old and not (old.get("stale") or old.get("preliminary")) and (opened or live)):  # never replace a final scan with a worse one
        f.write_text(json.dumps(out, indent=2))
    return out


def market_open(now_et):
    """Regular US session, Mon–Fri 9:30–16:00 ET. ponytail: market holidays not handled."""
    m = now_et.hour * 60 + now_et.minute
    return now_et.weekday() < 5 and 570 <= m < 960


NEWS_CHECK = 25   # how many top day-trade candidates get the overnight news check


def overnight_news(tickers, prev_session, session):
    """{ticker: True/False/None}: did the company report earnings or file an 8-K between the previous session's close
    and this session's open? (research2.py D3: fading NO-news gaps beat the plan in 2010–22 and 2000–09.)
    None = couldn't check; the planner treats that as 'skip'."""
    import requests
    import search_data
    lo = pd.Timestamp(f"{prev_session} 16:00", tz="America/New_York")
    hi = pd.Timestamp(f"{session} 09:30", tz="America/New_York")
    events, U = search_data.load("events"), search_data.load("universe")
    cik = dict(zip(U["ticker"], U["cik"]))
    out = {}
    for t in tickers:
        ed = (events.get(t) or (None, None))[1]
        earn = bool(ed is not None and len(ed) and ((pd.DatetimeIndex(ed.index) > lo) & (pd.DatetimeIndex(ed.index) <= hi)).any())
        try:
            r = requests.get(f"https://data.sec.gov/submissions/CIK{int(cik[t]):010d}.json", timeout=10,
                             headers={"User-Agent": "FormulaLab personal research project"}).json()["filings"]["recent"]
            acc = pd.to_datetime(pd.Series(r["acceptanceDateTime"][:40]).str[:19]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
            k8 = bool(((pd.Series(r["form"][:40]) == "8-K").to_numpy() & (acc > lo).to_numpy() & (acc <= hi).to_numpy()).any())
            out[t] = earn or k8
        except Exception:
            out[t] = True if earn else None
    return out


def passed_signals(now_et, top=10):
    """Live picks for the signals that passed signal_search.py's locked 2023+ test (S&P 500, same formulas).
    gap_rel   buy at today's open, sell at today's close: stocks that opened furthest BELOW the S&P 500's own gap.
    volume    buy at today's close, sell at tomorrow's open: stocks trading the most volume vs their 20-day normal.
    day/night combined score (combo.py): the average of each promoted signal's signed z-score across stocks,
              for the day trade (yesterday's close + today's open) and the overnight trade (today so far).
    Uses the latest session in the data: today's once the market opens, otherwise the last one (live=False)."""
    import search_data
    import signal_search as ss
    combo = json.loads((HERE / "combo.json").read_text())["trades"]
    held = {r["feature"]: r for r in json.loads((HERE / "holdout_results.json").read_text())["results"] if r.get("helps")}
    tickers = list(search_data.load("universe")["ticker"])
    raw = yf.download(tickers + ["SPY"], period="3mo", auto_adjust=True, group_by="ticker", progress=False, threads=8)  # 8 at a time: fewer open sockets
    spy = raw["SPY"].dropna(subset=["Open", "Close"])
    today = spy.index[-1]
    spy_gap = float(spy["Open"].iloc[-1] / spy["Close"].iloc[-2] - 1)
    gaps = spy["Open"].shift(-1) / spy["Close"] - 1
    rows, day_f, night_f = [], {}, {}
    for t in tickers:
        df = raw[t].dropna(subset=["Open", "Close"])
        if len(df) < 25 or df.index[-1] != today:
            continue
        s20 = df["Close"].iloc[:-1].pct_change().tail(20).std()  # normal daily move, completed days (as in the test)
        v = df["Volume"].tail(20)
        f = ss.stock_features(df, gaps, spy.index)
        day_f[t], night_f[t] = f.iloc[-2], f.iloc[-1]   # day trade: yesterday's row + today's gap; overnight: today so far
        rows.append({"ticker": t, "open": float(df["Open"].iloc[-1]), "price": float(df["Close"].iloc[-1]),
                     "gap": float(df["Open"].iloc[-1] / df["Close"].iloc[-2] - 1),
                     "gap_rel": float((df["Open"].iloc[-1] / df["Close"].iloc[-2] - 1 - spy_gap) / s20),
                     "volume": float(np.log(v.iloc[-1] / v.mean()))})
    def combined(F, comps):
        F = pd.DataFrame(F).T.astype(float)
        z = [(sign * (F[c] - F[c].mean()) / F[c].std()).clip(-3, 3) for c, sign in comps.items()]
        return pd.concat(z, axis=1).mean(axis=1)
    day, night = combined(day_f, combo["h0"]["components"]), combined(night_f, combo["ovn"]["components"])
    for r in rows:
        r["day"], r["night"] = float(day.get(r["ticker"], np.nan)), float(night.get(r["ticker"], np.nan))
    rows = [r for r in rows if all(np.isfinite(r[k]) for k in ("gap_rel", "volume", "day", "night"))]
    if today == now_et.tz_localize(None).normalize() and market_open(now_et):  # live: check the best day candidates for news
        cand = [r["ticker"] for r in sorted(rows, key=lambda r: -r["day"])[:NEWS_CHECK]]
        news = overnight_news(cand, str(spy.index[-2].date()), str(today.date()))
        for r in rows:
            r["news"] = news.get(r["ticker"], None)
    if len(rows) < 450:  # a partial download must never replace a full ranking
        raise RuntimeError(f"only {len(rows)} of {len(tickers)} stocks downloaded; ranking not saved")
    live = today == now_et.tz_localize(None).normalize() and market_open(now_et)
    return {"date": str(today.date()), "live": live, "spy_gap": spy_gap, "n": len(rows), "tested": held,
            "gap_rel": sorted(rows, key=lambda r: r["gap_rel"])[:top],             # most negative = strongest buy
            "volume": sorted(rows, key=lambda r: -r["volume"])[:top],
            "day": sorted(rows, key=lambda r: -r["day"])[:top], "night": sorted(rows, key=lambda r: -r["night"])[:top],
            "combo": {k: {"use": v["use_combo"], "components": v["components"], "result": {x: v["2023-now"]["combo"][x] for x in
                      ("top10_net", "top10_win", "top10_bad_day", "fifth_excess", "fifth_t")},
                      "fifths": v["fifths_2023"], "all": v["all_2023"]} for k, v in combo.items()},
            "all": rows}


def summarize(xs):
    if not xs:
        return None
    a = np.array(xs)
    return {"n": len(a), "win": float((a > 0).mean()), "avg": float(a.mean())}


def track_record():
    """Score every past scan: buy at the open on the buy day, sell at the close H trading days later."""
    rows = []
    for f in sorted(OUT.glob("*.json")):
        s = json.loads(f.read_text())
        if s.get("stale") or s.get("preliminary"):
            continue  # never scored: missing the latest session, or made before the close
        rows += [{"buy_day": s["buy_day"], "ticker": p["ticker"], "flagged": p["flagged"], "side": p.get("side") or "long",
                  "hold": s.get("hold_days", 2)}  # each scan is scored by the hold it was made for
                 for p in s["picks"] if "error" not in p]
    if not rows:
        return {"flagged": None, "all": None, "pending": 0, "trades": []}
    tickers = sorted({r["ticker"] for r in rows})
    start = pd.Timestamp(min(r["buy_day"] for r in rows)) - pd.Timedelta(days=3)
    px = yf.download(tickers, start=start, auto_adjust=True, group_by="ticker", progress=False)
    for r in rows:
        df = px[r["ticker"]].dropna(subset=["Open", "Close"])
        i = df.index.searchsorted(pd.Timestamp(r["buy_day"]))  # first trading day on/after the buy day
        if i + r["hold"] < len(df):
            gross = float(df["Close"].iloc[i + r["hold"]] / df["Open"].iloc[i] - 1)
            r["ret"] = (gross if r["side"] == "long" else -gross) - COST
            r["long_ret"] = gross - COST
    done = [r for r in rows if "ret" in r]
    return {"flagged": summarize([r["ret"] for r in done if r["flagged"]]), "all": summarize([r["long_ret"] for r in done]),
            "pending": len(rows) - len(done), "trades": [r for r in rows if r["flagged"]]}


def refresh_picks():
    """Fast path (~15–25 s): recompute only the two tested rankings and put them into the latest scan file."""
    now_et = pd.Timestamp.now(tz="America/New_York")
    files = sorted(OUT.glob("*.json"))
    if not files:
        return scan()
    f = files[-1]
    s = json.loads(f.read_text())
    s["passed"] = passed_signals(now_et)
    s["picks_time"] = datetime.now().isoformat(timespec="seconds")
    f.write_text(json.dumps(s, indent=2))
    if s["passed"].get("live"):  # keep exactly what the plan showed at this minute (sim.py follows these)
        snap = OUT / "snapshots"
        snap.mkdir(exist_ok=True)
        (snap / f"{s['passed']['date']}-{now_et.strftime('%H%M')}.json").write_text(json.dumps(s["passed"]))
    return s


if __name__ == "__main__":
    import sys
    if "--picks" in sys.argv:
        refresh_picks()
        sys.exit(0)
    now = pd.Timestamp.now(tz="America/New_York")
    # launchd runs this every 30 minutes; outside 9:00–16:30 ET on weekdays there is nothing new to fetch.
    if "--trading-hours" in sys.argv and not (now.weekday() < 5 and 540 <= now.hour * 60 + now.minute <= 990):
        sys.exit(0)
    o = scan()
    flagged = [p["ticker"] for p in o["picks"] if p.get("flagged")]
    errors = [p["ticker"] for p in o["picks"] if "error" in p]
    print(f"{o['scan_time']} scan for buy day {o['buy_day']}: {len(flagged)} flagged {flagged}, {len(errors)} errors {errors}")
