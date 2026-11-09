"""Simulated trades: what following the plan would have done, at the times broker.py trades, at real prices.

For a session D (a weekday the market was open):
  1. sell last session's overnight stocks at D's open (first 5-minute bar's open, ~9:30–9:31 ET)
  2. day leg: buy the plan's 5 day picks at 9:35 ET, sell at the close (last 5-minute bar, ~15:59)
  3. night leg: buy the plan's 5 overnight picks at the close (~15:59); they are sold at the next session's open
Picks come from the ranking the Plan page showed when that leg could be followed: the latest snapshot saved by
the pick refresh before 9:34 ET (day) / 15:57 ET (night). If none was saved (Mac asleep), the after-close ranking
is used and the trade says so. $10 each at a $100 budget. Prices are Yahoo 5-minute
trade prices with 0.025% added on each buy and taken off each sale (a 0.05% round trip, like the tests).
Written to journal.json as source "sim", so the Portfolio page tracks it. Safe to re-run: a step already
simulated for that session and stock is skipped.

Run: .venv/bin/python sim.py run [YYYY-MM-DD]   (default: the latest session in the ranking; launchd runs it after the close)
"""
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
yf.set_tz_cache_location(__import__("tempfile").mkdtemp(prefix="yf-tz-"))  # own cache per process

import app
import planner

HERE = Path(__file__).parent
HALF_COST = 0.00025
ET = "America/New_York"


def bars(tickers, day):
    """Yahoo 5-minute bars for `day` (ET), per ticker."""
    raw = yf.download(sorted(set(tickers)), period="7d", interval="5m", group_by="ticker", progress=False, threads=8)
    out = {}
    for t in set(tickers):
        try:
            b = raw[t].dropna(subset=["Open", "Close"])
        except KeyError:
            continue
        b.index = b.index.tz_convert(ET)
        b = b[b.index.strftime("%Y-%m-%d") == day]
        if len(b):
            out[t] = b
    return out


def price_at(b, kind):
    """open = first bar's open (the broker sells at open+1 min); 935 = close of the 9:30 bar (buys at 9:35);
    close = close of the last bar (the broker trades at close−1 min)."""
    if kind == "open":
        return float(b["Open"].iloc[0])
    if kind == "935":
        return float(b["Close"].iloc[0])
    return float(b["Close"].iloc[-1])


def entry(ticker, action, shares, price, day, hhmm, leg, review_date, fc):
    return {"ticker": ticker, "action": action, "shares": round(shares, 6), "price": round(price, 4), "source": "sim", "step": leg,
            "horizon": 1, "sector": "", "review_date": review_date, "forecast": fc, "id": uuid.uuid4().hex[:8],
            "date": day, "created": f"{day}T{hhmm}:00", "note": "simulated: plan followed at broker times, 5-minute prices, 0.05% costs"}


def ranking_for(day, cutoff, fallback):
    """The latest pick snapshot for `day` taken at or before `cutoff` (HHMM, ET), else the after-close ranking."""
    snaps = sorted(f for f in (HERE / "research" / "snapshots").glob(f"{day}-*.json") if f.stem[-4:] <= cutoff)
    if snaps:
        return json.loads(snaps[-1].read_text()), f"ranking shown at {snaps[-1].stem[-4:-2]}:{snaps[-1].stem[-2:]} ET"
    return fallback, "after-close ranking (no snapshot from the follow window)"


def run(day=None):
    scan = app.latest_scan()
    P = scan.get("passed") or {}
    day = day or P.get("date")
    if P.get("date") != day:
        raise SystemExit(f"No ranking for {day} (latest ranking is for {P.get('date')}).")
    src = {}
    plan = {"trades": []}
    for leg, cutoff in (("day", "0934"), ("night", "1557")):
        R, src[leg] = ranking_for(day, cutoff, P)
        plan["trades"] += [t for t in planner.build({"passed": dict(R, live=True)})["trades"] if t["leg"] == leg]
    budget = 100.0
    entries = app.read_journal()
    done = {(e.get("step"), e["action"], e["date"], e["ticker"]) for e in entries if e.get("source") == "sim"}
    held = {}
    for e in sorted((e for e in entries if e.get("source") == "sim"), key=lambda e: e["created"]):
        held[e["ticker"]] = held.get(e["ticker"], 0) + (e["shares"] if e["action"] == "BUY" else -e["shares"])
    overnight = [e for e in entries if e.get("source") == "sim" and e["step"] == "night" and e["action"] == "BUY"
                 and e["date"] < day and held.get(e["ticker"], 0) > 1e-9]
    longs = {leg: [t for t in plan["trades"] if t["leg"] == leg and t["side"] == "long"] for leg in ("day", "night")}
    B = bars([e["ticker"] for e in overnight] + [t["symbol"] for t in longs["day"] + longs["night"]], day)
    if not B:
        raise SystemExit(f"No 5-minute prices for {day} yet.")
    combo = json.loads((HERE / "combo.json").read_text())["trades"]
    fc = lambda trade: {"mu_raw": combo[trade]["2023-now"]["combo"]["top5"]["avg"], "mu": combo[trade]["2023-now"]["combo"]["top5"]["avg"],
                        "mu_net": combo[trade]["2023-now"]["combo"]["top5"]["avg"] - 2 * HALF_COST, "sigma": 0.0}
    nxt = str((pd.Timestamp(day) + pd.offsets.BDay(1)).date())
    new, log = [], []
    # 1. last session's overnight stocks: sell at this session's open
    for e in overnight:
        if e["ticker"] in B and ("night", "SELL", day, e["ticker"]) not in done:
            px = price_at(B[e["ticker"]], "open") * (1 - HALF_COST)
            new.append(entry(e["ticker"], "SELL", held[e["ticker"]], px, day, "09:31", "night", day, e["forecast"]))
            log.append(f"sold {e['ticker']} (overnight) at ${px:.2f}")
    # 2. day leg: 9:35 -> close;  3. night leg: buy at the close
    for leg, buy_at, trade in (("day", "935", "h0"), ("night", "close", "ovn")):
        for t in longs[leg]:
            if (leg, "BUY", day, t["symbol"]) in done:
                continue
            b = B.get(t["symbol"])
            if b is None:
                log.append(f"skipped {t['symbol']}: no 5-minute prices"); continue
            usd = budget * t["weight"]
            px_in = price_at(b, buy_at) * (1 + HALF_COST)
            sh = usd / px_in
            new.append(entry(t["symbol"], "BUY", sh, px_in, day, "09:35" if leg == "day" else "15:59", leg, day if leg == "day" else nxt, fc(trade)) | {"ranking": src[leg]})
            if leg == "day":
                px_out = price_at(b, "close") * (1 - HALF_COST)
                new.append(entry(t["symbol"], "SELL", sh, px_out, day, "15:59", leg, day, fc(trade)))
                log.append(f"day {t['symbol']}: bought ${px_in:.2f}, sold ${px_out:.2f} ({(px_out / px_in - 1) * 100:+.2f}%, {usd * (px_out / px_in - 1):+.2f} $)")
            else:
                log.append(f"night {t['symbol']}: bought ${px_in:.2f} at the close, sells at the next open")
    if new:
        app.JOURNAL.write_text(json.dumps(entries + new, indent=2))
        app._jcache.clear()
    return log


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        for line in run(sys.argv[2] if len(sys.argv) > 2 else None):
            print(line)
