"""Checks the price-variable math on made-up prices. Run: .venv/bin/python test_vars.py"""
import numpy as np
import pandas as pd

from app import price_vars

n = 600
idx = pd.bdate_range("2022-01-03", periods=n)
close = pd.Series(100 * 1.001 ** np.arange(n), idx)          # steady +0.1%/day
spy = pd.Series(100 * (1 + 0.01 * np.sin(np.arange(n))), idx)  # wiggles

v = price_vars(close, pd.Series(1000.0, idx), spy)
assert abs(v["rev_5d"] - (1.001 ** 5 - 1)) < 1e-12
assert abs(v["mom_12_1"] - (1.001 ** 231 - 1)) < 1e-12           # 252 days ago -> 21 days ago
assert v["trend_200"] > 0 and v["vol_63d"] < 1e-12
assert abs(v["adv_usd"] - close.tail(20).mean() * 1000) < 1e-6
assert abs(v["beta_mkt"]) < 1e-9                                  # constant returns don't move with SPY

double = 100 * (1 + 2 * spy.pct_change().fillna(0)).cumprod()      # moves exactly 2x SPY
assert abs(price_vars(double, pd.Series(1.0, idx), spy)["beta_mkt"] - 2) < 1e-9
print("price vars ok")

# Regression recovers known sensitivities: y = 0.001 + 1.2*x1 - 0.3*x2 + tiny noise
from app import ols
rng = np.random.default_rng(0)
X = rng.normal(size=(300, 2))
b, t, r2, _ = ols(0.001 + X @ [1.2, -0.3] + rng.normal(scale=1e-4, size=300), X)
assert np.allclose(b, [0.001, 1.2, -0.3], atol=1e-4) and r2 > 0.999 and abs(t[1]) > 100
print("ols ok")

# Positions: buy 10 @100, buy 10 @120 (avg 110), sell 5 @130 -> realized 5*20 = 100, 15 left
from app import positions, calibrate
E = [{"ticker": "X", "action": a, "shares": n, "price": p, "created": f"2026-01-0{i}"}
     for i, (a, n, p) in enumerate([("BUY", 10, 100), ("BUY", 10, 120), ("HOLD", 0, 125), ("SELL", 5, 130)], 1)]
(pos,) = positions(E, {"X": 140})
assert pos["shares"] == 15 and abs(pos["avg_cost"] - 110) < 1e-9 and abs(pos["realized"] - 100) < 1e-9
assert abs(pos["unrealized"] - 15 * 30) < 1e-9

# Learning: no results -> k = 1. Forecasts exactly 2x too big -> best fit 0.5, k pulled toward 1 by the prior.
assert calibrate([])["k"] == 1.0
c = calibrate([(0.02, 0.01, 0.05), (-0.04, -0.02, 0.05)] * 10)
assert abs(c["k_raw"] - 0.5) < 1e-12 and abs(c["k"] - 8 / 12) < 1e-12 and c["hit"] == 1.0
assert calibrate([(0.01, -0.05, 0.05)] * 100)["k"] == 0.0         # consistently wrong -> forecasts switched off
assert abs(calibrate([(0.02, 0.07, 0.07)])["k"] - 1.05) < 0.01     # one typical result barely moves k
# Short: short 10 @50, cover 4 @40 -> realized 4*10 = 40, 6 still short, price 45 -> unrealized 6*5 = 30
S = [{"ticker": "Y", "action": a, "shares": n, "price": p, "created": f"2026-02-0{i}"}
     for i, (a, n, p) in enumerate([("SHORT", 10, 50), ("COVER", 4, 40)], 1)]
(sp,) = positions(S, {"Y": 45})
assert sp["shares"] == -6 and abs(sp["realized"] - 40) < 1e-9 and abs(sp["unrealized"] - 30) < 1e-9
print("journal ok")

# daily_model.features must not peek ahead: adding future days can't change past rows
import daily_model
idx = pd.bdate_range("2023-01-02", periods=400)
rng = np.random.default_rng(1)
c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
df = pd.DataFrame({"Open": c.shift().bfill(), "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": rng.integers(1e6, 2e6, 400).astype(float)})
spy, vix = c * 4, pd.Series(20 + rng.normal(0, 1, 400), index=idx)
full, part = daily_model.features(df, spy, vix), daily_model.features(df.iloc[:-10], spy.iloc[:-10], vix.iloc[:-10])
pd.testing.assert_frame_equal(full.iloc[:-11], part.iloc[:-1])
print("daily model ok")

# qualitative.q_features timing: news counts only from the first open after it was public
import qualitative
idx = pd.bdate_range("2026-03-02", periods=10)  # Mon Mar 2 .. Fri Mar 13
ny = lambda s: pd.Timestamp(s, tz="America/New_York")
ed = pd.DataFrame({"Surprise(%)": [5.0, -8.0]}, index=pd.DatetimeIndex([ny("2026-03-03 16:00"), ny("2026-03-10 12:00")]))
ud = pd.DataFrame({"Action": ["up", "down", "up"], "priceTargetAction": ["Raises", "Lowers", "Raises"]},
                  index=pd.DatetimeIndex(["2026-03-04 10:00", "2026-03-07 09:00", "2026-03-09 08:00"]))  # Wed, Sat, Mon
q = qualitative.q_features(idx, ud, ed)
d = lambda s: q.index.get_loc(pd.Timestamp(s))
assert q["earn_new"].iloc[d("2026-03-03")] == 5 and q["earn_new"].sum() == 5 - 8   # Tue after-close -> trade Wed
assert q["earn_new"].iloc[d("2026-03-10")] == -8                                   # Tue midday -> known Wed open
assert q["earn_recent"].iloc[d("2026-03-03")] == 0 and q["earn_recent"].iloc[d("2026-03-04")] == 5
assert q["earn_ahead"].iloc[d("2026-03-02")] == 1 and q["earn_ahead"].iloc[d("2026-03-03")] == 0
assert q["analyst_net_1d"].iloc[d("2026-03-04")] == 1 and q["analyst_net_1d"].iloc[d("2026-03-09")] == 0  # Sat down + Mon up
assert q["analyst_net_1d"].iloc[d("2026-03-06")] == 0 and q["analyst_net_20d"].iloc[-1] == 1
q2 = qualitative.q_features(idx, None, pd.DataFrame({"Surprise(%)": [np.nan]}, index=pd.DatetimeIndex([ny("2026-04-30 16:00")])))
assert q2["earn_ahead"].sum() == 0  # a report weeks past the data must not land on the last row
q3 = qualitative.q_features(idx, None, pd.DataFrame({"Surprise(%)": [np.nan]}, index=pd.DatetimeIndex([ny("2026-03-16 16:00")])))
assert q3["earn_ahead"].iloc[-1] == 1  # last row trades Mon Mar 16; report after that close lands in a 2-day hold
print("qualitative timing ok")

# signal_search timing: SEC filings count from the first open after acceptance; past rows never see the future
import signal_search as ss
cal = pd.bdate_range("2026-03-02", periods=10)  # Mon Mar 2 .. Fri Mar 13
f = pd.DataFrame({"form": ["8-K", "8-K", "8-K", "4"], "items": ["2.02,9.01", "8.01", "1.01", ""],
                  "acceptanceDateTime": ["2026-03-03T22:00:00.000Z", "2026-03-04T15:00:00.000Z", "2026-03-07T17:00:00.000Z", "2026-03-09T13:30:00.000Z"]})  # UTC
sf = ss.sec_features(f, cal)
r = lambda s: sf.index.get_loc(pd.Timestamp(s))
assert sf["sec_8k_overnight"].iloc[r("2026-03-03")] == 1 and sf["sec_202_overnight"].iloc[r("2026-03-03")] == 1  # Tue 5 PM -> trade Wed
assert sf["sec_8k_day"].iloc[r("2026-03-04")] == 1 and sf["sec_801_overnight"].sum() == 0                     # Wed 10 AM -> in Wed's close
assert sf["sec_101_overnight"].iloc[r("2026-03-06")] == 1                                                      # Saturday -> trade Monday
assert sf["sec_form4_recent5"].iloc[r("2026-03-06")] == 0 and sf["sec_form4_recent5"].iloc[r("2026-03-09")] > 0  # Mon 9:30 -> Mon session
idx = pd.bdate_range("2023-01-02", periods=400)
c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
df = pd.DataFrame({"Open": c.shift().bfill() * 1.001, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": rng.integers(1e6, 2e6, 400).astype(float)})
gap = pd.Series(0.0, index=idx)
full, part = ss.stock_features(df, gap, idx), ss.stock_features(df.iloc[:-10], gap, idx[:-10])
keep = [x for x in full.columns if x not in ("gap_D", "gap_D_rel") and not x.startswith("y")]  # y* = future returns by design
pd.testing.assert_frame_equal(full[keep].iloc[:-11], part[keep].iloc[:-1])
print("signal search timing ok")

# bigmove.earn_next: a report after Tue's close or before Wed's close flags Tue (the Tue-close -> Wed-close move)
import bigmove
idx = pd.bdate_range("2026-03-02", periods=5)  # Mon..Fri
ny = lambda s: pd.Timestamp(s, tz="America/New_York")
e = bigmove.earn_next(idx, pd.DataFrame(index=pd.DatetimeIndex([ny("2026-03-03 16:00"), ny("2026-03-05 07:00"), ny("2026-03-09 16:05")])))
assert list(e) == [0, 1, 1, 0, 0], e  # Tue after close -> Tue row; Thu 7 AM -> Wed row; next Mon -> outside
print("big move timing ok")

# plan + health safety rules (made-up scan, fake broker, no network)
import json, tempfile, time as _time
from pathlib import Path
import app as A, health as HL, planner, broker as BK
_day = "2026-10-16"
_rows = [{"ticker": f"S{i:03d}", "price": 50.0, "gap": -0.02, "gap_rel": -1.0, "volume": 0.5, "day": float(i), "night": float(-i),
          "news": False} for i in range(60)]
_rows.append({"ticker": "NEWS", "price": 50.0, "gap": -0.05, "gap_rel": -3.0, "volume": 0.5, "day": 99.0, "night": -99.0, "news": True})
scan = {"scan_time": "2026-10-16T09:31:00", "passed": {"date": _day, "live": True, "n": 60, "all": _rows}}
plan = planner.build(scan)
longs = [t for t in plan["trades"] if t["side"] == "long"]
assert len(longs) == 10 and abs(sum(t["weight"] for t in longs) - 1.0) < 1e-9          # day leg paused: 10 overnight x 10%
assert {t["leg"] for t in longs} == {"night"} and {t["symbol"] for t in longs} == {f"S{i:03d}" for i in range(10)}
assert len(plan["all"]) == 61 and sum(1 for t in plan["trades"] if t["side"] == "short") == 5
_paused, planner.PAUSED = planner.PAUSED, {}                                             # the day rule still works if resumed
_p2 = planner.build(scan)
assert {t["symbol"] for t in _p2["trades"] if t["leg"] == "day" and t["side"] == "long"} == {f"S{i:03d}" for i in range(55, 60)}
assert _p2["skipped_news"] == [{"symbol": "NEWS", "why": "news overnight"}]              # top day score, but news
planner.PAUSED = _paused
orders, skipped = BK.size(plan["trades"], 100)
assert len(orders) == 10 and all(o["usd"] == 10.0 for o in orders) and all("margin" in k["reason"] for k in skipped)
assert planner.build({"passed": dict(scan["passed"], live=False)})["trades"] == []        # no live session, no trades
A.RESEARCH, A.HEALTH_FILE = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "health.json"
(A.RESEARCH / "2026-10-19.json").write_text(json.dumps(scan))
planner.HERE = Path(tempfile.mkdtemp())                                                  # plan files go to a temp dir
sent = []
BK.follow = lambda p, now=None: sent.append(p) or {"queued": p["trades"], "skipped": []}
BK.status = lambda: {"orders": []}
at = lambda hm: pd.Timestamp(f"{_day} {hm}", tz="America/New_York")
A._health.update(t=_time.time() + 1e9, r={"ok": False, "excluded": {}, "time": "x", "checks": [{"name": "x", "blocks": True, "detail": "test block", "ok": False, "fixing": None}]})
for hm, leg, msg in (("09:32", "day", "can't be followed"), ("15:52", "night", "Blocked"), ("12:00", "night", "can't be followed")):
    try:
        A.follow_leg(leg, now=at(hm)); raise AssertionError(f"{leg} at {hm} was not refused")
    except ValueError as err:
        assert msg in str(err), err
A._health.update(r={"ok": True, "excluded": {}, "time": "x", "checks": []})
A.follow_leg("night", now=at("15:52"))
assert len(sent) == 1 and {t["leg"] for t in sent[0]["trades"]} == {"night"} and sent[0]["date"] == _day
A._health.update(t=0, r=None)
assert HL.bad_picks({"passed": {"day": [{"ticker": "X", "price": 0, "gap": 0}], "night": [{"ticker": "Y", "price": 5, "gap": 0.4, "volume": 0}]}}) \
    == {"X": "missing or zero price", "Y": "opened +40% (possible split or bad data)"}
old = {"scan_time": "2026-10-16T08:00:00", "stale": True, "picks": [], "passed": {"n": 300, "date": "2026-10-15"}}
assert len(HL.scan_issues(old, pd.Timestamp("2026-10-16 11:00", tz="America/New_York"))) == 3  # stale, incomplete, not today
ok = {"scan_time": "2026-10-16T09:00:00", "picks_time": "2026-10-16T15:30:00", "picks": [], "passed": {"n": 500, "date": "2026-10-16"}}
assert HL.scan_issues(ok, pd.Timestamp("2026-10-16 15:35", tz="America/New_York")) == []
assert len(HL.scan_issues(ok, pd.Timestamp("2026-10-16 15:52", tz="America/New_York"))) == 1  # 22 min old in a buy window
print("plan safety ok")

# sim.price_at: which 5-minute price stands for each broker time
import sim
_b = pd.DataFrame({"Open": [10.0, 11.0, 12.0], "Close": [10.5, 11.5, 12.5]},
                  index=pd.DatetimeIndex(["2026-10-16 09:30", "2026-10-16 12:00", "2026-10-16 15:55"]).tz_localize("America/New_York"))
assert sim.price_at(_b, "open") == 10.0 and sim.price_at(_b, "935") == 10.5 and sim.price_at(_b, "close") == 12.5
print("sim ok")

# sim.ranking_for: the snapshot the Plan page showed before each follow window, else the after-close ranking
_snap = Path(tempfile.mkdtemp())
(_snap / "research" / "snapshots").mkdir(parents=True)
for hhmm, tag in (("0931", "a"), ("0933", "b"), ("1555", "c"), ("1600", "d")):
    (_snap / "research" / "snapshots" / f"2026-10-16-{hhmm}.json").write_text(json.dumps({"tag": tag}))
sim.HERE = _snap
assert sim.ranking_for("2026-10-16", "0934", {"tag": "x"})[0]["tag"] == "b"
assert sim.ranking_for("2026-10-16", "1557", {"tag": "x"})[0]["tag"] == "c"
assert sim.ranking_for("2026-10-19", "0934", {"tag": "x"})[0]["tag"] == "x"   # no snapshot -> fallback
print("sim snapshots ok")

