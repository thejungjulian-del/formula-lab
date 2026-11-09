"""Checks broker.py against a fake Alpaca: sizing, scheduling, entry window, exits. Run: .venv/bin/python test_broker.py"""
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

import broker

calls, clock = [], {"is_open": True}


def fake_api(method, path, **kw):
    calls.append((method, path, kw.get("json")))
    if path == "/v2/calendar":
        days = [{"date": "2026-10-19", "open": "09:30", "close": "16:00"},
                {"date": "2026-10-20", "open": "09:30", "close": "13:00"},  # half day
                {"date": "2026-10-21", "open": "09:30", "close": "16:00"}]
        return [d for d in days if d["date"] >= kw["params"]["start"]]
    if path.startswith("/v2/assets/"):
        return {"tradable": True, "fractionable": not path.endswith("BRKA")}
    if path == "/v2/orders:by_client_order_id":
        raise RuntimeError("Alpaca 404: not found")
    if path == "/v2/orders":
        return {"id": f"ord{len(calls)}"}
    if path.startswith("/v2/orders/"):
        return {"filled_qty": "0.25", "filled_avg_price": "200"}
    if path == "/v2/clock":
        return clock


def et(day, hm):
    return datetime.strptime(f"2026-10-{day} {hm}", "%Y-%m-%d %H:%M").replace(tzinfo=broker.ET)


broker.api = fake_api
tmp = Path(tempfile.mkdtemp())
broker.QUEUE, broker.HEALTH = tmp / "orders.json", tmp / "health.json"
broker.env = lambda: {"BUDGET": "100"}


def run(now, ok=True, age_min=1, reasons=()):
    """run_due with the server's health file written `age_min` minutes before `now`."""
    broker.HEALTH.write_text(json.dumps({"ok": ok, "time": (now - timedelta(minutes=age_min)).isoformat(), "reasons": list(reasons)}))
    return broker.run_due(now)


def follow(day, *trades):
    return broker.follow({"date": f"2026-10-{day}", "trades": [dict(side="long", **t) for t in trades]}, now=et(day, "08:00"))


# sizing: shorts and sub-$1 rows skipped, weights over 100% scaled down
orders, skipped = broker.size([{"symbol": "A", "side": "long", "weight": 0.6}, {"symbol": "B", "side": "long", "weight": 0.6},
                               {"symbol": "C", "side": "short", "weight": 0.2}, {"symbol": "D", "side": "long", "weight": 0.001}], 100)
assert [o["usd"] for o in orders] == [49.96, 49.96], orders  # 0.6/1.201 of $100
assert {s["symbol"] for s in skipped} == {"C", "D"}
leg = [{"symbol": s, "side": "long", "weight": 0.1} for s in "VWXYZ"]  # planner leg = 5 × 10% of the budget
assert sum(o["usd"] for o in broker.size(leg, 100, free=50)[0]) == 50.0   # other leg holding $50: still full size
assert sum(o["usd"] for o in broker.size(leg, 100, free=20)[0]) == 20.0   # only $20 free: shrinks to fit

plan = {"date": "2026-10-19", "trades": [
    {"symbol": "AAPL", "side": "long", "weight": 0.4, "enter": "open", "enter_delay_min": 5, "hold_days": 0, "exit": "close"},
    {"symbol": "MSFT", "side": "long", "weight": 0.4, "enter": "close", "hold_days": 1, "exit": "close"},
    {"symbol": "NVDA", "side": "long", "weight": 0.1, "enter": "open", "hold_days": 0, "exit": "close"},
    {"symbol": "BRKA", "side": "long", "weight": 0.1, "enter": "open", "hold_days": 0, "exit": "close"},
    {"symbol": "TSLA", "side": "short", "weight": 0.2, "enter": "open", "hold_days": 0, "exit": "close"}]}
r = broker.follow(plan, now=et(19, "08:00"))
assert [x["symbol"] for x in r["queued"]] == ["AAPL", "MSFT", "NVDA"], r
assert {s["symbol"] for s in r["skipped"]} == {"BRKA", "TSLA"}
q = {x["symbol"]: x for x in r["queued"]}
assert q["AAPL"]["enter_at"] == et(19, "09:35").isoformat() and q["NVDA"]["enter_at"] == et(19, "09:31").isoformat()
assert q["AAPL"]["exit_at"] == et(19, "15:59").isoformat() and q["MSFT"]["enter_at"] == et(19, "15:59").isoformat()
assert q["MSFT"]["enter_by"] == (et(19, "16:00") - timedelta(seconds=10)).isoformat()  # never sent after the bell
assert q["MSFT"]["exit_at"] == et(20, "12:59").isoformat()  # half-day close is respected

# a second plan can only use what is left of the $100 ($40 + $40 + $10 committed, $10 free)
r2 = follow(19, {"symbol": "AMD", "weight": 1.0, "enter": "close", "hold_days": 1, "exit": "open"})
assert r2["queued"][0]["usd"] == 10.0, r2

assert run(et(19, "09:30")) == []                                 # not yet
assert run(et(19, "09:32")) == ["NVDA -> open"]                   # default 1-minute delay
assert run(et(19, "09:36")) == ["AAPL -> open"]                   # 5-minute delay
buys = [c[2] for c in calls if c[:2] == ("POST", "/v2/orders")]
assert [(b["symbol"], b["notional"], b["side"]) for b in buys] == [("NVDA", "10.00", "buy"), ("AAPL", "40.00", "buy")], buys

run(et(19, "15:59"))                                              # AAPL/NVDA sell 0.25 sh, MSFT/AMD buy
sells = [c[2] for c in calls if c[:2] == ("POST", "/v2/orders") and c[2]["side"] == "sell"]
assert [(s["symbol"], s["qty"]) for s in sells] == [("AAPL", "0.25"), ("NVDA", "0.25")], sells

clock["is_open"] = False                                          # Mac slept through both exits
assert run(et(20, "18:00")) == []                                 # market closed: AMD and MSFT wait
clock["is_open"] = True
assert sorted(run(et(21, "09:40"))) == ["AMD -> closed", "MSFT -> closed"]  # next open: late exits go out

# entries found late are skipped, never placed late or after the bell
follow(19, {"symbol": "KO", "weight": 0.05, "enter": "open", "hold_days": 0, "exit": "close"})
assert run(et(19, "09:45")) == ["KO -> missed"]
follow(20, {"symbol": "T", "weight": 0.05, "enter": "close", "hold_days": 1, "exit": "open"})
assert run(et(20, "13:00")) == ["T -> missed"]                    # half-day bell at 13:00

# health gate blocks entries, never exits
follow(19, {"symbol": "PEP", "weight": 0.05, "enter": "open", "hold_days": 0, "exit": "close"},
       {"symbol": "XOM", "weight": 0.05, "enter": "open", "enter_delay_min": 2, "hold_days": 0, "exit": "close"},
       {"symbol": "CVX", "weight": 0.05, "enter": "close", "hold_days": 1, "exit": "open"})
assert run(et(19, "09:31"), age_min=6) == ["PEP -> blocked"]      # stale file
assert run(et(19, "09:32")) == ["XOM -> open"]
assert sorted(run(et(19, "15:59"), ok=False, reasons=["prices stale"])) == ["CVX -> blocked", "XOM -> closed"]
with broker.queue() as qq:
    why = {x["symbol"]: x.get("reason") for x in qq}
assert why["PEP"].startswith("health file is 6 min old") and why["CVX"] == "prices stale", why
broker.HEALTH.unlink()
follow(19, {"symbol": "IBM", "weight": 0.05, "enter": "open", "hold_days": 0, "exit": "close"})
assert broker.run_due(et(19, "09:31")) == ["IBM -> blocked"]      # no health file at all
print("broker ok")
