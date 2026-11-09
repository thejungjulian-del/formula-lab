"""Sends a followed plan to Alpaca at the right times. Practice (paper) account unless .env says otherwise.

  follow(plan)  size the plan for the budget, schedule each entry and exit, save to orders.json
  run_due()     place every order whose time has come (launchd runs this every minute)
  status()      the queue, plus account cash when keys are set

Plan format (written by the plan generator; one row per trade):
  {"date": "2026-10-19",                                   # first session the plan trades
   "trades": [{"symbol": "AAPL", "side": "long", "weight": 0.10,      # weight = share of the budget
               "enter": "open", "hold_days": 0, "exit": "close", "why": "..."}]}
  enter/exit are "open" or "close"; hold_days = sessions from entry day to exit day
  (buy open, sell close same day = 0; buy close, sell next open = 1; 5-day hold = 5).
  Optional "enter_delay_min" (default 1): minutes after the open an "open" entry goes out (picks that need the
  opening gap only exist after 9:30). "close" entries and exits go out 1 minute before the close.

Entries are skipped (never placed late, never after the close) when found past their window, and blocked when
data/health.json (written by the server) says ok=false or is over 5 minutes old. Exits always go out.

Realistic $100 rules (Alpaca docs, checked 2026-10-17): fractional orders are market + day only, in regular
hours; no fractional short sales; $1 minimum; shorting needs a $2,000 margin account -> short rows are skipped.
Paper fills ignore slippage and fees, so paper results flatter real ones a little.

.env:  ALPACA_KEY_ID=...  ALPACA_SECRET_KEY=...  ALPACA_PAPER=true (false = real money)  BUDGET=100
Run:   .venv/bin/python broker.py check | status | run | follow plan.json
"""
import fcntl
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).parent
QUEUE = HERE / "orders.json"
HEALTH = HERE / "data" / "health.json"
ET = ZoneInfo("America/New_York")
BEFORE_CLOSE = timedelta(minutes=1)    # measured: last-minute buys kept ~77% of the overnight edge, 15:55 ~39%
CLOSE_CUTOFF = timedelta(seconds=10)   # no entry is sent later than this before the bell
WINDOW = timedelta(minutes=10)         # an entry this late (Mac asleep) is skipped, never placed late
HEALTH_MAX_AGE = timedelta(minutes=5)
MIN_ORDER = 1.0


def env():
    e = dict(os.environ)
    p = HERE / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                e.setdefault(k.strip(), v.strip())
    return e


def paper():
    return env().get("ALPACA_PAPER", "true").lower() != "false"


def api(method, path, **kw):
    e = env()
    if not e.get("ALPACA_KEY_ID") or not e.get("ALPACA_SECRET_KEY"):
        raise RuntimeError("No Alpaca keys: add ALPACA_KEY_ID and ALPACA_SECRET_KEY to ~/formula-lab/.env")
    base = "https://paper-api.alpaca.markets" if paper() else "https://api.alpaca.markets"
    try:
        r = requests.request(method, base + path, timeout=15, **kw, headers={
            "APCA-API-KEY-ID": e["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": e["ALPACA_SECRET_KEY"]})
    except requests.RequestException as err:
        raise RuntimeError(f"network: {err}") from err
    if not r.ok:
        raise RuntimeError(f"Alpaca {r.status_code}: {r.text[:300]}")
    return r.json()


def place(body):
    """Idempotent order: if this client_order_id already reached Alpaca (say a timeout after it was accepted),
    return that order instead of sending a second one."""
    try:
        return api("GET", "/v2/orders:by_client_order_id", params={"client_order_id": body["client_order_id"]})
    except RuntimeError:
        return api("POST", "/v2/orders", json=body)


@contextmanager
def queue():
    """orders.json under a file lock, so the web server's follow() and launchd's run_due() never clobber each other."""
    with open(QUEUE.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        q = json.loads(QUEUE.read_text()) if QUEUE.exists() else []
        yield q
        QUEUE.write_text(json.dumps(q, indent=1))


def sessions(start, days=45):
    """Trading sessions from `start`: [(open, close)] as ET datetimes (skips holidays, knows half days)."""
    out = []
    for d in api("GET", "/v2/calendar", params={"start": start.isoformat(), "end": (start + timedelta(days)).isoformat()}):
        day = date.fromisoformat(d["date"])
        t = lambda s: datetime.combine(day, datetime.strptime(s.replace(":", ""), "%H%M").time(), ET)
        out.append((t(d["open"]), t(d["close"])))
    return out


def size(trades, budget, free=None):
    """Dollars per trade for a long-only cash account -> (orders, skipped). weight = share of the whole budget;
    if money already committed elsewhere leaves less than that `free`, every trade shrinks to fit."""
    longs = [t for t in trades if t["side"] == "long" and t.get("weight", 0) > 0]
    skipped = [dict(t, reason="shorting needs a $2,000 margin account") for t in trades if t["side"] != "long"]
    total = sum(t["weight"] for t in longs)
    free = budget if free is None else free
    scale = min(1 / total if total > 1 else 1, free / (budget * total) if total else 1)  # never spend more than is free
    orders = []
    for t in longs:
        usd = round(budget * t["weight"] * scale, 2)
        if usd >= MIN_ORDER:
            orders.append(dict(t, usd=usd))
        else:
            skipped.append(dict(t, reason=f"${usd:.2f} is under Alpaca's $1 minimum"))
    return orders, skipped


def follow(plan, now=None):
    now = now or datetime.now(ET)
    sess = sessions(date.fromisoformat(plan["date"]))
    at = lambda i, when, delay=1: sess[i][0] + timedelta(minutes=delay) if when == "open" else sess[i][1] - BEFORE_CLOSE
    with queue() as q:
        # ponytail: budget ignores realized P&L (no compounding); add it once there is a fill history
        budget = float(env().get("BUDGET", 100))
        free = budget - sum(x["usd"] for x in q if x["state"] in ("queued", "open"))
        orders, skipped = size(plan["trades"], budget, max(free, 0))
        added = []
        for t in orders:
            enter_at, exit_at = at(0, t["enter"], t.get("enter_delay_min", 1)), at(t["hold_days"], t["exit"])
            enter_by = min(enter_at + WINDOW, sess[0][1] - CLOSE_CUTOFF)
            asset = api("GET", f"/v2/assets/{t['symbol']}")
            if not (asset.get("tradable") and asset.get("fractionable")):
                skipped.append(dict(t, reason="Alpaca can't trade this in fractional shares"))
            elif enter_at < now:
                skipped.append(dict(t, reason="entry time already passed"))
            elif exit_at <= enter_at:
                skipped.append(dict(t, reason="exit is not after entry"))
            else:
                added.append(dict(id=uuid.uuid4().hex[:12], symbol=t["symbol"], usd=t["usd"], why=t.get("why", ""),
                                  plan=plan["date"], enter_at=enter_at.isoformat(), enter_by=enter_by.isoformat(),
                                  exit_at=exit_at.isoformat(), state="queued"))
        q.extend(added)
    return {"queued": added, "skipped": skipped, "paper": paper()}


def health_problem(now):
    """Why entries must not go out right now, or None when the server says all is well."""
    try:
        h = json.loads(HEALTH.read_text())
        t = datetime.fromisoformat(h["time"])
        t = t if t.tzinfo else t.astimezone()  # naive = this Mac's local time
    except (OSError, ValueError, KeyError) as err:
        return f"no readable health file ({err})"
    if now - t > HEALTH_MAX_AGE:
        return f"health file is {int((now - t).total_seconds() // 60)} min old"
    if not h.get("ok"):
        return "; ".join(h.get("reasons") or ["server says not ok"])
    return None


def run_due(now=None):
    """Entries: placed between enter_at and enter_by when healthy, else skipped/blocked for good.
    Exits: placed at their time, or at the next open if the market is closed. An open position is never left behind."""
    now = now or datetime.now(ET)
    done = []
    with queue() as q:
        for x in q:
            try:
                if x["state"] == "queued" and now >= datetime.fromisoformat(x["enter_at"]):
                    problem = health_problem(now)
                    if now > datetime.fromisoformat(x["enter_by"]):
                        x["state"] = "missed"
                    elif problem:
                        x.update(state="blocked", reason=problem)
                    else:
                        o = place({"symbol": x["symbol"], "notional": f"{x['usd']:.2f}", "side": "buy",
                                   "type": "market", "time_in_force": "day", "client_order_id": f"fl-{x['id']}-in"})
                        x.update(state="open", entry_order=o["id"])
                elif x["state"] == "open" and now >= datetime.fromisoformat(x["exit_at"]):
                    if not api("GET", "/v2/clock")["is_open"]:
                        continue  # late exit (Mac asleep through the bell): wait for the next open
                    entry = api("GET", f"/v2/orders/{x['entry_order']}")
                    qty = float(entry.get("filled_qty") or 0)
                    x["entry_px"] = entry.get("filled_avg_price")
                    if qty == 0:
                        x["state"] = "not_filled"
                    else:
                        o = place({"symbol": x["symbol"], "qty": entry["filled_qty"], "side": "sell",
                                   "type": "market", "time_in_force": "day", "client_order_id": f"fl-{x['id']}-out"})
                        x.update(state="closed", exit_order=o["id"])
                else:
                    continue
                done.append(f"{x['symbol']} -> {x['state']}")
            except RuntimeError as err:  # one bad order must not block the rest; logged, retried next minute
                x["error"] = str(err)
                done.append(f"{x['symbol']} error: {err}")
    return done


def status():
    with queue() as q:
        for x in q:
            if x["state"] == "closed" and "exit_px" not in x:
                try:
                    x["exit_px"] = api("GET", f"/v2/orders/{x['exit_order']}").get("filled_avg_price")
                except RuntimeError:
                    pass
        rows = list(q)
    try:
        a = api("GET", "/v2/account")
        account = {"cash": a["cash"], "equity": a["equity"], "status": a["status"]}
    except RuntimeError as err:
        account = {"error": str(err)}
    return {"paper": paper(), "budget": float(env().get("BUDGET", 100)), "account": account, "orders": rows}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        for line in run_due():
            print(datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"), line)
    elif cmd == "follow":
        print(json.dumps(follow(json.loads(Path(sys.argv[2]).read_text())), indent=1))
    elif cmd == "check":
        print("PRACTICE (paper)" if paper() else "REAL MONEY", json.dumps(status()["account"]))
    else:
        print(json.dumps(status(), indent=1))
