"""Health checks that stand between the plan and a purchase.

Each check either passes, gets fixed automatically (only safe fixes: re-run the scan, clear caches, reload the
scan job), or BLOCKS the plan's "Follow plan" button with a plain reason. Nothing here ever guesses a price,
edits the portfolio, or trades.

Run by hand: .venv/bin/python health.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
_tests = {"t": 0, "ok": None, "out": ""}
_last_fix = {"scan": 0}


def market_open(now_et):
    m = now_et.hour * 60 + now_et.minute
    return now_et.weekday() < 5 and 570 <= m < 960


def run_tests():
    """The project's self-checks (test_vars.py), at most once an hour."""
    if time.time() - _tests["t"] > 3600:
        r = subprocess.run([sys.executable, str(HERE / "test_vars.py")], capture_output=True, text=True, timeout=300, cwd=HERE)
        _tests.update(t=time.time(), ok=r.returncode == 0, out=(r.stdout + r.stderr)[-400:])
    return _tests["ok"], _tests["out"]


def scan_issues(scan, now_et):
    """Problems with the latest research scan that a fresh scan could fix."""
    if not scan:
        return ["No research scan has run yet."]
    out = []
    ranked_at = max(pd.Timestamp(scan["scan_time"]), pd.Timestamp(scan.get("picks_time", scan["scan_time"])))
    age_min = (now_et.tz_localize(None) - ranked_at).total_seconds() / 60
    m = now_et.hour * 60 + now_et.minute
    if market_open(now_et) and (570 <= m < 580 or 950 <= m < 960) and age_min > 10:
        out.append(f"The ranking is {age_min:.0f} minutes old during a buy window (must be under 10). It refreshes at 6:31/6:33 and 12:56/12:58 PT.")
    if scan.get("stale"):
        out.append("The scan is missing the latest trading day's prices.")
    errors = sum(1 for p in scan.get("picks", []) if "error" in p)
    if errors > 3:
        out.append(f"{errors} stocks failed to load in the scan.")
    P = scan.get("passed") or {}
    if "error" in P:
        out.append("The tested-signal ranking failed: " + str(P["error"])[:120])
    elif P.get("n", 0) < 450:
        out.append(f"Only {P.get('n', 0)} of ~500 S&P 500 stocks were ranked (data is incomplete).")
    if market_open(now_et) and P.get("date") != str(now_et.date()):
        out.append("The ranking is not from today's session yet.")
    return out


def bad_picks(scan):
    """Tickers whose numbers look broken (bad price, impossible gap or volume). They are left out of the plan."""
    P = (scan or {}).get("passed") or {}
    bad = {}
    for r in P.get("day", []) + P.get("night", []) + P.get("gap_rel", []) + P.get("volume", []):
        px = r.get("price")
        if not (isinstance(px, (int, float)) and np.isfinite(px) and px > 0):
            bad[r["ticker"]] = "missing or zero price"
        elif abs(r.get("gap", 0)) > 0.25:
            bad[r["ticker"]] = f"opened {r['gap'] * 100:+.0f}% (possible split or bad data)"
        elif np.exp(r.get("volume", 0)) > 50:
            bad[r["ticker"]] = "volume 50x normal (possible bad data)"
    return bad


def check(app, fix=True):
    """Returns {"ok": can the plan be followed, "checks": [...], "excluded": {ticker: why}}."""
    now_et = pd.Timestamp.now(tz="America/New_York")
    checks = []

    def add(name, ok, detail, blocks=True, fixing=None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "blocks": blocks and not ok, "fixing": fixing})

    # 1. model and test-result files exist and can be read
    missing = []
    for f in ("daily_model.json", "passed_stats.json", "holdout_results.json", "shortterm_results.json"):
        try:
            json.loads((HERE / f).read_text())
        except Exception:
            missing.append(f)
    add("Model files", not missing, "All model files load." if not missing else "Cannot read: " + ", ".join(missing))

    # 2. latest scan: fresh, complete, from today's session (fix: run a new scan)
    files = sorted((HERE / "research").glob("*.json"))
    scan = json.loads(files[-1].read_text()) if files else None
    issues = scan_issues(scan, now_et)
    fixing = None
    if issues and fix and not app._research["running"] and time.time() - _last_fix["scan"] > 300:
        _last_fix["scan"] = time.time()
        import threading, daily_scan
        threading.Thread(target=lambda: daily_scan.refresh_picks(), daemon=True).start()
        fixing = "Refreshing the ranking (about 10 seconds)."
    elif issues and app._research["running"]:
        fixing = "A fresh scan is running."
    add("Research scan", not issues, "Fresh and complete." if not issues else " ".join(issues), fixing=fixing)

    # 3. live prices are arriving (market hours only)
    if market_open(now_et):
        try:
            q = app.quote("SPY")
            age = time.time() - q["time"]
            add("Live prices", age < 180, f"Last S&P 500 trade {age:.0f} s ago." if age < 180 else
                f"Prices are {age / 60:.0f} minutes old. The data source may be down; wait and refresh.")
        except Exception as e:
            add("Live prices", False, f"Could not get a live price ({str(e)[:80]}).")
    else:
        add("Live prices", True, "Market closed: nothing to check.", blocks=False)

    # 4. suspicious tickers are dropped from the plan instead of blocking everything
    excluded = bad_picks(scan)
    add("Price sanity", True, "All picks look normal." if not excluded else
        "Left out: " + "; ".join(f"{t} ({w})" for t, w in excluded.items()), blocks=False)

    # 5. the project's own self-tests
    ok, out = run_tests()
    add("Self-tests", ok, "All calculations pass their checks." if ok else "A self-test failed: " + out[-200:])

    # 6. portfolio file is readable
    try:
        app.positions(app.read_journal(), {})
        add("Portfolio", True, "Portfolio log reads correctly.")
    except Exception as e:
        add("Portfolio", False, f"The portfolio log has a problem: {str(e)[:120]}")

    # 7. the scheduled scan job is loaded (fix: load it again)
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
    job = subprocess.run(["launchctl", "print", f"gui/{uid}/com.formulalab.scan"], capture_output=True, text=True)
    jfix = None
    if job.returncode != 0 and fix:
        plist = Path.home() / "Library/LaunchAgents/com.formulalab.scan.plist"
        subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(plist)], capture_output=True)
        jfix = "Reloaded the scan schedule."
    add("Scan schedule", job.returncode == 0, "Scheduled scans are on." if job.returncode == 0 else
        "Scheduled scans were off.", blocks=False, fixing=jfix)

    if any(c["blocks"] for c in checks) and fix:
        app._cache.clear(), app._jcache.clear(), app._qcache.clear()  # next request reloads everything fresh
    return {"ok": not any(c["blocks"] for c in checks), "time": now_et.isoformat(timespec="seconds"),
            "checks": checks, "excluded": excluded}


if __name__ == "__main__":
    import app
    r = check(app, fix=False)
    print("OK TO TRADE" if r["ok"] else "BLOCKED")
    for c in r["checks"]:
        print(f"  {'✓' if c['ok'] else '✗'} {c['name']}: {c['detail']}")
