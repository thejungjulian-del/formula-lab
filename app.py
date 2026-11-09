"""Formula Lab: baseline variables for any ticker + your own formulas on top.

Run:  .venv/bin/python app.py   then open http://localhost:8802
"""
import json
import sys
import time
import uuid
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import yfinance as yf
yf.set_tz_cache_location(__import__("tempfile").mkdtemp(prefix="yf-tz-"))  # own cache per process: a shared one fails when two run at once

HERE = Path(__file__).parent
FORMULAS = HERE / "formulas.json"
PORT = 8802

# name -> (group, meaning, unit). Order here = order in the table.
META = {}
for name, group, meaning, unit, z in [
    ("ret_1d", "Signals", "Return from the previous close to the last close", "pct", True),
    ("rev_5d", "Signals", "Return over the last 5 trading days (high tends to give some back)", "pct", True),
    ("mom_12_1", "Signals", "Return from 12 months ago to 1 month ago", "pct", True),
    ("trend_200", "Signals", "Price ÷ 200-day average − 1", "pct", True),
    ("earn_yield", "Signals", "Last 12 months of earnings per share ÷ price", "pct", False),
    ("gp_assets", "Signals", "Gross profit ÷ total assets (quality)", "num", False),
    ("asset_growth", "Signals", "1-year change in total assets (high tends to lag)", "pct", False),
    ("si_float", "Signals", "Shares sold short ÷ shares available to trade", "pct", False),
    ("vol_63d", "Risk", "Daily volatility σ over the last 63 trading days", "pct", False),
    ("beta_mkt", "Risk", "How much it moves per 1% move in SPY (252 days)", "num", False),
    ("days_to_earn", "Risk", "Trading days until the next earnings report", "days", False),
    ("spread_bps", "Cost", "Bid-ask spread from the live quote (blank outside regular market hours)", "bps", False),
    ("adv_usd", "Cost", "Average daily dollar volume, last 20 days", "usd", False),
    ("price", "Info", "Last close", "usd", False),
    ("market_cap", "Info", "Market value of all shares", "usd", False),
]:
    META[name] = (group, meaning, unit)
    if z:
        META["z_" + name] = (group, f"{name} as a z-score vs. its own 5-year history", "z")

DEFAULT = {
    "params": {"lam": 4, "max_weight": 0.10, "trade_rate": 0.3, "equity": 10000, "w_old": 0},
    "formulas": [
        {"name": "mu", "expr": "0.0002*z_mom_12_1 - 0.0001*z_rev_5d + 0.0001*z_trend_200",
         "note": "Expected daily return (stage 1). Weights are placeholders, not fitted."},
        {"name": "w_star", "expr": "mu / (lam * vol_63d^2)", "note": "Ideal position size (stage 3 shortcut)"},
        {"name": "w_target", "expr": "max(-max_weight, min(max_weight, w_star))", "note": "Capped at max_weight"},
        {"name": "w_new", "expr": "w_old + trade_rate*(w_target - w_old)", "note": "Partial move toward the target (stage 4)"},
        {"name": "trade_usd", "expr": "(w_new - w_old) * equity", "note": "Dollars to buy (+) or sell (−) today"},
    ],
}


def price_vars(close, volume, spy_close):
    """Price-based variables from daily closes. Each z_ value compares today to the series' own history."""
    r = close.pct_change()
    series = {
        "ret_1d": r,
        "rev_5d": close / close.shift(5) - 1,
        "mom_12_1": close.shift(21) / close.shift(252) - 1,
        "trend_200": close / close.rolling(200).mean() - 1,
        "vol_63d": r.rolling(63).std(),
    }
    out = {"price": close.iloc[-1], "adv_usd": (close * volume).tail(20).mean()}
    for k, s in series.items():
        s = s.dropna()
        out[k] = s.iloc[-1]
        if "z_" + k in META:
            out["z_" + k] = (s.iloc[-1] - s.mean()) / s.std()
    both = pd.concat([r, spy_close.pct_change()], axis=1).dropna().tail(252)
    out["beta_mkt"] = both.cov().iloc[0, 1] / both.iloc[:, 1].var()
    return out


def fundamentals(tk, price):
    """Company-level variables. Yahoo leaves many of these blank (ETFs have none), so each is best-effort."""
    try:
        info = tk.info
    except Exception:
        info = {}
    eps, bid, ask = info.get("trailingEps"), info.get("bid"), info.get("ask")
    out = {
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
        "si_float": info.get("shortPercentOfFloat"),
        "target_mean": info.get("targetMeanPrice"), "rec_mean": info.get("recommendationMean"),
        "rec_key": info.get("recommendationKey"), "n_analysts": info.get("numberOfAnalystOpinions"),
        "earn_yield": eps / price if eps else (1 / info["trailingPE"] if info.get("trailingPE") else None),
        "spread_bps": (ask - bid) / ((ask + bid) / 2) * 1e4 if bid and ask and ask > bid and info.get("marketState") == "REGULAR" else None,
    }
    try:
        assets = tk.balance_sheet.loc["Total Assets"]
        out["gp_assets"] = tk.financials.loc["Gross Profit"].iloc[0] / assets.iloc[0]
        out["asset_growth"] = assets.iloc[0] / assets.iloc[1] - 1
    except Exception:
        pass
    try:
        d = tk.calendar.get("Earnings Date", [None])[0]
        out["days_to_earn"] = int(np.busday_count(date.today(), d)) if d else None
    except Exception:
        pass
    return out


def clean(x):
    if isinstance(x, str) or x is None:
        return x
    x = float(x)
    return x if np.isfinite(x) else None


def load(ticker):
    tk = yf.Ticker(ticker)
    h = tk.history(period="5y", auto_adjust=True).dropna(subset=["Close"])  # today's row is blank until the close
    if h.empty:
        raise ValueError(f"No price data for '{ticker}'. Check the ticker symbol.")
    spy = yf.Ticker("SPY").history(period="5y", auto_adjust=True)["Close"].dropna()
    for s in (h, spy):
        s.index = s.index.tz_localize(None).normalize()
    v = {**price_vars(h["Close"], h["Volume"], spy), **fundamentals(tk, float(h["Close"].iloc[-1]))}
    return {"ticker": ticker.upper(), "asof": str(h.index[-1].date()),
            "vars": {k: clean(x) for k, x in v.items()}, "meta": META}


# ---- Sector forecast -------------------------------------------------------------------------

# Macro drivers. "pp" = change in percentage points (yields), "pct" = % change (prices).
DRIVERS = {
    "rates": {"label": "10-year Treasury yield", "unit": "pp", "src": ["^TNX"]},
    "curve": {"label": "Yield curve (10-year minus 3-month)", "unit": "pp", "src": ["^TNX", "^IRX"]},
    "dollar": {"label": "US dollar index", "unit": "pct", "src": ["DX-Y.NYB"]},
    "oil": {"label": "Crude oil (WTI)", "unit": "pct", "src": ["CL=F"]},
    "natgas": {"label": "Natural gas", "unit": "pct", "src": ["NG=F"]},
    "copper": {"label": "Copper", "unit": "pct", "src": ["HG=F"]},
    "gold": {"label": "Gold", "unit": "pct", "src": ["GC=F"]},
}

# Each sector: its ETF, the drivers that move it (with why), and things to watch that have no clean number.
SECTORS = {
    "tech": {"name": "Technology", "etf": "XLK", "yahoo": "Technology",
             "drivers": {"rates": "Tech profits sit far in the future, so higher rates shrink what they're worth today.",
                         "dollar": "A large share of tech revenue is earned abroad; a stronger dollar shrinks it in dollar terms."},
             "watch": ["AI and data-center spending plans from the biggest companies", "Semiconductor orders and inventories",
                       "Analysts raising or cutting earnings estimates", "Valuation: earnings yield vs. the 10-year yield"]},
    "comms": {"name": "Communication", "etf": "XLC", "yahoo": "Communication Services",
              "drivers": {"rates": "Growth-stock valuations fall when rates rise.",
                          "dollar": "The big platforms earn a lot of revenue abroad."},
              "watch": ["Digital ad spending", "Streaming subscriber growth", "Antitrust and regulation news"]},
    "health": {"name": "Healthcare", "etf": "XLV", "yahoo": "Healthcare",
               "drivers": {"rates": "Steady earnings make it partly a bond substitute, so it reacts to rates.",
                           "dollar": "Drugmakers and device makers sell worldwide."},
               "watch": ["Drug-pricing policy and Medicare negotiations", "FDA approvals and clinical-trial results",
                         "Patents expiring on top-selling drugs", "Insurer and hospital reimbursement rates"]},
    "fin": {"name": "Financials", "etf": "XLF", "yahoo": "Financial Services",
            "drivers": {"rates": "Higher long rates usually let banks earn more on new loans.",
                        "curve": "Banks borrow short and lend long; a steeper curve widens that margin."},
            "watch": ["Loan growth", "Credit losses and late payments", "Capital rules and stress-test results",
                      "Deal-making and trading activity"]},
    "energy": {"name": "Energy", "etf": "XLE", "yahoo": "Energy",
               "drivers": {"oil": "The oil price is most energy companies' revenue.",
                           "natgas": "Gas producers and pipelines earn from gas prices."},
               "watch": ["OPEC+ production decisions", "Weekly US oil inventories (EIA report)", "Rig counts", "Refining margins"]},
    "materials": {"name": "Materials", "etf": "XLB", "yahoo": "Basic Materials",
                  "drivers": {"copper": "Copper tracks global industrial demand.",
                              "gold": "Gold and metal miners earn from metal prices.",
                              "dollar": "Commodities are priced in dollars; a stronger dollar pushes them down."},
                  "watch": ["China construction and factory data", "Steel and chemical prices", "Global manufacturing surveys (PMIs)"]},
    "industrials": {"name": "Industrials", "etf": "XLI", "yahoo": "Industrials",
                    "drivers": {"copper": "Copper rises when factories and construction are busy.",
                                "rates": "Big industrial projects and equipment are financed with debt."},
                    "watch": ["ISM manufacturing index (PMI)", "Business spending plans", "Freight and rail volumes",
                              "Defense and infrastructure budgets"]},
    "discretionary": {"name": "Consumer Discretionary", "etf": "XLY", "yahoo": "Consumer Cyclical",
                      "drivers": {"rates": "Higher rates raise the cost of car loans, mortgages and credit cards.",
                                  "oil": "Higher gas prices leave shoppers less to spend."},
                      "watch": ["Retail sales", "Consumer confidence", "Credit-card late payments", "Housing activity"]},
    "staples": {"name": "Consumer Staples", "etf": "XLP", "yahoo": "Consumer Defensive",
                "drivers": {"rates": "Steady dividend payers compete with bonds for investors.",
                            "dollar": "The big brands sell worldwide."},
                "watch": ["Input costs (grains, packaging, freight)", "Price increases vs. units sold", "Store-brand competition"]},
    "utilities": {"name": "Utilities", "etf": "XLU", "yahoo": "Utilities",
                  "drivers": {"rates": "Utilities act like bonds and carry heavy debt.",
                              "natgas": "Gas is a main fuel cost for power plants."},
                  "watch": ["Electricity demand from data centers", "Regulators' rate decisions", "Wildfire and storm liabilities"]},
    "realestate": {"name": "Real Estate", "etf": "XLRE", "yahoo": "Real Estate",
                   "drivers": {"rates": "Property values and REIT borrowing costs move with long rates."},
                   "watch": ["Mortgage rates", "Office and apartment vacancy", "Prices of property sales (cap rates)",
                             "Debt coming due for refinancing"]},
}
HORIZONS = (1, 2)  # days only: this is a day-trading tool
_cache = {}


def ols(y, X):
    """Least squares with an intercept. Returns coefficients, t-stats, R², residual std dev."""
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, b / se, 1 - e.var() / y.var(), np.sqrt(s2)


def level(d, px):
    a = px[DRIVERS[d]["src"]]
    return a.iloc[:, 0] - a.iloc[:, 1] if a.shape[1] == 2 else a.iloc[:, 0]


def change(s, unit, n=1):
    return s.diff(n) if unit == "pp" else s.pct_change(n)


def forecast_data(ticker, key=None):
    """Everything the forecast calculator needs for one stock, auto-filled from Yahoo."""
    ticker = ticker.upper()
    stock = load(ticker)
    found = next((k for k, s in SECTORS.items() if s["etf"] == ticker or s["yahoo"] == stock["vars"].get("sector")), None)
    auto = key is None and found is not None  # False = you picked it, or we guessed Technology
    key = key or found or "tech"
    if (ticker, key) in _cache and time.time() - _cache[ticker, key][0] < 60:  # 1 minute, so prices stay live
        return {**_cache[ticker, key][1], "auto": auto}
    s = SECTORS[key]
    srcs = sorted({t for d in s["drivers"] for t in DRIVERS[d]["src"]})
    px = yf.download(sorted({"SPY", ticker, *srcs}), period="5y", auto_adjust=True, progress=False)["Close"].ffill()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    wk = px.resample("W-FRI").last()
    names = ["SPY", *s["drivers"]]
    df = pd.concat([wk[ticker].pct_change(), wk["SPY"].pct_change(),
                    *[change(level(d, wk), DRIVERS[d]["unit"]) for d in s["drivers"]]],
                   axis=1, keys=["y", *names]).dropna().tail(156)  # last 3 years of weeks
    b, t, r2, resid = ols(df["y"].values, df[names].values)
    spy = px["SPY"]
    out = {
        "ticker": ticker, "asof": stock["asof"], "vars": stock["vars"],
        "sector": key, "sector_name": s["name"], "etf": s["etf"], "auto": auto,
        "sectors": {k: v["name"] for k, v in SECTORS.items()},
        "fit": {"weeks": len(df), "since": str(df.index[0].date()), "r2": clean(r2), "resid_wk": clean(resid)},
        "market": {"beta": clean(b[1]), "t": clean(t[1]),
                   "avg": {h: clean((spy / spy.shift(h) - 1).mean()) for h in HORIZONS},
                   "chg": {h: clean(spy.iloc[-1] / spy.iloc[-1 - h] - 1) for h in HORIZONS}},
        "drivers": [{"key": d, "label": DRIVERS[d]["label"], "unit": DRIVERS[d]["unit"], "why": why,
                     "level": clean(level(d, px).iloc[-1]), "beta": clean(b[i + 2]), "t": clean(t[i + 2]),
                     "chg": {h: clean(change(level(d, px), DRIVERS[d]["unit"], h).iloc[-1]) for h in HORIZONS}}
                    for i, (d, why) in enumerate(s["drivers"].items())],
        "watch": s["watch"],
    }
    _cache[ticker, key] = (time.time(), out)
    return out


# ---- Journal: trades you log, your positions, and how accurate past forecasts were -------------

JOURNAL = HERE / "journal.json"
TAU = 0.5  # prior spread of k around 1: before any results, k is believed to be 1 ± 0.5
_jcache = []


def read_journal():
    return json.loads(JOURNAL.read_text()) if JOURNAL.exists() else []


SIGN = {"BUY": 1, "COVER": 1, "SELL": -1, "SHORT": -1}


def positions(entries, prices):
    """Signed positions with average cost: shares > 0 = long, < 0 = short.
    Closing trades (SELL a long, COVER a short) realize profit against the average price."""
    book = {}
    for e in sorted(entries, key=lambda e: e["created"]):
        if e["action"] == "HOLD":
            continue
        p = book.setdefault(e["ticker"], {"ticker": e["ticker"], "shares": 0.0, "cost": 0.0, "realized": 0.0})
        n = SIGN[e["action"]] * e["shares"]
        if p["shares"] == 0 or (n > 0) == (p["shares"] > 0):          # opening or adding
            p["cost"] += n * e["price"]
        else:                                                          # closing (no flips: add_entry blocks them)
            avg = p["cost"] / p["shares"]
            p["realized"] += -n * (e["price"] - avg)
            p["cost"] += n * avg
        p["shares"] += n
    out = []
    for p in book.values():
        price = prices.get(p["ticker"])
        value = p["shares"] * price if price else None
        out.append({**p, "avg_cost": p["cost"] / p["shares"] if p["shares"] else None, "price": price, "value": value,
                    "unrealized": value - p["cost"] if value is not None else None})
    return out


def calibrate(rows):
    """rows = (predicted return, actual return, forecast σ). k = how much to scale future forecasts.
    Each result is weighted by 1/σ², so a forecast that is small next to normal swings teaches little.
    k_raw is the best-fit scale; k blends it with the prior belief k = 1 (Bayesian shrinkage)."""
    n = len(rows)
    if not n:
        return {"n": 0, "hit": None, "k_raw": None, "k": 1.0, "avg_pred": None, "avg_actual": None}
    pr, ac, sd = np.array(rows, dtype=float).T
    sxy, sxx = np.sum(pr * ac / sd**2), np.sum(pr**2 / sd**2)
    return {"n": n, "hit": float(np.mean(np.sign(pr) == np.sign(ac))), "k_raw": float(sxy / sxx) if sxx > 0 else None,
            "k": float(min(max((sxy + TAU**-2) / (sxx + TAU**-2), 0.0), 1.5)),
            "avg_pred": float(pr.mean()), "avg_actual": float(ac.mean())}


def journal_summary():
    if _jcache and time.time() - _jcache[0] < 600:
        return _jcache[1]
    entries = read_journal()
    tickers = sorted({e["ticker"] for e in entries})
    px = pd.DataFrame()
    if tickers:
        px = yf.download(tickers, period="2y", auto_adjust=True, progress=False)["Close"].ffill()
        px.index = pd.to_datetime(px.index).tz_localize(None)
    prices = {t: clean(px[t].iloc[-1]) for t in tickers if t in px}
    resolved = []
    for e in entries:
        if e.get("source") in ("plan", "sim"):
            continue  # plan/simulated trades are scored by their own results, not used to tune the forecast's k
        due = pd.Timestamp(e["review_date"])
        if e["ticker"] in px and due <= px.index[-1]:
            actual = float(px[e["ticker"]].asof(due) / e["price"] - 1)
            resolved.append({"id": e["id"], "ticker": e["ticker"], "date": e["date"], "review_date": e["review_date"],
                             "action": e["action"], "predicted": e["forecast"]["mu_raw"], "actual": actual,
                             "sigma": e["forecast"]["sigma"]})
    data = {"entries": entries, "positions": positions(entries, prices), "resolved": resolved,
            "calib": calibrate([(r["predicted"], r["actual"], r["sigma"]) for r in resolved]),
            "last_close": str(px.index[-1].date()) if len(px) else None}
    _jcache[:] = [time.time(), data]
    return data


def add_entry(e):
    if e.get("action") not in ("BUY", "SELL", "SHORT", "COVER", "HOLD"):
        raise ValueError("Action must be BUY, SELL, SHORT, COVER or HOLD")
    e["shares"], e["price"] = float(e.get("shares") or 0), float(e.get("price") or 0)
    if e["price"] <= 0 or e["shares"] < 0 or (e["action"] != "HOLD" and e["shares"] == 0):
        raise ValueError("Enter a positive price and number of shares")
    entries = read_journal()
    held = next((p["shares"] for p in positions(entries, {}) if p["ticker"] == e["ticker"]), 0)
    if e["action"] == "SELL" and e["shares"] > held + 1e-9:
        raise ValueError(f"You only own {max(held, 0):g} shares of {e['ticker']} to sell")
    if e["action"] == "COVER" and e["shares"] > -held + 1e-9:
        raise ValueError(f"You are only short {max(-held, 0):g} shares of {e['ticker']}")
    if e["action"] == "BUY" and held < 0:
        raise ValueError(f"You are short {e['ticker']}; log a COVER first")
    if e["action"] == "SHORT" and held > 0:
        raise ValueError(f"You own {e['ticker']}; log a SELL first")
    e.update(id=uuid.uuid4().hex[:8], created=datetime.now().isoformat(timespec="seconds"), date=str(date.today()))
    JOURNAL.write_text(json.dumps(entries + [e], indent=2))
    _jcache.clear()
    return e


def delete_entry(entry_id):
    JOURNAL.write_text(json.dumps([e for e in read_journal() if e["id"] != entry_id], indent=2))
    _jcache.clear()


def quote(ticker):
    """Latest trade price and the exchange time it happened (Yahoo's real-time feed, no 15-minute delay)."""
    tk = yf.Ticker(ticker)
    tk.history(period="1d", interval="1m")
    m = tk.history_metadata
    return {"ticker": ticker.upper(), "price": m["regularMarketPrice"], "time": m["regularMarketTime"].timestamp(),
            "delayed_min": m.get("exchangeDataDelayedBy") or 0}


def model_today(df, spy, vix):
    """Today's values of the tested day-trading signals (daily_model.py) and what each adds to the forecast."""
    import daily_model
    m = json.loads((HERE / "daily_model.json").read_text())
    f = daily_model.features(df, spy, vix).iloc[-1]
    out = {}
    for h, r in m["horizons"].items():
        rows = []
        for n, w in r["weights"].items():
            z = clean(np.clip((f[n] - m["mean"][n]) / m["std"][n], -5, 5))
            rows.append({"name": n, "desc": m["features"][n], "z": z, "w": w["w"], "adds": None if z is None else w["w"] * z})
        out[h] = {"passed": r["passed"], "rows": rows, "oos": r.get("oos"), "tested": len(m["features"])}
    # Qualitative reasons (qualitative.py): shown with their test results; none passed, so none adds to the forecast.
    # ponytail: if a future re-run sets "use", wire its combined weights in here.
    qj = json.loads((HERE / "qualitative.json").read_text())
    qv = qual_today(df.index, ticker=df.attrs.get("ticker"))
    for h in out:
        out[h]["qual"] = [{"name": n, "desc": d, "value": qv.get(n), **{k: qj["horizons"][h]["single"][n][k] for k in ("in_t", "out_t", "both")}}
                          for n, d in qj["qualitative"].items()]
        out[h]["qual_use"] = qj["horizons"][h]["use"]
    return out


_qcache = {}


def qual_today(idx, ticker):
    """Latest analyst/earnings values for a ticker, cached for an hour (they change a few times a quarter)."""
    import qualitative
    if ticker in _qcache and time.time() - _qcache[ticker][0] < 3600:
        return _qcache[ticker][1]
    try:
        v = {k: clean(x) for k, x in qualitative.q_features(idx, *qualitative.events(ticker)).iloc[-1].items()}
    except Exception:
        v = {}
    _qcache[ticker] = (time.time(), v)
    return v


def signals_today(ticker):
    """Where a stock ranks today on the signals that passed the locked test (from the latest research scan)."""
    import daily_scan
    files = sorted(daily_scan.OUT.glob("*.json"))
    P = json.loads(files[-1].read_text()).get("passed") if files else None
    stats = json.loads((HERE / "passed_stats.json").read_text())
    if not P or "all" not in P:
        return {"available": False, "why": "Run a research scan first."}
    rows = {r["ticker"]: r for r in P["all"]}
    key = {"gap_D_rel": "gap_rel", "vol_surprise": "volume"}
    out = {"available": True, "date": P["date"], "live": P["live"], "n": len(rows), "test_cost": stats["test_cost"],
           "ranked": ticker in rows, "signals": {}}
    for name, st in stats["signals"].items():
        sig = {"trade": st["trade"], "desc": st["desc"], "holdout_t": st["holdout_t"],
               "all": {k: st["all"][k] for k in ("avg", "quantiles")}}
        if ticker in rows:
            x = rows[ticker][key[name]]
            vals = sorted(r[key[name]] * st["sign"] for r in rows.values())
            pct = float(np.searchsorted(vals, x * st["sign"], side="right") / len(vals))  # 1.0 = strongest buy
            fifth = str(min(5, max(1, int(np.ceil(pct * 5)))))
            sig.update({"value": x, "raw": rows[ticker].get("gap") if name == "gap_D_rel" else float(np.exp(x)),
                        "pct": pct, "fifth": int(fifth), "edge": st["fifths"][fifth]["avg"] - st["all"]["avg"],  # measured, 2023+
                        "fifth_stats": {k: st["fifths"][fifth][k] for k in ("avg", "quantiles", "trades")}})
        out["signals"][name] = sig
    return out


# ---- Plan: today's two legs, sent to Alpaca by broker.py ----------------------------------------------------

RESEARCH = HERE / "research"
HEALTH_FILE = HERE / "data" / "health.json"
_health = {"t": 0, "r": None}
# Follow windows (minutes after midnight ET). The day leg needs the opening gap (picks ~9:31) and enters at 9:35;
# the night leg needs the day's trading (picks 15:50/15:55) and enters at 15:59.
WINDOW = {"day": (571, 575), "night": (950, 958)}
WHEN = {"day": "Follow 6:31–6:34 AM PT · buys at 6:35, sells at the 1:00 PM close",
        "night": "Follow 12:50–12:57 PM PT · buys at 12:59, sells at the next 6:30 AM open"}


def health_state():
    """Cached health check; also writes data/health.json, which broker.py reads before every entry."""
    import health
    if time.time() - _health["t"] > 60:
        r = health.check(sys.modules[__name__])
        _health.update(t=time.time(), r=r)
        HEALTH_FILE.write_text(json.dumps({"ok": r["ok"], "time": r["time"], "reasons": [c["detail"] for c in r["checks"] if c["blocks"]]}))
    return _health["r"]


def health_loop():
    """Keeps data/health.json fresh while the server runs, even with no page open."""
    from datetime import timezone
    while True:
        try:
            health_state()
        except Exception as e:  # a crashed check must read as "not ok", never as a stale "ok"
            HEALTH_FILE.write_text(json.dumps({"ok": False, "time": datetime.now(timezone.utc).isoformat(), "reasons": [f"health check crashed: {e}"]}))
        time.sleep(60)


def latest_scan():
    files = sorted(RESEARCH.glob("*.json"))
    return json.loads(files[-1].read_text()) if files else {}


def leg_status(leg, plan, now):
    m = now.hour * 60 + now.minute
    lo, hi = WINDOW[leg]
    if not plan["live"] or plan["date"] != str(now.date()):
        return "later"
    return "now" if lo <= m < hi else "missed" if m >= hi else "later"


def plan_state(now=None, include_plan=False):
    import broker
    import planner
    now = now or pd.Timestamp.now(tz="America/New_York")
    H = health_state()
    scan = latest_scan()
    plan = planner.build(scan, H["excluded"])
    planner.write(plan)
    # Off-hours: show what the last session's ranking picked (reference only) and when the next window opens
    P0 = scan.get("passed") or {}
    rows0 = [dict(r, news=False if r.get("news") is None else r["news"]) for r in P0.get("all", [])]   # reference only: news not checked
    preview = planner.build({"passed": dict(P0, live=True, all=rows0)}, H["excluded"]) if not plan["live"] else None
    m = now.hour * 60 + now.minute
    nxt_day = now.normalize() if now.weekday() < 5 and m < WINDOW["day"][0] else now.normalize() + pd.offsets.BDay(1)
    nxt_night = now.normalize() if now.weekday() < 5 and m < WINDOW["night"][0] else now.normalize() + pd.offsets.BDay(1)
    next_at = {"day": (nxt_day + pd.Timedelta(minutes=WINDOW["day"][0])).isoformat(),      # ponytail: holidays not skipped
               "night": (nxt_night + pd.Timedelta(minutes=WINDOW["night"][0])).isoformat()}
    budget = float(broker.env().get("BUDGET", 100))
    combo = json.loads((HERE / "combo.json").read_text())["trades"]
    ex = json.loads((HERE / "exec7.json").read_text())                 # 7 years of real fills at the broker's times
    legs = []
    for leg, trade in (("day", "h0"), ("night", "ovn")):
        kept = max(0.0, min(1.0, ex[leg]["0935" if leg == "day" else "0931"]["kept"]))
        t5 = json.loads((HERE / "plan_stats.json").read_text())[leg]   # the plan as it trades now (day leg: no-news only)
        shift = (1 - kept) * t5["avg"]                     # edge lost to real fill times (exec_check.py)
        q = [x - shift for x in t5["quantiles"]]
        longs = [dict(x, usd=round(budget * x["weight"], 2)) for x in plan["trades"] if x["leg"] == leg and x["side"] == "long"]
        usd = sum(x["usd"] for x in longs)
        legs.append({"leg": leg, "when": WHEN[leg], "status": "paused" if leg in planner.PAUSED else leg_status(leg, plan, now),
                     "paused": planner.PAUSED.get(leg), "longs": longs, "next_at": next_at[leg],
                     "preview": [x["symbol"] for x in (preview or {"trades": []})["trades"] if x["leg"] == leg and x["side"] == "long"],
                     "skipped_news": plan.get("skipped_news", []) if leg == "day" else [],
                     "shorts": [x for x in plan["trades"] if x["leg"] == leg and x["side"] == "short"],
                     "metrics": {"expected": usd * (t5["avg"] - shift - 0.0005), "win": float(1 - np.interp(0.0005, q, np.linspace(0, 1, len(q)))),
                                 "cost": usd * 0.0005, "bad": usd * (q[10] - 0.0005), "kept": kept}})
    try:
        B = broker.status()
    except Exception as e:
        B = {"error": str(e)}
    scan_conf = json.loads((HERE / "confidence.json").read_text())
    return {"now": now.isoformat(timespec="seconds"), "health": H, "plan_date": plan["date"], "live": plan["live"],
            "label": plan["label"], "note": plan["note"], "legs": legs, "ranked": len(plan["all"]), "budget": budget,
            "confidence": json.loads((HERE / "plan_stats.json").read_text())["account"], "broker": B, **({"plan": plan} if include_plan else {})}


def follow_leg(leg, now=None):
    """Send one leg of today's plan to broker.follow. Blocked outside its window or when a health check fails."""
    import broker
    st = plan_state(now, include_plan=True)
    L = next(x for x in st["legs"] if x["leg"] == leg)
    if L["status"] != "now":
        raise ValueError(f"This part can't be followed right now. {L['when']}.")
    if not st["health"]["ok"]:
        raise ValueError("Blocked: " + " ".join(c["detail"] for c in st["health"]["checks"] if c["blocks"]))
    plan = st["plan"]
    return broker.follow({"date": plan["date"], "trades": [t for t in plan["trades"] if t["leg"] == leg]})


# ---- Short-term setups: which tested setups are active for a ticker today -----------------------

def setups_today(ticker):
    import shortterm
    ticker = ticker.upper()
    raw = yf.download(sorted({ticker, "^VIX", "SPY"}), period="2y", auto_adjust=True, group_by="ticker", progress=False)
    df = raw[ticker].dropna(subset=["Open", "Close"])
    df.attrs["ticker"] = ticker
    sig = shortterm.signals(df, raw["^VIX"]["Close"]).iloc[-1].to_dict()
    # Live fixes: the next trading day isn't in the data yet. Holidays are ignored (approximation).
    nxt = df.index[-1] + pd.offsets.BDay(1)
    month_days = pd.bdate_range(nxt.replace(day=1), nxt + pd.offsets.BMonthEnd(0))
    sig["turn_of_month"] = bool(list(month_days).index(nxt) < 3 or nxt == month_days[-1])
    sig["gap_down"] = sig["gap_up"] = "open"  # only known at the open
    for name in shortterm.SETUPS:
        if "+" in name:
            parts = [sig[x] for x in name.split("+")]
            sig[name] = "open" if all(x in (True, "open") for x in parts) and "open" in parts else all(x is True for x in parts)
    tested = json.loads((HERE / "shortterm_results.json").read_text())
    res = {}
    for r in tested["results"]:
        res.setdefault(r["setup"], {})[r["hold"]] = {k: r[k] for k in ("in_sample", "out_of_sample", "verified", "out_of_sample_baseline")}
    return {"ticker": ticker, "asof": str(df.index[-1].date()), "next_day": str(nxt.date()), "tested_on": tested["generated"],
            "universe": len(tested["coverage"]), "cost": tested["cost"],
            "model": model_today(df, raw["SPY"]["Close"].dropna(), raw["^VIX"]["Close"].dropna()),
            "setups": [{"name": n, "desc": d, "side": side, "active": sig[n] if sig[n] == "open" else bool(sig[n]), "results": res[n]}
                       for n, (d, _, side) in shortterm.SETUPS.items()]}


# ---- Daily research --------------------------------------------------------------------------

_research = {"running": False, "error": None, "track": None}


def research_state():
    import daily_scan
    files = sorted(daily_scan.OUT.glob("*.json")) if daily_scan.OUT.exists() else []
    if not _research["track"] or time.time() - _research["track"][0] > 600:
        _research["track"] = (time.time(), daily_scan.track_record())
    return {"latest": json.loads(files[-1].read_text()) if files else None, "dates": [f.stem for f in files],
            "track": _research["track"][1], "running": _research["running"], "error": _research["error"]}


def run_research():
    import daily_scan, threading
    if _research["running"]:
        return

    def work():
        _research.update(running=True, error=None)
        try:
            daily_scan.scan()
            _research["track"] = None
        except Exception as e:
            _research["error"] = str(e)
        _research["running"] = False
    threading.Thread(target=work, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/plan"):
            return self.reply(200, (HERE / "plan.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/plan":
            try:
                return self.reply(200, plan_state())
            except Exception as e:
                return self.reply(400, {"error": str(e)})
        if u.path == "/api/ping":
            return self.reply(200, {"ok": True})
        if u.path == "/api/health":
            _health["t"] = 0
            return self.reply(200, health_state())
        if u.path == "/forecast":
            return self.reply(200, (HERE / "forecast.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/builder":
            return self.reply(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/vars":
            t = parse_qs(u.query).get("t", [""])[0].strip()
            try:
                return self.reply(200, load(t))
            except Exception as e:
                return self.reply(400, {"error": str(e)})
        if u.path == "/api/forecast":
            q = parse_qs(u.query)
            try:
                return self.reply(200, forecast_data(q.get("t", [""])[0].strip(), q.get("s", [None])[0]))
            except Exception as e:
                return self.reply(400, {"error": str(e)})
        if u.path == "/api/quote":
            try:
                return self.reply(200, quote(parse_qs(u.query).get("t", [""])[0].strip()))
            except Exception as e:
                return self.reply(400, {"error": str(e)})
        if u.path == "/api/signals":
            try:
                return self.reply(200, signals_today(parse_qs(u.query).get("t", [""])[0].strip().upper()))
            except Exception as e:
                return self.reply(400, {"error": str(e)})
        if u.path == "/journal":
            return self.reply(200, (HERE / "journal.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/journal":
            return self.reply(200, journal_summary())
        if u.path == "/research":
            return self.reply(200, (HERE / "research.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/research":
            return self.reply(200, research_state())
        if u.path == "/api/setups":
            try:
                return self.reply(200, setups_today(parse_qs(u.query).get("t", [""])[0].strip()))
            except Exception as e:
                return self.reply(400, {"error": str(e)})
        if u.path == "/api/formulas":
            return self.reply(200, json.loads(FORMULAS.read_text()) if FORMULAS.exists() else DEFAULT)
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            if self.path == "/api/formulas":
                FORMULAS.write_text(json.dumps(data, indent=2))
                return self.reply(200, {"ok": True})
            if self.path == "/api/journal":
                return self.reply(200, add_entry(data))
            if self.path == "/api/follow":
                return self.reply(200, follow_leg(data["leg"]))
            if self.path == "/api/plan/refresh":
                import daily_scan
                daily_scan.refresh_picks()
                _health["t"] = 0
                return self.reply(200, {"ok": True})
            if self.path == "/api/research/run":
                run_research()
                return self.reply(200, {"ok": True})
            if self.path == "/api/journal/delete":
                delete_entry(data["id"])
                return self.reply(200, {"ok": True})
        except Exception as e:
            return self.reply(400, {"error": str(e)})
        self.reply(404, {"error": "not found"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import threading
    threading.Thread(target=health_loop, daemon=True).start()
    print(f"Formula Lab on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
