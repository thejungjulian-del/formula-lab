"""Downloads and caches everything the signal search needs (data/ folder). Re-run to refresh.

  universe   S&P 500 members today (Wikipedia list): ticker, sector, SEC CIK, Wikipedia article
  prices     daily open/high/low/close/volume since 2009 (Yahoo)
  macro      SPY, VIX, bitcoin, Nikkei, 10-year yield, dollar, oil, gold (Yahoo)
  wiki       daily Wikipedia pageviews per company since 2015-07 (Wikimedia API)
  sec        every SEC filing per company with its exact acceptance time (data.sec.gov)
  events     analyst actions + earnings surprises (Yahoo, same as qualitative.py)

Run: .venv/bin/python search_data.py [step ...]
"""
import io
import pickle
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
import yfinance as yf
yf.set_tz_cache_location(__import__("tempfile").mkdtemp(prefix="yf-tz-"))  # own cache per process: a shared one fails when two run at once

DATA = Path(__file__).with_name("data")
UA = {"User-Agent": "FormulaLab personal research project"}
MACRO = {"spy": "SPY", "vix": "^VIX", "btc": "BTC-USD", "nikkei": "^N225", "tnx": "^TNX", "dollar": "DX-Y.NYB",
         "oil": "CL=F", "gold": "GC=F"}


def save(name, obj):
    DATA.mkdir(exist_ok=True)
    pickle.dump(obj, open(DATA / f"{name}.pkl", "wb"))


def load(name):
    return pickle.load(open(DATA / f"{name}.pkl", "rb"))


def universe():
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=60).text
    t = pd.read_html(io.StringIO(html), attrs={"id": "constituents"}, extract_links="body")[0]
    u = pd.DataFrame({
        "ticker": [x[0].replace(".", "-") for x in t["Symbol"]],
        "name": [x[0] for x in t["Security"]],
        "wiki": [unquote(x[1].split("/wiki/")[-1]) if x[1] else None for x in t["Security"]],
        "sector": [x[0] for x in t["GICS Sector"]],
        "cik": [int(x[0]) for x in t["CIK"]],
    })
    save("universe", u)
    print(len(u), "companies")


def prices():
    u = load("universe")
    out = {}
    tickers = list(u["ticker"])
    for i in range(0, len(tickers), 100):
        raw = yf.download(tickers[i:i + 100], start="2009-01-01", auto_adjust=True, group_by="ticker", progress=False, threads=True)
        for t in tickers[i:i + 100]:
            try:
                df = raw[t].dropna(subset=["Open", "Close"])
                if len(df) > 300:
                    out[t] = df.astype("float32")
            except KeyError:
                pass
        print(f"prices {min(i + 100, len(tickers))}/{len(tickers)}: {len(out)} ok")
    save("prices", out)


def macro():
    raw = yf.download(list(MACRO.values()), start="2009-01-01", auto_adjust=True, group_by="ticker", progress=False)
    save("macro", {k: raw[v].dropna(subset=["Close"]) for k, v in MACRO.items()})
    print("macro ok")


def wiki():
    u = load("universe")
    out = load("wiki") if (DATA / "wiki.pkl").exists() else {}
    end = pd.Timestamp.now().strftime("%Y%m%d")
    for i, (t, art) in enumerate(zip(u["ticker"], u["wiki"])):
        if not isinstance(art, str) or t in out:
            continue
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
               f"{requests.utils.quote(art, safe='')}/daily/20150701/{end}")
        for wait in (0, 5, 20, 60):  # back off when rate-limited
            time.sleep(wait)
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code != 429:
                break
        if not r.ok:
            print("wiki fail", t, r.status_code)
        if r.ok:
            items = r.json()["items"]
            out[t] = pd.Series([x["views"] for x in items], index=pd.to_datetime([x["timestamp"][:8] for x in items]), dtype="float32")
        time.sleep(0.25)
        if i % 100 == 0:
            print(f"wiki {i}/{len(u)}", flush=True)
    save("wiki", out)
    print(len(out), "wiki series")


def sec():
    u = load("universe")
    out = {}
    for i, (t, cik) in enumerate(zip(u["ticker"], u["cik"])):
        base = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        try:
            d = requests.get(base, headers=UA, timeout=60).json()
            frames = [pd.DataFrame(d["filings"]["recent"])]
            for f in d["filings"]["files"]:
                time.sleep(0.12)
                frames.append(pd.DataFrame(requests.get("https://data.sec.gov/submissions/" + f["name"], headers=UA, timeout=60).json()))
            f = pd.concat(frames)[["form", "items", "acceptanceDateTime", "filingDate"]]
            out[t] = f[f["filingDate"] >= "2009-01-01"].reset_index(drop=True)
        except Exception as e:
            print("sec fail", t, e)
        time.sleep(0.12)  # SEC limit is 10 requests/second
        if i % 50 == 0:
            print(f"sec {i}/{len(u)}")
    save("sec", out)
    print(len(out), "sec histories")


def events():
    import qualitative
    u = load("universe")
    out = {}
    for i, t in enumerate(u["ticker"]):
        out[t] = qualitative.events(t)
        if i % 50 == 0:
            print(f"events {i}/{len(u)}")
    save("events", out)


def fomc():
    """Scheduled Fed rate-decision days (the second day of each 2-day meeting), 2009 onward."""
    import re
    mon = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
    days = []
    for y in range(2009, 2021):
        html = requests.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm", headers=UA, timeout=60).text
        for m1, m2, m3, d2, yy in re.findall(r"([A-Z][a-z]+)(?:/([A-Z][a-z]+))? \d+-(?:([A-Z][a-z]+) )?(\d+)\*?\s+Meeting - (\d{4})", html):
            days.append(pd.Timestamp(int(yy), mon[(m3 or m2 or m1)[:3]], int(d2)))
        for m1, d1, yy in re.findall(r"([A-Z][a-z]+) (\d+)\*?\s+Meeting - (\d{4})", html):  # one-day meetings
            days.append(pd.Timestamp(int(yy), mon[m1[:3]], int(d1)))
        time.sleep(0.5)
    html = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=UA, timeout=60).text
    for block in re.split(r'<h4><a id="\d+">', html)[1:]:
        y = int(block[:4])
        months = re.findall(r'fomc-meeting__month[^>]*><strong>([^<]+)', block)
        dates = re.findall(r'fomc-meeting__date[^>]*>([^<]+)', block)
        for m, d in zip(months, dates):
            if "-" in d and "notation" not in d:
                d2 = re.match(r"\d+-(\d+)", d).group(1)
                days.append(pd.Timestamp(y, mon[m.split("/")[-1][:3]], int(d2)))
    days = sorted(set(days))
    save("fomc", days)
    print(len(days), "FOMC decision days", days[0].date(), "to", days[-1].date())


WIDE_OUT = re.compile(r"\bwarrants?\b|\brights?\b|\bunits?\b|preferred|depositary|\bnotes\b|debentures?|%|\betns?\b|\bfund\b", re.I)


def wide():
    """Liquid US common stocks (Julian, 2026-10-17): listed on a US exchange, not an ETF/test issue/warrant/right/
    unit/preferred/ADR/fund; pre-filtered on today's price > $5 and > $5M traded. The strict rule (> $5 and a
    20-day average of $20M+ a day) is applied day by day later, so history only counts a stock when it qualified."""
    lists = []
    for url, sym in (("https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt", "Symbol"),
                     ("https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt", "ACT Symbol")):
        df = pd.read_csv(io.StringIO(requests.get(url, headers=UA, timeout=60).text), sep="|", dtype=str)
        df = df[~df[sym].str.startswith("File Creation", na=True) & (df["ETF"] == "N") & (df["Test Issue"] == "N")]
        if "Financial Status" in df:
            df = df[df["Financial Status"] != "Q"]  # Q = bankrupt
        lists.append(pd.DataFrame({"symbol": df[sym], "name": df["Security Name"]}))
    L = pd.concat(lists).drop_duplicates("symbol")
    L = L[L["symbol"].str.fullmatch(r"[A-Z]+(\.[A-Z])?") & ~L["name"].str.contains(WIDE_OUT)]
    rows = requests.get("https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true", timeout=60,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh) FormulaLab personal research", "Accept": "application/json"}).json()["data"]["rows"]
    S = pd.DataFrame(rows)
    S["px"] = pd.to_numeric(S["lastsale"].str.replace("[$,]", "", regex=True), errors="coerce")
    S["dv"] = S["px"] * pd.to_numeric(S["volume"], errors="coerce")
    S = S[(S["px"] > 5) & (S["dv"] > 5e6)]
    U = L.merge(S[["symbol", "px", "dv", "sector"]], on="symbol")
    U["ticker"] = U["symbol"].str.replace(".", "-", regex=False)
    save("wide_universe", U)
    print(f"{len(L)} common stocks listed, {len(U)} pass the loose price/volume pre-filter")


def prices_wide():
    """Daily history for the wide universe (reuses the S&P 500 download), 100 tickers at a time with pauses."""
    U, have = load("wide_universe"), load("prices")
    out = {t: df for t, df in have.items()}
    todo = [t for t in U["ticker"] if t not in out]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start="2009-01-01", auto_adjust=True, group_by="ticker", progress=False, threads=8)
                break
            except Exception as e:
                print("retry", e); time.sleep(30)
        for t in chunk:
            try:
                df = raw[t].dropna(subset=["Open", "Close"])
                if len(df) > 60:
                    out[t] = df.astype("float32")
            except KeyError:
                pass
        print(f"prices_wide {min(i + 100, len(todo))}/{len(todo)}: {len(out)} total", flush=True)
        time.sleep(3)  # stay under Yahoo's rate limit
    save("prices_wide", out)


def prices_long():
    """1998–2009 for today's S&P 500 members + SPY/QQQ/VIX: a decade the daily signals were never tested on."""
    u = load("universe")
    out = {}
    tickers = list(u["ticker"]) + ["SPY", "QQQ", "^VIX"]
    for i in range(0, len(tickers), 100):
        raw = yf.download(tickers[i:i + 100], start="1998-01-01", end="2010-01-01", auto_adjust=True, group_by="ticker", progress=False, threads=8)
        for t in tickers[i:i + 100]:
            try:
                df = raw[t].dropna(subset=["Open", "Close"])
                if len(df) > 300:
                    out[t] = df.astype("float32")
            except KeyError:
                pass
        print(f"prices_long {min(i + 100, len(tickers))}/{len(tickers)}: {len(out)} ok", flush=True)
        time.sleep(3)
    save("prices_long", out)


def sec_long():
    """Same as sec() but keeps filings from 2001 on (EDGAR's full-text era) for the 2000–2009 news test."""
    u = load("universe")
    out = {}
    for i, (t, cik) in enumerate(zip(u["ticker"], u["cik"])):
        try:
            d = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=UA, timeout=60).json()
            frames = [pd.DataFrame(d["filings"]["recent"])]
            for f in d["filings"]["files"]:
                time.sleep(0.12)
                frames.append(pd.DataFrame(requests.get("https://data.sec.gov/submissions/" + f["name"], headers=UA, timeout=60).json()))
            f = pd.concat(frames)[["form", "items", "acceptanceDateTime", "filingDate"]]
            out[t] = f[(f["filingDate"] >= "2000-01-01") & (f["form"] == "8-K")].reset_index(drop=True)
        except Exception as e:
            print("sec fail", t, e)
        time.sleep(0.12)
        if i % 100 == 0:
            print(f"sec_long {i}/{len(u)}", flush=True)
    save("sec_long", out)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["universe", "prices", "macro", "wiki", "sec", "events"]
    for s in steps:
        globals()[s]()
