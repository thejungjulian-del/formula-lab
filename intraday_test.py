"""Can a 15-minute trade make money? First look with the 5-minute bars already saved (S&P 500, ~60 days).

PRE-REGISTERED (written before the first run):
  Trade     at the end of a 5-minute bar (time T), buy; sell 15 minutes later (T+15). Entries 9:35–15:40 ET, so
            every trade closes by 15:55. Cost 0.05% per round trip (also shown at 0.02%).
  Signals   known at T, ranked across stocks at each T (cross-sectional), in units of the stock's own 5-min swing:
              rev_5     last 5-minute move          (expect: reversal, negative)
              rev_15    last 15-minute move         (reversal)
              mom_30    last 30-minute move         (momentum, positive)
              day_ret   move since today's open     (intraday momentum)
              vwap_dev  price vs today's volume-weighted average price (reversion)
              range_pos where the price sits in today's high–low range so far
              vol_spike this bar's volume vs its last 20 bars
  Split     first half of the days = SEARCH, second half = CONFIRM (no holdout: only ~60 days exist).
  Promote   |t| >= 3 in SEARCH and same sign |t| >= 2 in CONFIRM (t over 5-minute timestamps, Newey-West 3 lags,
            because consecutive 15-minute windows overlap).
  Money     for each signal: buy the top 10 stocks every 15 minutes; average result per trade after costs.

Run: .venv/bin/python intraday_test.py
"""
import numpy as np
import pandas as pd

import signal_search as ss

COST = 0.0005


def features(b):
    """b: one stock's 5-minute closes + volumes over many days (ET index). One row per bar end."""
    c, v = b["Close"], b["Volume"]
    day = c.index.normalize()
    r1 = c.groupby(day).pct_change()
    sd = r1.groupby(day).transform(lambda x: x.expanding().std()).shift(0).fillna(r1.abs().mean())
    sd = sd.where(sd > 0, r1.abs().mean())
    first = c.groupby(day).transform("first")
    pv = (c * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    hi, lo = c.groupby(day).cummax(), c.groupby(day).cummin()
    f = pd.DataFrame({
        "rev_5": r1 / sd, "rev_15": c.groupby(day).pct_change(3) / sd, "mom_30": c.groupby(day).pct_change(6) / sd,
        "day_ret": (c / first - 1) / sd, "vwap_dev": (c / pv - 1) / sd, "range_pos": (c - lo) / (hi - lo) - 0.5,
        "vol_spike": np.log(v / v.rolling(20).mean()),
        "y": c.groupby(day).shift(-3) / c - 1,          # 15 minutes later, same day only
    })
    hm = c.index.hour * 60 + c.index.minute
    return f[(hm >= 575) & (hm <= 940)].replace([np.inf, -np.inf], np.nan)  # bar ends 9:35..15:40 (starts 9:30..15:35)


def main():
    B = pd.read_pickle(ss.HERE / "data" / "bars5m.pkl")  # saved closes only: used for the ticker list; volume needs a fresh download
    import yfinance as yf
    yf.set_tz_cache_location(__import__("tempfile").mkdtemp(prefix="yf-tz-"))
    rows = []
    tickers = sorted(B)
    for i in range(0, len(tickers), 50):
        raw = yf.download(tickers[i:i + 50], period="60d", interval="5m", group_by="ticker", progress=False, threads=8)
        for t in tickers[i:i + 50]:
            try:
                b = raw[t][["Close", "Volume"]].dropna()
            except KeyError:
                continue
            if len(b) < 500:
                continue
            b.index = b.index.tz_convert("America/New_York")
            b.index = b.index + pd.Timedelta(minutes=5)   # label each bar by its END: the price is known then
            f = features(b)
            f["ticker"] = t
            rows.append(f)
    p = pd.concat(rows).dropna()
    days = sorted(p.index.normalize().unique())
    cut = days[len(days) // 2]
    names = ["rev_5", "rev_15", "mom_30", "day_ret", "vwap_dev", "range_pos", "vol_spike"]
    print(f"{p['ticker'].nunique()} stocks, {len(days)} days ({days[0].date()} to {days[-1].date()}), {p.index.nunique()} five-minute times")
    out = []
    for n in names:
        res, sign = {}, None
        for per, m in (("search", p.index < cut), ("confirm", p.index >= cut)):
            d = p[m]
            r = ss.fm(d[n], d["y"], d.index, lags=3)
            sign = sign if sign is not None else (1 if r and r[1] > 0 else -1)   # direction fixed by SEARCH only
            top = d.assign(k=sign * d[n]).sort_values("k").groupby(level=0).tail(10).groupby(level=0)["y"].mean()
            res[per] = (r[1] if r else None, float(top.mean()))
        st, ct = res["search"][0], res["confirm"][0]
        ok = st is not None and ct is not None and abs(st) >= 3 and abs(ct) >= 2 and np.sign(st) == np.sign(ct)
        out.append((n, st, ct, res["confirm"][1], ok))
    print(f"{'signal':10s} {'search t':>9s} {'confirm t':>10s} {'top-10 per trade (confirm)':>28s} {'after 0.05%':>12s} {'after 0.02%':>12s}")
    for n, st, ct, g, ok in out:
        print(f"{n:10s} {st:+9.2f} {ct:+10.2f} {g * 100:+27.3f}% {(g - COST) * 100:+11.3f}% {(g - 0.0002) * 100:+11.3f}%  {'PROMOTED' if ok else ''}")
    avg = p[p.index >= cut]["y"].mean()
    print(f"average stock, 15 minutes: {avg * 100:+.4f}% before costs")


if __name__ == "__main__":
    main()
