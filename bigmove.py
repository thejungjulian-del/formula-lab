"""Which stocks are about to make a HUGE move (either direction)? Magnitude, not direction.

PRE-REGISTERED (written before the first run):
  Target    "huge move" = tomorrow's close vs today's close moves 5% or more, up OR down.
            Secondary: 3x or more the stock's normal daily move (20-day).
  Inputs    only things known at today's close, so you could act before the move (e.g. buy options at 3:50 PM):
            scheduled earnings before tomorrow's close, recent volatility, today's absolute move / gap / range,
            volume surprise, VIX level and change, size, price level, trend position, Fed day, 8-K filed today.
  Model     linear probability model (OLS) fit on 2010–2018 only, scored on 2019–2022, then ONCE on 2023–today.
  Report    base rate, hit rate and lift in the top 1% and 5% of each day's ranking, AUC, and the direction
            split (share up) among the flagged stocks. Compared with "earnings tomorrow" alone.
  Same universe/caveats as signal_search.py (today's S&P 500 members).

Run: .venv/bin/python bigmove.py
"""
import numpy as np
import pandas as pd

import search_data as sd
import signal_search as ss

INPUTS = ["earn_next", "volatility", "abs_ret_1d", "abs_overnight", "range", "vol_surprise", "vix_level", "vix_chg",
          "dollar_vol", "price_level", "near_high", "dist_ma200", "fomc_day", "sec_8k_day"]


def earn_next(idx, ed):
    """1 if a scheduled earnings report falls after today's close and before tomorrow's close (calendar-known)."""
    out = np.zeros(len(idx))
    if ed is None or not len(ed):
        return out
    t = pd.DatetimeIndex(ed.index).tz_convert("America/New_York").tz_localize(None)
    ext = idx.append(pd.DatetimeIndex([idx[-1] + pd.offsets.BDay(1)]))
    k = (ext + pd.Timedelta(hours=16)).searchsorted(t, side="right")  # first close after the report = day D
    rows = k - 1
    out[rows[(rows >= 0) & (rows < len(idx))]] = 1
    return out


def auc(score, y):
    r = pd.Series(score).rank().to_numpy()
    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    p = ss.build()
    events = sd.load("events")
    p["earn_next"] = 0.0
    for tk, g in p.groupby("ticker"):
        p.loc[p["ticker"] == tk, "earn_next"] = earn_next(g.index, events.get(tk, (None, None))[1])
    p["abs_ret_1d"], p["abs_overnight"] = p["ret_1d"].abs(), p["overnight"].abs()
    move = (1 + p["yovn"]) * (1 + p["yh0"]) - 1                      # tomorrow close vs today close
    normal = np.exp(p["volatility"])
    p["big5"], p["big3x"], p["move"] = (move.abs() >= 0.05).astype(float), (move.abs() >= 3 * normal).astype(float), move
    p = p.dropna(subset=INPUTS + ["move"])
    srch, conf, ho = p.index < ss.CONFIRM, (p.index >= ss.CONFIRM) & (p.index < ss.HOLDOUT), p.index >= ss.HOLDOUT
    m, s = p.loc[srch, INPUTS].mean(), p.loc[srch, INPUTS].std()
    X = np.column_stack([np.ones(len(p)), ((p[INPUTS] - m) / s).clip(-5, 5).to_numpy()])
    for target in ("big5", "big3x"):
        y = p[target].to_numpy()
        b = np.linalg.lstsq(X[srch], y[srch], rcond=None)[0]
        p["score"] = X @ b
        print(f"\n=== {target}: {'moves of 5%+' if target == 'big5' else 'moves of 3x normal+'} ===")
        print("weights (per 1 std):", {n: round(w * 100, 2) for n, w in zip(INPUTS, b[1:]) if abs(w) * 100 >= 0.1})
        for name, mask in (("2010-18 (fit)", srch), ("2019-22 (check)", conf), ("2023-now (final)", ho)):
            d = p[mask]
            rk = d.groupby(level=0)["score"].rank(pct=True, ascending=False)
            base = d[target].mean()
            top1, top5 = d[rk <= 0.01], d[rk <= 0.05]
            earn = d[d["earn_next"] == 1]
            print(f"{name:17s} base {base * 100:4.1f}% | top 1%: {top1[target].mean() * 100:5.1f}% ({top1[target].mean() / base:4.1f}x) "
                  f"| top 5%: {top5[target].mean() * 100:5.1f}% ({top5[target].mean() / base:4.1f}x) | AUC {auc(d['score'].to_numpy(), d[target].to_numpy()):.3f} "
                  f"| earnings-only: {earn[target].mean() * 100:5.1f}% of {len(earn)} | top 1% up {(top1['move'] > 0).mean() * 100:.0f}%, "
                  f"avg {top1['move'].mean() * 100:+.2f}%, avg size {top1['move'].abs().mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
