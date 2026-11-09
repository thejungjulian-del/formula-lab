"""How much confidence do the plan's two trades deserve? Every method that can raise OR lower it.

The unit of evidence is ONE TRADING DAY (the 10 picks on a day move together, so counting them as 10 separate
results would overstate confidence). A "cycle" = the overnight trade (buy at t's close, sell at the next open)
followed by the day trade (buy at that open, sell at that close). Costs are charged on every trade.

Sections
  A  point estimates by period, against two baselines (average S&P stock doing the same trades; holding SPY)
  B  block bootstrap: confidence ranges for average profit, yearly return, win rate, Sharpe
  C  market-neutral excess (strategy minus the average stock): the part survivorship bias can't inflate much
  D  randomization test: our picks vs 2,000 sets of 10 random stocks per day
  E  Monte Carlo account paths for $100 (6 months, 1 year), with and without the edge shrinking by half
  F  multiple testing: probabilistic and deflated Sharpe ratio (Bailey & Lopez de Prado)
  G  stability: by year, by fear level (VIX), on up vs down market days
  H  cost: break-even cost, and results at 0 / 2 / 5 / 10 bps
  I  survivorship check: the universe's average stock vs SPY on the same trades
  J  realistic timing: real fills come minutes after the open / before the close (exec_check.py measured how much
     of the average edge survives); each trade's average is lowered by the lost share, then B/E/H are redone

Run: .venv/bin/python confidence.py   -> confidence.json + printed report
"""
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import combo
import search_data as sd
import signal_search as ss

HERE = Path(__file__).parent
COST = 0.0005          # per round trip, per trade (the plan's default)
BLOCK = 10             # bootstrap block length in trading days (keeps calm/volatile stretches together)
SEED = 7
N_BOOT, N_PATHS, N_RANDOM = 10_000, 10_000, 2_000
EULER = 0.5772156649015329
N01 = NormalDist()
OTHER_TRIALS = 87 + 36 + 12 + 2  # shortterm setups + daily_model (18 features x 2 holds) + qualitative (6 x 2) + 2 combos


def block_bootstrap_idx(n, size, rng, block=BLOCK):
    """Circular block bootstrap: `size` row-index arrays of length n made of runs of `block` consecutive days."""
    starts = rng.integers(0, n, size=(size, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)) % n
    return idx.reshape(size, -1)[:, :n]


def psr(sr, t, skew, kurt, sr0=0.0):
    """Probabilistic Sharpe ratio: chance the TRUE per-period Sharpe exceeds sr0, allowing for fat tails and skew."""
    return N01.cdf((sr - sr0) * np.sqrt(t - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))


def expected_max_sr(n_trials, t):
    """Expected best Sharpe among n_trials strategies with NO real edge (each estimate ~ N(0, 1/t))."""
    return np.sqrt(1 / t) * ((1 - EULER) * N01.inv_cdf(1 - 1 / n_trials) + EULER * N01.inv_cdf(1 - 1 / (n_trials * np.e)))


def stats_of(r):
    r = np.asarray(r)
    eq = np.cumprod(1 + r)
    return {"mean": float(r.mean()), "sd": float(r.std(ddof=1)), "win": float((r > 0).mean()),
            "sharpe_yr": float(r.mean() / r.std(ddof=1) * np.sqrt(252)), "per_year": float(eq[-1] ** (252 / len(r)) - 1),
            "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min()), "days": int(len(r))}


def daily_portfolios(p, k=10):
    """Per trading day (row t): top-k combo picks and the average stock, for both trades."""
    C = json.loads((HERE / "combo.json").read_text())["trades"]
    p = p.assign(day=combo.score(p, C["h0"]["components"]), night=combo.score(p, C["ovn"]["components"]))
    out = {}
    for name, sc, y in (("day", "day", "yh0"), ("night", "night", "yovn")):
        d = p[[sc, y]].dropna()
        d = d.assign(dt=d.index)
        top = d.sort_values(sc).groupby("dt").tail(k).groupby("dt")[y].mean()
        avg = d.groupby("dt")[y].mean()
        out[name], out[name + "_avg"] = top, avg
    df = pd.DataFrame(out).dropna()
    df["cycle_gross"] = (1 + df["night"]) * (1 + df["day"]) - 1
    df["avg_cycle_gross"] = (1 + df["night_avg"]) * (1 + df["day_avg"]) - 1
    return df


def random_picks_means(p, dates, y, rng, n=N_RANDOM, k=10):
    """Mean daily return of n sets of k random stocks per day (no skill), for the given dates."""
    tot = np.zeros(n)
    g = p.loc[p.index.isin(dates), y].dropna().groupby(level=0)
    for _, vals in g:
        v = vals.to_numpy()
        tot += v[rng.integers(0, len(v), size=(n, k))].mean(axis=1)
    return tot / g.ngroups


def main():
    rng = np.random.default_rng(SEED)
    p = ss.build()
    D = daily_portfolios(p)
    spy = sd.load("macro")["spy"]
    rep = {"cost": COST, "block": BLOCK, "seed": SEED}
    net = lambda df, c=COST: (1 + df["night"] - c) * (1 + df["day"] - c) - 1
    avg_net = lambda df, c=COST: (1 + df["night_avg"] - c) * (1 + df["day_avg"] - c) - 1
    periods = {"2019-22": (D.index >= ss.CONFIRM) & (D.index < ss.HOLDOUT), "2023-now": D.index >= ss.HOLDOUT}

    # A. point estimates
    rep["A"] = {}
    for per, m in periods.items():
        d = D[m]
        s = spy["Close"][(spy.index >= d.index.min()) & (spy.index <= d.index.max())]
        sp = s.pct_change().dropna()
        rep["A"][per] = {"strategy": stats_of(net(d)), "avg_stock_same_trades": stats_of(avg_net(d)), "hold_spy": stats_of(sp),
                         "day_only": stats_of(d["day"] - COST), "night_only": stats_of(d["night"] - COST)}

    H = D[periods["2023-now"]]
    r = net(H).to_numpy()
    ex = (net(H) - avg_net(H)).to_numpy()
    n = len(r)

    # B + C. block bootstrap
    idx = block_bootstrap_idx(n, N_BOOT, rng)
    R, X = r[idx], ex[idx]
    yearly = np.prod(1 + R, axis=1) ** (252 / n) - 1
    sharpe = R.mean(1) / R.std(1, ddof=1) * np.sqrt(252)
    q = lambda a: [float(x) for x in np.quantile(a, [0.05, 0.5, 0.95])]
    rep["B"] = {"mean_per_day_90": q(R.mean(1)), "per_year_90": q(yearly), "win_rate_90": q((R > 0).mean(1)), "sharpe_90": q(sharpe),
                "p_mean_above_0": float((R.mean(1) > 0).mean()), "p_beats_spy_year": float((yearly > rep["A"]["2023-now"]["hold_spy"]["per_year"]).mean())}
    rep["C"] = {"excess_per_day_90": q(X.mean(1)), "p_excess_above_0": float((X.mean(1) > 0).mean()),
                "excess_per_year": float(np.prod(1 + ex) ** (252 / n) - 1)}

    # D. randomization test (each trade separately, before costs: costs are the same for random picks)
    rep["D"] = {}
    for name, y in (("day", "yh0"), ("night", "yovn")):
        rnd = random_picks_means(p, H.index, y, rng)
        actual = float(H[name].mean())
        rep["D"][name] = {"ours": actual, "random_90": q(rnd), "share_random_as_good": float((rnd >= actual).mean())}

    # E. Monte Carlo account paths for $100
    rep["E"] = {}
    shrink = 0.5 * ex.mean()          # half of the measured edge disappears (published edges shrink 26–58%)
    for label, series in (("as measured", r), ("edge cut in half", r - shrink)):
        for days, horizon in ((126, "6 months"), (252, "1 year")):
            ii = block_bootstrap_idx(n, N_PATHS, rng)[:, :days] if days <= n else None
            paths = 100 * np.cumprod(1 + series[ii], axis=1)
            end, dd = paths[:, -1], (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
            rep["E"][f"{label} | {horizon}"] = {"end_5_25_50_75_95": [float(x) for x in np.quantile(end, [.05, .25, .5, .75, .95])],
                                                 "p_loss": float((end < 100).mean()), "p_10000": float((end >= 10_000).mean()),
                                                 "worst_drop_median": float(np.median(dd)), "worst_drop_bad_case": float(np.quantile(dd, 0.05))}

    # F. multiple testing
    sr = r.mean() / r.std(ddof=1)
    sk = float(pd.Series(r).skew())
    ku = float(pd.Series(r).kurt() + 3)  # pandas gives EXCESS kurtosis; the formula needs plain kurtosis
    ledger = len(json.loads((HERE / "search_ledger.json").read_text())["tests"])
    looks_at_2023 = len(json.loads((HERE / "promoted.json").read_text())["promoted"]) + 2  # 18 singles + 2 combos
    rep["F"] = {"daily_sharpe": float(sr), "skew": sk, "kurtosis": ku, "days": n,
                "psr_vs_0": float(psr(sr, n, sk, ku)),
                "dsr_looks_at_2023": float(psr(sr, n, sk, ku, expected_max_sr(looks_at_2023, n))), "n_looks_2023": looks_at_2023,
                "dsr_all_trials": float(psr(sr, n, sk, ku, expected_max_sr(ledger + OTHER_TRIALS, n))), "n_all_trials": ledger + OTHER_TRIALS}

    # G. stability
    s = pd.Series(r, index=H.index)
    x = pd.Series(ex, index=H.index)
    vix = sd.load("macro")["vix"]["Close"].reindex(H.index, method="ffill")
    mkt = H["day_avg"]
    allr, allx = net(D), net(D) - avg_net(D)
    rep["G"] = {"by_year": {str(yy): {"per_day": float(allr[allr.index.year == yy].mean()), "excess": float(allx[allx.index.year == yy].mean()),
                                      "days": int((allr.index.year == yy).sum())} for yy in range(2019, D.index.max().year + 1)},
                "months_up": float((s.groupby(s.index.to_period("M")).apply(lambda v: np.prod(1 + v) - 1) > 0).mean()),
                "vix_low": {"per_day": float(s[vix < 20].mean()), "excess": float(x[vix < 20].mean()), "days": int((vix < 20).sum())},
                "vix_high": {"per_day": float(s[vix >= 20].mean()), "excess": float(x[vix >= 20].mean()), "days": int((vix >= 20).sum())},
                "market_up_days": {"excess": float(x[mkt > 0].mean())}, "market_down_days": {"excess": float(x[mkt <= 0].mean())}}

    # H. costs
    rep["H"] = {"break_even_cost_day": float(H["day"].mean()), "break_even_cost_night": float(H["night"].mean()),
                "by_cost": {f"{c * 1e4:.0f} bps": stats_of(net(H, c)) for c in (0, 0.0002, 0.0005, 0.0010)}}

    # I. survivorship check: our universe's average stock vs SPY itself on the same two trades
    so, sc = spy["Open"], spy["Close"]
    spy_cycle = ((so.shift(-1) / sc) * (sc.shift(-1) / so.shift(-1)) - 1).reindex(H.index)
    rep["I"] = {"avg_stock_cycle": float(H["avg_cycle_gross"].mean()), "spy_cycle": float(spy_cycle.mean()),
                "gap_per_day": float(H["avg_cycle_gross"].mean() - spy_cycle.mean())}

    # J. realistic timing
    rep["J"] = {}
    for label, f in (("on time (6:35 AM / last minute)", "exec_check_0935_1600.json"), ("a few minutes late (6:40 AM / 12:55 PM)", "exec_check_0940_1555.json")):
        ex = json.loads((HERE / f).read_text())
        kept = {k: max(0.0, min(1.0, ex[k]["top10"]["real"] / ex[k]["top10"]["official"])) for k in ("day", "night")}
        Hr = H.assign(day=H["day"] - (1 - kept["day"]) * H["day"].mean(), night=H["night"] - (1 - kept["night"]) * H["night"].mean())
        rr = net(Hr).to_numpy()
        RR = rr[block_bootstrap_idx(n, N_BOOT, rng)]
        paths = 100 * np.cumprod(1 + rr[block_bootstrap_idx(n, N_PATHS, rng)[:, :126]], axis=1)[:, -1]
        rep["J"][label] = {"kept": kept, "days_measured": ex["days"], "strategy": stats_of(rr),
                           "p_mean_above_0": float((RR.mean(1) > 0).mean()), "per_year_90": q(np.prod(1 + RR, axis=1) ** (252 / n) - 1),
                           "six_months_5_50_95": [float(x) for x in np.quantile(paths, [.05, .5, .95])], "six_months_p_loss": float((paths < 100).mean()),
                           "break_even_cost": float((Hr["day"].mean() + Hr["night"].mean()) / 2)}
    # K. the plan as actually sent to Alpaca at $100: 5 picks per leg, each leg on HALF the budget (so sale money
    #    never funds the other leg the same day), realistic timing applied to each leg's average
    D5 = daily_portfolios(p, k=5)
    H5 = D5[D5.index >= ss.HOLDOUT]
    ex = json.loads((HERE / "exec_check_0935_1600.json").read_text())
    kept = {k: max(0.0, min(1.0, ex[k]["top10"]["real"] / ex[k]["top10"]["official"])) for k in ("day", "night")}
    day_r = H5["day"] - (1 - kept["day"]) * H5["day"].mean() - COST
    night_r = H5["night"] - (1 - kept["night"]) * H5["night"].mean() - COST
    acct = (0.5 * day_r + 0.5 * night_r).to_numpy()   # account return per day: half the money in each leg
    n5 = len(acct)
    RR = acct[block_bootstrap_idx(n5, N_BOOT, rng)]
    paths = 100 * np.cumprod(1 + acct[block_bootstrap_idx(n5, N_PATHS, rng)[:, :126]], axis=1)[:, -1]
    rep["K"] = {"kept": kept, "strategy": stats_of(acct), "p_mean_above_0": float((RR.mean(1) > 0).mean()),
                "per_year_90": q(np.prod(1 + RR, axis=1) ** (252 / n5) - 1),
                "six_months_5_50_95": [float(x) for x in np.quantile(paths, [.05, .5, .95])], "six_months_p_loss": float((paths < 100).mean())}
    (HERE / "confidence.json").write_text(json.dumps(rep, indent=1))
    return rep


def show(rep):
    pc = lambda v, d=3: f"{v * 100:+.{d}f}%"
    print("A. POINT ESTIMATES (per full day = overnight + day trade, after 0.05% per trade)")
    for per, a in rep["A"].items():
        for k, v in a.items():
            print(f"  {per} {k:22s} per day {pc(v['mean'])}  per year {pc(v['per_year'], 1)}  win {v['win'] * 100:.1f}%  Sharpe {v['sharpe_yr']:.2f}  worst drop {pc(v['max_drawdown'], 1)}")
    b, c = rep["B"], rep["C"]
    print("B. BOOTSTRAP 90% RANGES (2023-now)")
    print(f"  per day {[pc(v) for v in b['mean_per_day_90']]}  per year {[pc(v, 1) for v in b['per_year_90']]}  win {[f'{v * 100:.1f}%' for v in b['win_rate_90']]}  Sharpe {[round(v, 2) for v in b['sharpe_90']]}")
    print(f"  chance true average > 0: {b['p_mean_above_0'] * 100:.1f}%   chance it beats holding SPY (same years): {b['p_beats_spy_year'] * 100:.1f}%")
    print(f"C. EXCESS over the average stock: per day {[pc(v) for v in c['excess_per_day_90']]}  chance > 0: {c['p_excess_above_0'] * 100:.1f}%  ≈ {pc(c['excess_per_year'], 1)}/yr")
    print("D. RANDOM-PICKS TEST")
    for k, v in rep["D"].items():
        print(f"  {k}: ours {pc(v['ours'])}/trade vs random 90% range {[pc(x) for x in v['random_90']]}; random sets as good: {v['share_random_as_good'] * 100:.2f}%")
    print("E. MONTE CARLO, $100")
    for k, v in rep["E"].items():
        e = v["end_5_25_50_75_95"]
        print(f"  {k:28s} 5%/25%/50%/75%/95%: " + " / ".join(f"${x:,.0f}" for x in e) + f"  loss {v['p_loss'] * 100:.0f}%  reach $10k {v['p_10000'] * 100:.2f}%  worst drop median {pc(v['worst_drop_median'], 0)} bad {pc(v['worst_drop_bad_case'], 0)}")
    f = rep["F"]
    print(f"F. SHARPE TESTS: daily SR {f['daily_sharpe']:.3f} (skew {f['skew']:.2f}, kurtosis {f['kurtosis']:.1f}, {f['days']} days)")
    print(f"  PSR(>0) {f['psr_vs_0'] * 100:.1f}%   deflated for {f['n_looks_2023']} looks at 2023+: {f['dsr_looks_at_2023'] * 100:.1f}%   deflated for all {f['n_all_trials']} ideas: {f['dsr_all_trials'] * 100:.1f}%")
    g = rep["G"]
    print("G. STABILITY  " + "  ".join(f"{y}: {pc(v['per_day'])} (excess {pc(v['excess'])})" for y, v in g["by_year"].items()))
    print(f"  months up {g['months_up'] * 100:.0f}%  VIX<20 {pc(g['vix_low']['per_day'])} ({g['vix_low']['days']}d)  VIX>=20 {pc(g['vix_high']['per_day'])} ({g['vix_high']['days']}d)  "
          f"excess on market-up days {pc(g['market_up_days']['excess'])}, down days {pc(g['market_down_days']['excess'])}")
    h = rep["H"]
    print(f"H. BREAK-EVEN COST: day trade {pc(h['break_even_cost_day'])}, overnight {pc(h['break_even_cost_night'])} per trade")
    for k, v in h["by_cost"].items():
        print(f"  {k}: per day {pc(v['mean'])}  per year {pc(v['per_year'], 1)}  win {v['win'] * 100:.1f}%")
    i = rep["I"]
    print(f"I. SURVIVORSHIP: average stock {pc(i['avg_stock_cycle'])}/day vs SPY {pc(i['spy_cycle'])}/day on the same trades -> gap {pc(i['gap_per_day'])}/day")
    k = rep["K"]
    print(f"K. THE $100 PLAN (5 picks/leg, half the budget per leg, on time, 0.05% costs): per year {pc(k['strategy']['per_year'], 1)} "
          f"(90%: {pc(k['per_year_90'][0], 0)} to {pc(k['per_year_90'][2], 0)}), chance > 0: {k['p_mean_above_0'] * 100:.0f}%, "
          f"$100 in 6 months ${k['six_months_5_50_95'][0]:.0f} / ${k['six_months_5_50_95'][1]:.0f} / ${k['six_months_5_50_95'][2]:.0f} "
          f"(loss {k['six_months_p_loss'] * 100:.0f}%), worst drop {pc(k['strategy']['max_drawdown'], 0)}")
    print("J. REALISTIC TIMING (after 0.05% per trade)")
    for k, v in rep["J"].items():
        st = v["strategy"]
        print(f"  {k}: keeps day {v['kept']['day'] * 100:.0f}% / night {v['kept']['night'] * 100:.0f}% ({v['days_measured']} days measured) -> per day {pc(st['mean'])}, "
              f"per year {pc(st['per_year'], 1)} (90%: {pc(v['per_year_90'][0], 0)} to {pc(v['per_year_90'][2], 0)}), chance > 0: {v['p_mean_above_0'] * 100:.0f}%, "
              f"$100 in 6 months: ${v['six_months_5_50_95'][0]:.0f} / ${v['six_months_5_50_95'][1]:.0f} / ${v['six_months_5_50_95'][2]:.0f} (loss {v['six_months_p_loss'] * 100:.0f}%), "
              f"break-even cost {pc(v['break_even_cost'])}/trade")


if __name__ == "__main__":
    show(main())
