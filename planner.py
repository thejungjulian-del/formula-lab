"""Today's plan for EVERY ranked stock, in broker.py's format (plus keys the broker ignores).

Two legs, each from a combined score that passed its test (combo.py):
  day    buy at the open (+5 min: picks need the opening gap), sell at that day's close   -> follow 6:31–6:34 AM PT
  night  buy just before the close, sell at the next open                                -> follow 12:50–12:57 PM PT
Day leg: only stocks with NO news overnight (no earnings report, no 8-K between the previous close and the open);
research2.py D3 passed in 2010–22 (t 3.9) and in the untouched 2000–09 (t 4.9). Unchecked = skipped.
Each leg gets half the budget, 5 picks x 10% ($10 each at $100), so one leg's sale money never has to fund the
other leg's purchase the same day (cash-account settlement) and the plan never sums past 100%.
The bottom 5 of each leg are listed as shorts so they stay visible; the broker skips them ("needs a $2,000
margin account"). Every other stock is listed in "all" with side "none".

Writes plan/<session date>.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
# 2026-10-17 exec7.py (7 years, ~9,000 real fills): the day leg keeps only 17% of its edge when bought at 9:35 (the
# bounce happens in the opening auction / first minute), so after costs it loses money -> PAUSED. The night leg
# keeps 80% (sold at 9:31) -> it gets the whole budget, 10 stocks x 10%.
PAUSED = {"day": "Paused: over 7 years, buying at 9:35 kept only 17% of the edge (the bounce happens in the first "
                 "minute), so after costs it lost money. It needs the official opening price, which $10 orders can't get."}
PICKS = {"day": 5, "night": 10}
WEIGHT = {"day": 0.10, "night": 0.10}
LEGS = {
    "day": {"score": "day", "enter": "open", "enter_delay_min": 5, "hold_days": 0, "exit": "close"},
    "night": {"score": "night", "enter": "close", "hold_days": 1, "exit": "open"},
}


def why(leg, r, rank, n):
    if leg == "day":
        return f"Day trade #{rank} of {n} (no overnight news): opened {r['gap'] * 100:+.1f}% ({r['gap_rel']:+.1f} normal moves vs the market's gap)"
    return f"Overnight #{rank} of {n}: volume {2.718281828 ** r['volume']:.1f}x normal, ranked on volume and overnight strength"


def build(scan, excluded=()):
    """scan = a research/<date>.json; excluded = tickers the health check flagged. Legs appear only when the
    ranking is from a live session (the day leg needs today's open, the night leg today's trading)."""
    P = scan.get("passed") or {}
    rows = [r for r in P.get("all", []) if r["ticker"] not in excluded]
    n, live = len(rows), bool(P.get("live"))
    trades, ranks = [], {r["ticker"]: {} for r in rows}
    for leg, spec in LEGS.items():
        order = sorted(rows, key=lambda r: -r[spec["score"]])
        for i, r in enumerate(order):
            ranks[r["ticker"]][leg] = i + 1
        if not live or leg in PAUSED:
            if leg == "day":
                skipped_news = []
            continue
        base = {k: v for k, v in spec.items() if k != "score"}
        n_pick, w = PICKS[leg], WEIGHT[leg]
        buyable = [r for r in order if r.get("news") is False] if leg == "day" else order
        if leg == "day":  # higher-ranked candidates passed over, and why
            cutoff = order.index(buyable[min(n_pick, len(buyable)) - 1]) + 1 if buyable else len(order)
            skipped_news = [{"symbol": r["ticker"], "why": "news overnight" if r.get("news") else "news not checked"}
                            for r in order[:cutoff] if r.get("news") is not False][:10]
        for i, r in enumerate(buyable[:n_pick]):
            trades.append({"symbol": r["ticker"], "side": "long", "weight": w, "leg": leg, "price": r["price"], **base,
                           "why": why(leg, r, i + 1, n)})
        for j, r in enumerate(order[-5:]):
            trades.append({"symbol": r["ticker"], "side": "short", "weight": w, "leg": leg, "price": r["price"], **base,
                           "why": f"Bottom {5 - j} of {n} for this leg. Not tradeable at $100 (shorting needs a $2,000 margin "
                                  "account), and the short side was not tested on its own."})
    picked = {(t["symbol"], t["leg"]): t["side"] for t in trades}
    return {"date": P.get("date"), "live": live, "label": "tested", "skipped_news": skipped_news if live else [], "paused": PAUSED,
            "note": "Overnight only: 10 stocks x 10%, bought just before the close, sold at the next open. Passed its tests; "
                    "with real fill times measured over 7 years, about +7.7% a year on 2023–today data. Paper-trade first.",
            "trades": trades,
            "all": [{"symbol": t, "day_rank": k.get("day"), "night_rank": k.get("night"),
                     "day": picked.get((t, "day"), "none"), "night": picked.get((t, "night"), "none")} for t, k in ranks.items()]}


def write(plan):
    if not plan.get("date"):
        return None
    d = HERE / "plan"
    d.mkdir(exist_ok=True)
    f = d / f"{plan['date']}.json"
    f.write_text(json.dumps(plan, indent=1))
    return f
