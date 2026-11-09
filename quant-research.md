# What makes quant traders profitable — and is crypto easier to forecast?

Researched 2026-10-17. Every number below links to its source. Things I could not verify are listed at the end and were left out on purpose.

---

## The short answer

1. **Quants don't win by predicting one stock well.** They win with *tiny* edges spread over thousands of bets, cheap execution, and very strict testing. Being right 52% of the time is normal for a working signal.
2. **The signals that keep surviving testing:** momentum, short-term reversal, value, profitability, low asset growth, low risk, and liquidity. They work best when you **rank many stocks against each other** over **weeks to months**, not when you call one ticker's next day.
3. **Most published signals shrink once people use them:** about −26% out-of-sample, −58% after publication. Signals that trade a lot mostly die after costs.
4. **Crypto is not easier to forecast.** The published crypto edges were measured mostly on 2014–2018 data and have faded since about 2020. My own check agrees. Crypto is 2–4× more volatile, fees for a small account are far higher, and much of the reported volume is fake. **Use bitcoin as an input to stock formulas, not as the main thing you forecast.**

---

## 1. How quant firms actually make money

### Small edge × many bets
The core rule is Grinold's **fundamental law of active management**: skill per bet × √(number of independent bets) ≈ risk-adjusted return ([CFI](https://corporatefinanceinstitute.com/resources/capital-markets/fundamental-law-of-active-management/), [AnalystPrep](https://analystprep.com/study-notes/?p=19180)).
- Example: the same weak skill applied to 1 stock once a day gives √252 ≈ 16 bets a year. Applied to 500 stocks, that becomes ≈ 355. That's about 22× more "breadth" for the same skill.
- Caveat: the law assumes the bets are independent, and real bets are not, so it overstates what's reachable ([AnalystPrep](https://analystprep.com/study-notes/?p=19180)).

### A real look inside: WorldQuant's "101 Formulaic Alphas"
WorldQuant published 101 formulas its traders actually used ([arXiv 1601.00991](https://arxiv.org/abs/1601.00991)):
- **Average holding period: 0.6–6.4 days.** This is the same horizon Formula Lab works at.
- **Average correlation between alphas: 15.9%.** They are many *different* weak signals, not one strong one.
- Their returns track volatility, not turnover.
- They are price/volume formulas built mostly on mean reversion and momentum ([summary](https://hedgefundalpha.com/strategies/101-formulaic-alphas/)).

### The famous result: Renaissance Medallion
- Reported returns are 66.1% a year before fees and 39.1% after fees, 1988–2018 ([Zuckerman 2019, via Guo & Liu](https://arxiv.org/pdf/2405.10917v2)).
- A later paper argues these are simple averages, and the compounded rate is lower (~63% gross per Cornell 2020) ([Guo & Liu](https://arxiv.org/pdf/2405.10917v2)).
- The fund is closed to outsiders, and its actual signals are secret. Anyone who claims to know them is guessing.

### Costs decide which signals live
- Most anomalies with **under 50% monthly one-sided turnover** survive trading costs. **Few higher-turnover ones do.** The cheapest signals to run are value, profitability, investment and size ([Novy-Marx & Velikov 2016](https://www.nber.org/papers/w20721)).
- Short-term reversal *can* survive costs, but only in **large caps** with turnover control: about 30–50 bps/week net ([de Groot, Huij & Zhou 2012](https://ideas.repec.org/a/eee/jbfina/v36y2012i2p371-382.html)).
- Reversal profits are basically **getting paid to provide liquidity**. They are much bigger when the **VIX is high** ([Nagel 2012](https://nber.org/papers/w17653)).

### Testing discipline
- A new signal should clear **t > 3.0**, not 2.0, because hundreds of signals have been tried ([Harvey, Liu & Zhu 2016](https://nber.org/papers/w20592)).
- AQR's century-long study found out-of-sample returns **~30% lower** than in-sample, mostly from overfitting ([AQR](https://www.aqr.com/Insights/Research/Working-Paper/Factor-Premia-and-Factor-Timing-A-Century-of-Evidence)). AQR often assumes future premia will be **half or less** of history ([Ilmanen/AQR](https://www.nomura.com/events/10th-annual-global-quantitative-investment-strategies-conference/resources/upload/Style_Premia_Ilmanen.pdf)).

### Day-trading reality check
- Taiwan: fewer than 3% of day traders were persistently profitable, so about 97% can expect to lose ([Barber, Lee, Liu, Odean](https://faculty.haas.berkeley.edu/odean/papers/day%20traders/Day%20Trading%20Skill%20110523.pdf)).
- Brazil: 97% of people who day-traded futures for 300+ days lost money. Only 0.4% earned more than a bank teller, and there was no evidence they improved with practice ([Chague et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101)).

---

## 2. The signals with the strongest evidence

| Signal | Plain meaning | Horizon | Evidence | In Formula Lab? |
|---|---|---|---|---|
| **Momentum (12-1 mo)** | Last year's winners keep winning (skipping the latest month) | 1–12 mo | Replicates ([Hou-Xue-Zhang](https://systematicallybiased.substack.com/p/pockets-of-replicability-post5)). Top-ranked predictor in machine-learning work ([Gu-Kelly-Xiu](https://www.nber.org/papers/w25398)). **Crash risk:** April 2009 = −45.5% in one month; Mar–May 2009 past losers +163% vs winners +8% ([Daniel & Moskowitz](https://www.nber.org/papers/w20439)) | `mom_12_1` ✓ |
| **Short-term reversal** | Last week/month's losers bounce | days–1 mo | #1 machine-learning predictor (`mom1m`) ([GKX](https://www.nber.org/papers/w25398)). Net of costs only in large caps ([de Groot](https://ideas.repec.org/a/eee/jbfina/v36y2012i2p371-382.html)). Pays more when VIX is high ([Nagel](https://nber.org/papers/w17653)) | `rev_5d` ✓ |
| **Industry momentum** | Strong industries keep going | 1–12 mo | Top-7 predictor in GKX | ✗ |
| **Value** | Cheap on earnings/sales/book | months–years | Replicates (HXZ). Low turnover = cheap to run (Novy-Marx & Velikov) | `earn_yield` ✓ |
| **Profitability / quality** | High gross profit per dollar of assets | months–years | Replicates (HXZ). Low turnover | `gp_assets` ✓ |
| **Investment** | Firms growing assets slowly beat fast growers | months–years | Replicates (HXZ). Low turnover | `asset_growth` ✓ |
| **Low risk** | Low vol / low beta stocks do better than their risk implies | months | Volatility and beta are top predictors in GKX | `vol_63d`, `beta_mkt` ✓ |
| **Liquidity** | Size, dollar volume, spread | months | Top predictors in GKX, **but** 96% of the "trading friction" family fail once microcaps are controlled ([HXZ](https://ideas.repec.org/a/oup/rfinst/v33y2020i5p2019-2133..html)) | `adv_usd`, `spread_bps`, `market_cap` ✓ |
| **Short interest (market-wide)** | Heavy total shorting → weak market ahead | months–1 yr | Called "arguably the strongest known predictor" of the market. Out-of-sample R² 13.24% annual ([Rapach, Ringgenberg & Zhou 2016](https://cxoadvisory.com/short-selling/aggregate-short-interest-and-future-stock-market-returns)) | `si_float` is per stock; the market-wide version is not built |
| **Earnings drift (PEAD)** | Drift after earnings surprises | weeks | **Gone for large stocks since 2006** ([Martineau 2022](https://cfr.ivo-welch.info/published/papers/martineau2021rest.pdf)). Disputed by two 2025 papers ([UCLA Anderson](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/)). Your own test: faded out-of-sample | tested, 0 weight |
| **Attention** | Search spikes → higher prices ~2 weeks, then reversal within the year (retail-driven) | 2 wk | Russell 3000, 2004–2008 only ([Da, Engelberg & Gao](https://rady.ucsd.edu/faculty/directory/engelberg/pub/portfolios/GOOGLE.pdf)) | Wikipedia views in `search_data.py` |
| **Filing changes ("Lazy Prices")** | Firms that change 10-K/10-Q wording underperform | **months** (no announcement-day effect) | Up to 188 bps/month, 1995–2014 ([Cohen, Malloy & Nguyen](https://www.nber.org/papers/w25084)) | SEC filings are in `search_data.py`, but it's a slow signal and a daily test may miss it |
| **Insider buying/selling** | Only "opportunistic" (non-routine) insider trades carry information | months | [Cohen, Malloy & Pomorski](https://hbswk.hbs.edu/item/decoding-inside-information) | ✗ (Yahoo only has ~2 yrs; SEC EDGAR Form 4s go back much further, worth checking) |

### Timing inside the day
- Momentum profits come almost entirely **overnight** (close → open). Reversal and most other strategies earn **during the day** ([Lou, Polk & Skouras 2019](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf)).
- Caveat: Alpha Architect couldn't replicate the intraday-momentum part ([summary](https://alphaarchitect.com/2015/02/overnight-momentum-vs-intraday-momentum/)).
- This matters for "buy at open, sell at close" rules: they skip the overnight part, which is where momentum is paid.

### What machine learning found (Gu, Kelly & Xiu, 94 signals, ~30,000 stocks, 1957–2016)
- The best model's **monthly** out-of-sample R² for single stocks is **0.33–0.40%**. Even the best model explains under half a percent of a stock's monthly move ([NBER w25398](https://www.nber.org/papers/w25398)).
- A long-short decile portfolio scored an out-of-sample Sharpe of 1.35 value-weighted and 2.45 equal-weighted, **before costs**. Simple OLS scored 0.61 / 0.83.
- The top signals were price trends (short-term reversal, 12-mo momentum, momentum change, industry momentum, max return), then liquidity, then volatility/beta, then valuation.
- Caveat: much of that profit comes from **microcaps and distressed stocks**. Excluding microcaps cut the payoff ~62% (press-release figure), and costs cut it further ([Avramov, Cheng & Metzker 2023](https://papers.ssrn.com/abstract=3450322), [CUHK summary](https://cbk.bschool.cuhk.edu.hk/the-limitations-of-using-ai-to-pick-stocks/)).

---

## 3. Why most "profitable" signals don't survive

- **Publication kills edges.** Across 97 predictors, returns were ~26% lower out-of-sample and ~58% lower after publication ([McLean & Pontiff 2016](https://www.gwern.net/doc/economics/2016-mclean.pdf)).
- **Most fail strict replication.** Of 452 anomalies, 65% fail at |t| ≥ 1.96 once you use NYSE breakpoints and value weights. 82% fail at the stricter 2.78 hurdle ([Hou, Xue & Zhang 2020](https://ideas.repec.org/a/oup/rfinst/v33y2020i5p2019-2133..html)).
- **But the big themes hold up.** Most factors replicate, cluster into 13 themes, and work in 93 countries ([Jensen, Kelly & Pedersen 2023](https://www.nber.org/papers/w28432)). The two views agree on the famous ones: value, momentum, investment and profitability replicate well ([summary](https://systematicallybiased.substack.com/p/pockets-of-replicability-post5)). What dies is the obscure, microcap and high-turnover stuff.

---

## 4. Is crypto better for forecasting?

### Evidence that made crypto look predictable
- **Strong time-series momentum**, and investor attention forecast returns. Crypto had no exposure to stock-market factors ([Liu & Tsyvinski, RFS 2021](https://ideas.repec.org/a/oup/rfinst/v34y2021i6p2689-2727..html)).
- Three factors (market, size, momentum) explain the cross-section of coins. Sample: **2014–2018** ([Liu, Tsyvinski & Wu, JF 2022](https://papers.ssrn.com/abstract=3379131)).
- Bitcoin prices differed across exchanges for hours or weeks, especially between countries. The cause was capital controls, which you can't trade around from the US ([Makarov & Schoar 2020](https://corpgov.law.harvard.edu/2019/09/21/trading-and-arbitrage-in-cryptocurrency-markets)).

### Evidence since then
- Size and volume effects come from **micro-cap coins of negligible importance**. Momentum survives in bigger coins, but costs are high and most of its alpha comes from *shorting*. Most abnormal returns happen in bull markets and **fade over time** ([Fieberg, Liedtke & Zaremba 2024](https://open.icm.edu.pl/handle/123456789/25544)).
- Only 13 of 49 crypto anomalies were significant over 2014–2023, and the size effect disappeared ([Lasso factor paper](https://www.sciencedirect.com/science/article/abs/pii/S0275531926000255)).
- Time-series momentum in 30 large coins was marginally positive before July 2020, and negative and insignificant after ([Arefev, SSRN preprint — not peer reviewed](https://papers.ssrn.com/sol3/Delivery.cfm/7405060.pdf?abstractid=7405060&mirid=1)).
- Counterpoint: *factor* momentum (momentum in crypto factors, not in coins) did not decline through 2022 ([Fieberg et al.](https://open.icm.edu.pl/server/api/core/bitstreams/86a51c47-8cd3-4201-88ee-42f44fb89227/content)).
- **Fake volume:** over 70% of reported volume on unregulated exchanges was wash trading ([Cong, Li, Tang & Yang 2023](https://nber.org/papers/w30783)). Any volume-based crypto signal is built on polluted data.
- **It's no longer a separate asset.** Bitcoin's correlation with stocks rose from ~0 before 2020 to mostly 0–0.6 afterward ([CME](https://www.cmegroup.com/insights/economic-research/2025/why-is-bitcoin-moving-in-tandem-with-equities.html)), with peaks of 0.87 in 2024 ([arXiv 2501.09911](https://arxiv.org/abs/2501.09911v1)).
- **Retail results:** a majority of crypto-app users in nearly all economies lost money on bitcoin, Aug 2015–Dec 2022 ([BIS Bulletin 69](https://www.bis.org/publ/bisbull69.htm)).

### My own check (Yahoo daily data, rules fixed before looking)
Period A = 2018–2021; Period B = 2022 to 2026-10-16/17. "Trend rule" = hold tomorrow only if the last 20 days were up, otherwise cash.

| | BTC | ETH | SPY | QQQ |
|---|---|---|---|---|
| Yearly volatility (B) | 51% | 69% | 17% | 23% |
| Worst single day (A) | −37% | −42% | −11% | −12% |
| Worst drop from peak (A / B) | −82% / −67% | −94% / −74% | −34% / −25% | −29% / −35% |
| Up days (B) | 49% | 50% | 53% | 54% |
| Does yesterday predict today? (B, lag-1 autocorr) | −0.03 (no) | −0.02 (no) | −0.03 (no) | −0.03 (no) |
| Trend rule Sharpe vs buy-and-hold (A) | **1.52 vs 0.78** | **1.36 vs 0.90** | 1.00 vs 0.88 | 1.33 vs 1.11 |
| Trend rule Sharpe vs buy-and-hold (B) | 0.57 vs 0.48 | **0.14 vs 0.21 (worse)** | 0.96 vs 0.74 | 0.86 vs 0.67 |
| Trades per year (B) | ~41 | ~42 | ~20 | ~20 |

How to read it:
- The crypto trend edge looked huge in 2018–2021 and **mostly disappeared** from 2022 on. That's the same fade the papers report.
- These are **before fees**. ~41 trades a year × roughly 0.4–1.2% per trade at Coinbase's entry tiers ([Bankrate](https://www.bankrate.com/investing/coinbase-review), [third-party review](https://financer.com/review/coinbase); Coinbase doesn't publish the full table, so check your account) = **~16–50% a year in fees**. That's more than the whole edge.
- For SPY/QQQ the trend rule mainly **cut drawdowns**. It didn't add return: it earned 2–3%/yr *less* than holding. That's risk control, not a forecasting edge.
- One note on Period A for stocks: the lag-1 autocorrelation was −0.22 there, but it is dominated by the March 2020 whipsaw days, so I didn't treat it as a signal.

### The noise problem
- How sure you can be about an edge grows with √(years of data): t ≈ Sharpe × √years.
- Crypto has ~10 usable years. A real edge with Sharpe 0.5 would show t ≈ 1.6, which **can't be confirmed**, even before the market changes underneath it.
- Bigger daily swings make the same % edge harder to see, not easier.

### Practical differences
- The old reason to day-trade crypto ("no $25k pattern-day-trader rule") is mostly gone. FINRA's PDT rule was replaced with intraday margin standards: SEC approved 2026-04-14, effective 2026-06-04. Brokers may phase it in until Oct 2027 ([E*TRADE](https://us.etrade.com/knowledge/library/margin/pattern-day-trading-rule-change), [tastytrade](https://tastytrade.com/learn/markets/industry/pattern-day-trading/)). Check your broker.
- Crypto trades 24/7, so there's no "overnight" period to exploit or avoid.

### Verdict
**Don't switch your focus to crypto for forecasting.** It's not more predictable *now*, costs more to trade at small size, and has worse data. Two sensible uses:
1. Use **bitcoin as an input** to stock formulas: a risk-appetite gauge, which is now correlated with Nasdaq. `search_data.py` already loads BTC-USD as a macro series, so the signal search will test it.
2. Owning some crypto is a *holding* decision, not a forecasting edge.

---

## 5. What this means for the formula (for Julian and the formula agent)

1. **Keep the testing discipline.** Pre-registration, search/confirm/holdout splits, |t| ≥ 3, the Bonferroni holdout and cost deductions in `signal_search.py` are exactly the professional standard ([HLZ](https://nber.org/papers/w20592)). Your results so far match the research:
   - 0/87 setups passed.
   - Earnings drift faded.
   - The same-day model only barely passed, and it runs on *reversal*, which is liquidity provision.
2. **Biggest upgrade: change the question.**
   - Today the question is "will THIS stock go up tomorrow?"
   - Ask instead "which stocks will beat the others over the next **5–21 days**?"
   - That's where momentum, value, profitability, investment and low-risk live. It's also where turnover is low enough to survive costs.
   - The `xs` test in `signal_search.py` is already cross-sectional. Adding 5- and 21-day holding periods is the natural next test.
3. **Combine many weak signals.**
   - Rank or z-score each signal across stocks each day, then average.
   - Don't hunt for one strong signal. WorldQuant's real alphas are only ~16% correlated with each other.
4. **Know what success looks like.**
   - 52–53% right, and monthly R² under 0.5%, is what *good* looks like ([GKX](https://www.nber.org/papers/w25398)).
   - Anything showing 60%+ in a backtest is almost certainly overfit.
5. **Condition on the VIX.**
   - Reversal pays more when the VIX is high ([Nagel](https://nber.org/papers/w17653)).
   - Momentum crashes after market drops, when volatility is high and the market rebounds ([Daniel & Moskowitz](https://www.nber.org/papers/w20439)).
6. **Mind the overnight/intraday split.** Momentum is paid overnight and reversal intraday ([Lou, Polk & Skouras](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf)). Open→close tests will miss momentum by design.
7. **Avoid microcaps.** That's where most anomalies and machine-learning gains live, and you can't trade them cheaply. S&P 500 is the right universe.
8. **Free data to check which signals still work after publication:**
   - [Open Source Asset Pricing](https://www.federalreserve.gov/econres/feds/open-source-cross-sectional-asset-pricing.htm) (Chen & Zimmermann): monthly long-short returns for 200+ published predictors. 98% of the originally significant ones reproduce ([paper](https://cfr.ivo-welch.info/published/papers/chen2021open.pdf)).
   - [AQR Century of Factor Premia](https://www.aqr.com/Insights/Datasets/Century-of-Factor-Premia-Monthly).
   - [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

---

## Checked but left out (couldn't verify)
- The often-repeated quote that Renaissance is "right 50.75% of the time." I couldn't find it in a checkable source.
- The "momentum lost 73% in 2009" figure from blogs. The paper's own numbers are above.
- Coinbase's exact current fee tiers. Sources conflict, and Coinbase shows them only after login.
- Whether a pending bill will apply the wash-sale rule to crypto. Tax blogs say it doesn't apply under current law (crypto = property), but the bill's status conflicts across sources ([Gordon Law](https://gordonlaw.com/learn/crypto-wash-sale-rule-in-2026-is-the-loophole-finally-closed/)).
