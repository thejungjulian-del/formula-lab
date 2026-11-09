# Formula Lab: notes for agents

Read `quant-research.md` before designing or testing any formula or signal. It's verified research (sources linked) on which quant factors survive testing, and on why crypto is not easier to forecast.

Keep these in mind:
- **Durable signals:** momentum 12-1, short-term reversal, industry momentum, value, profitability, investment, low risk, liquidity.
  - They work **cross-sectionally** (ranking stocks against each other) at **5–21+ day** horizons.
  - Daily single-ticker edges are tiny: 52–53% right is good. 60%+ in a backtest = overfit.
- **Combine many weak signals.** Rank or z-score each signal across stocks per day, then average. Don't hunt for one strong signal.
- **Costs kill high-turnover signals.** Reversal survives costs only in large caps with turnover control. Avoid microcaps.
- **Condition on VIX.** Reversal pays more when VIX is high. Momentum crashes in rebounds after market drops.
- **Overnight vs intraday:** momentum is paid overnight, reversal intraday. Open→close tests miss momentum by design.
- **Testing:** keep pre-registration, search/confirm/holdout splits and |t| ≥ 3. Expect out-of-sample returns ~30–60% below in-sample.
- **Crypto:** use BTC as an *input* (risk appetite), not as a forecasting target. Its published edges faded after ~2020, and fees eat the rest.
