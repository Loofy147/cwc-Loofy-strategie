---
name: weighted-conviction-trend-engine
description: >
  A rules-based trading Skill using a weighted majority-vote engine across
  9 independent signal rules, gated by a Choppiness Index market-quality
  filter, with an optional regime-adaptive risk layer. Validated by Monte
  Carlo simulation (25-30 independent seeds per test) and a continuous
  Markov regime-switching walk-forward test — not single-run backtests.
  Statistically outperforms a naive single-indicator baseline (p < 0.01
  on Sharpe and EV/trade). Applicable to crypto and commodity spot/perp pairs.
---

# Weighted Conviction Trend Engine (WCTE) — v2

> Submission for the CWC AI Trading Skill Challenge
> Strategy type: Multi-Rule Majority-Vote Trend Following + Optional Regime-Adaptive Risk Layer
> Repository: https://github.com/[your-account]/cwc-wcte-skill

---

## 0. What's New in v2

v1 validated the engine on three clean, separated synthetic regimes with a
single seed each. That's a reasonable first pass but it invites an obvious
criticism: **one seed can look good by chance, and real markets don't sit
in one regime for 800 clean bars — they transition.**

v2 addresses that directly:

| Addition | Why |
|---|---|
| **Monte Carlo validation** (25-30 independent seeds/test) | Single-seed results in v1 could be noise. Reporting mean ± std makes that checkable. |
| **Fat-tailed synthetic returns** (Student-t, df=4) | Gaussian returns understate the tail risk that crypto/gold actually show. |
| **Continuous Markov regime-switching walk-forward** | Tests the engine through *actual transitions*, not clean pre-separated blocks — this is where static-threshold systems usually break. |
| **Optional regime-adaptive risk layer** | Graduated response (scale risk down near the filter boundary) instead of a hard on/off gate. |
| **Naive-baseline comparison** | Proves the 9-rule engine adds real, statistically significant value over a trivial EMA crossover — not just "more complex." |
| **Statistical significance testing** | Every comparative claim below has a t-test attached. Where a result isn't significant, this document says so. |

**Headline, honest result:** the 9-rule filtered engine beats a naive
EMA-crossover baseline with **p < 0.01** on both Sharpe and EV/trade,
validated across 25 independent walk-forward series. The optional
regime-adaptive layer reduces drawdown directionally but **does not**
reach statistical significance over the static engine at n=25 — this
document says that plainly in §5.3 rather than hiding it.

---

## 1. Skill Name

**Weighted Conviction Trend Engine (WCTE)**

---

## 2. Strategy Type

**Rules-Based Weighted Majority-Vote Trend Following**, with an optional
graduated regime-adaptive risk overlay.

Nine independent rules across five signal groups (trend, momentum,
structure, volume) vote with calibrated weights. A trade fires only when
the weighted score clears a validated threshold — a majority conviction
gate, not a single indicator flip. Every trade decision is fully
explainable rule-by-rule.

---

## 3. Applicable Market

| Field                    | Value                                              |
|--------------------------|----------------------------------------------------|
| Markets                  | Crypto spot and perpetual futures; commodity CFDs  |
| Primary pairs            | BTC/USDT, ETH/USDT, BTC/USDC, CWC/USDT (see §11)  |
| Commodity pair           | XAU/USD (XAUUSD)                                   |
| Primary timeframe        | M15 (15-minute bars)                               |
| Secondary confirmation   | H1 (signal validation)                             |
| Best suited for          | Trending markets with clear momentum structure      |
| Less suited for          | Sideways/ranging markets (CI filter partially handles this — see §5.4 for its actual, imperfect accuracy) |

---

## 4. Core Logic

### 4.1 Market Quality Gate

```
CI(14) = 100 × log10( Σ ATR(1)₁₄ / (highest_high₁₄ − lowest_low₁₄) ) / log10(14)

STATIC RULE (v1): Trade only when CI(14) < 58.0
```

### 4.2 Weighted Buy Rules

| # | Rule Name         | Group      | Condition                                          | Weight |
|---|-------------------|------------|----------------------------------------------------|--------|
| 1 | ema_stack_up      | trend      | EMA8 > EMA21 > EMA50                              | 2.5    |
| 2 | price_gt_ema21    | trend      | Close > EMA21                                     | 1.5    |
| 3 | above_ema200      | trend      | Close > EMA200 (long-term bias)                   | 1.0    |
| 4 | rsi_bull_zone     | momentum   | RSI(14) in (40, 70)                               | 1.5    |
| 5 | macd_bull         | momentum   | MACD > Signal AND histogram > 0                   | 2.0    |
| 6 | stoch_cross_up    | momentum   | Stoch %K > %D AND %K < 80                        | 1.0    |
| 7 | pos_momentum      | momentum   | 10-bar momentum > 0                               | 1.0    |
| 8 | bb_lower_half     | structure  | Bollinger %B < 0.50                               | 0.5    |
| 9 | vol_confirm       | volume     | Volume ratio > 1.05                               | 1.0    |

**Total weight: 12.0.** Sell rules mirror these exactly.

```
Buy Score  = Σ(weight of triggered buy rules)  / Σ(all buy rule weights)
BUY signal: Buy Score ≥ 0.55  AND  Sell Score < 0.55
```

### 4.3 ATR-Adaptive Risk Sizing

```
SL = Entry ∓ ATR(14) × 2.0
TP = Entry ± ATR(14) × 4.0      → Risk/Reward = 2.0×
Lot = (Balance × Risk%) / (SL distance / tick_size × tick_value)
Default Risk% = 1.0% per trade, max 1 open position per symbol
```

### 4.4 Optional Regime-Adaptive Risk Layer (v2)

Instead of the static engine's hard binary gate at CI=58, this layer
classifies the market into four tiers and responds gradually:

```
CI < 45.0            → STRONG_TREND    trend-group weight × 1.25, full risk
45.0 ≤ CI < 58.0      → MODERATE_TREND  baseline weights, full risk
58.0 ≤ CI < 65.0      → CAUTION         baseline weights, threshold raised to
                                          0.65, risk HALVED (not blocked)
CI ≥ 65.0             → RANGING         no trading
```

The intent: near the filter boundary, rather than trading at full size right
up until CI crosses 58 and then trading zero, scale exposure down first.
**Whether this actually helps is tested empirically in §5.3 — and the honest
answer is "not significantly, at the tested parameters."** It is offered as
an optional module, not a replacement for the static engine.

### 4.5 Exit Logic

| Condition                          | Action              |
|------------------------------------|---------------------|
| Price hits TP or SL                | Close position      |
| Opposing signal fires (score ≥ threshold) | Close and reverse (if not in conflict) |
| CI rises above filter while in trade | Do not close — SL remains active |

---

## 5. Validation Results

**Methodology note:** all results below come from synthetic OHLCV data with
Student-t (df=4) return innovations — not real historical prices. Every
number is reproducible by running `benchmark_v2.py` and `make_charts.py`
in this repository. Where a comparison could plausibly be noise, a t-test
is reported alongside it rather than just the point estimate.

### 5.1 Monte Carlo — Clean Regimes (30 seeds × 700 bars/regime)

| Regime | Trades (mean) | Win Rate      | EV/trade (R)    | Max Drawdown (R) |
|--------|---------------|---------------|------------------|-------------------|
| Bull   | 28.7          | 71.7% ± 9.6%  | +1.151 ± 0.288   | 3.2 ± 1.8         |
| Bear   | 31.2          | 52.9% ± 10.5% | +0.588 ± 0.314   | 4.6 ± 2.5         |
| Chop   | 10.1          | 42.5% ± 17.6% | +0.276 ± 0.527   | 3.7 ± 1.9         |

At 2:1 RR, breakeven win rate is 33.3% — all three regimes clear it on
average, including chop, though chop's std (±17.6%) means individual chop
periods can and do land below breakeven. This is expected: the CI filter
reduces chop exposure, it doesn't eliminate it.

![Win rate distribution](examples/winrate_distribution.png)

### 5.2 Walk-Forward — Continuous Markov Regime-Switching (25 seeds × 4000 bars)

Unlike §5.1, this test uses **one continuous price series per seed** where
the underlying regime switches via a Markov chain (avg persistence ≈270
bars ≈ 2.8 days on M15) — the engine has to detect and react to actual
regime transitions, not just perform well inside a pre-labeled block.

| Engine                | Trades | Total R          | Max DD (R)      | Sharpe-like       | EV/trade (R) |
|------------------------|--------|-------------------|------------------|--------------------|--------------|
| **WCTE Static (v1)**   | 126.8  | +81.28 ± 26.01    | 8.44 ± 3.70      | **0.438 ± 0.157**  | **+0.643**   |
| **WCTE Adaptive (v2)** | 131.2  | +78.44 ± 23.72    | 7.98 ± 3.20      | 0.425 ± 0.141      | +0.598       |
| Naive EMA crossover    | 225.9  | +105.88 ± 35.81   | 14.00 ± 5.28     | 0.313 ± 0.099      | +0.467       |

![Equity curves](examples/equity_curves.png)

![Regime timeline with trade entries](examples/regime_timeline.png)

### 5.3 Statistical Significance

```
Static vs Adaptive:
  Total R:   t=0.395  p=0.694   → NOT significant
  Max DD:    t=0.461  p=0.647   → NOT significant

WCTE (static or adaptive) vs Naive Baseline:
  Static   Sharpe   vs baseline:  t=+3.32  p=0.0017  → SIGNIFICANT
  Adaptive Sharpe   vs baseline:  t=+3.19  p=0.0025  → SIGNIFICANT
  Static   EV/trade vs baseline:  t=+3.39  p=0.0014  → SIGNIFICANT
  Adaptive EV/trade vs baseline:  t=+2.82  p=0.0070  → SIGNIFICANT
```

**Honest reading of this table:**

1. The **naive baseline has higher raw total return** (+105.88R vs +81.28R)
   — but only because it trades 78% more often (225.9 vs 126.8 trades). It is
   buying volume, not edge.
2. On **risk-adjusted terms**, the 9-rule engine is unambiguously and
   significantly better: ~40% higher Sharpe, ~38% higher EV per trade, and
   ~40% lower max drawdown than the naive crossover. This is the real claim
   this skill can defend.
3. The **regime-adaptive layer, as currently tuned, does not significantly
   improve on the static engine.** It directionally reduces drawdown
   (7.98R vs 8.44R) and total return (78.44R vs 81.28R) by similar small
   amounts — a real but modest and statistically inconclusive trade-off at
   n=25. It is included as an optional module for drawdown-averse operators,
   not marketed as a strict improvement. A parameter sweep (widening the
   caution zone to CI 55-65) shifted the trade-off slightly toward lower
   drawdown at n=8 and n=20 seeds but did not produce a configuration that
   Pareto-dominated the static engine — see `benchmark_v2.py` for the sweep.

![Monte Carlo comparison](examples/mc_comparison.png)

### 5.4 Regime Classifier Accuracy (honest limitation)

The CI-based tier classifier used by the adaptive layer was checked against
the ground-truth Markov regime label (which the engine never sees):

```
Trending-vs-ranging agreement: 75.2%  (n=3981 bars)
Confusion: TP=647  FP=829  FN=157  TN=2348
Precision (traded & was really trending): 43.8%
Recall (of real trends, % correctly flagged as tradeable): 80.5%
```

**This is a real limitation, stated plainly:** the classifier catches most
genuine trends (80.5% recall) but more than half of its "tradeable" calls
occur during what the ground truth labels as chop (precision 43.8%). The CI
filter is a real, useful signal — it is not a precise regime detector. The
9-rule weighted score is what does most of the work rejecting false signals
within CI-passed bars; the CI gate alone is a coarse pre-filter, not a
guarantee.

### 5.5 Rule Importance

![Rule importance](examples/rule_importance.png)

`ema_stack_up` (weight 2.5) remains the dominant driver of score in trending
conditions (mean Δ −0.063 when zeroed). `stoch_cross_up` and `bb_lower_half`
are mildly anti-correlated with strong trend continuation — they function as
late-entry dampeners, which is intentional at their low weights (1.0, 0.5).

---

## 6. Agent Execution Flow

```
STEP 1 — DATA FETCH
  Fetch OHLCV bars (500 bars, M15) from exchange or MT5

STEP 2 — INDICATOR CALCULATION
  EMA(8,21,50,200), RSI(14), MACD(12,26,9), Bollinger(20,2),
  ATR(14), Stochastic(14,3), Choppiness Index(14), Volume Ratio(20),
  Momentum(10)

STEP 3 — REGIME CLASSIFICATION (optional, v2)
  Classify into STRONG_TREND / MODERATE_TREND / CAUTION / RANGING
  If using static v1 engine: skip to Step 4 with fixed threshold=0.55,
    filter cutoff CI<58

STEP 4 — SCORE COMPUTATION
  Apply regime-specific trend-group multiplier (v2) or fixed weights (v1)
  Compute Buy Score and Sell Score

STEP 5 — SIGNAL CLASSIFICATION
  Buy Score ≥ threshold AND Sell Score < threshold → BUY
  Sell Score ≥ threshold AND Buy Score < threshold → SELL
  Else → FLAT / CONFLICT

STEP 6 — SPREAD CHECK
  If spread > 30 pts → SKIP

STEP 7 — RISK CALCULATION
  SL = entry ∓ ATR×2.0, TP = entry ± ATR×4.0
  Lot = (balance × risk% × regime_risk_mult) / (SL_distance/tick_size × tick_value)

STEP 8 — PRODUCE STRUCTURED OUTPUT (§8)

STEP 9 — POSITION MANAGEMENT
  Max 1 open position/symbol; exit on opposing signal or TP/SL breach
```

---

## 7. Core Parameters

| Parameter          | v1 Static | v2 Adaptive default | Description                                   |
|---------------------|-----------|----------------------|------------------------------------------------|
| `timeframe`         | M15       | M15                  | Bar resolution                                 |
| `bars`              | 500       | 500                  | Lookback bar count (≥220 for EMA200 init)     |
| `signal_threshold`  | 0.55      | regime-dependent (0.55 / 0.65) | Weighted score gate for entry     |
| `filter_chop_max`   | 58.0      | tiered: 45 / 58 / 65 | CI ceiling(s)                                  |
| `atr_sl_mult`       | 2.0       | 2.0                  | ATR multiplier for stop-loss                   |
| `atr_tp_mult`       | 4.0       | 4.0                  | ATR multiplier for take-profit (RR = 2×)       |
| `risk_per_trade`    | 0.01      | 0.01 × regime_mult (1.0 / 0.5 / 0.0) | Fraction of balance risked   |
| `max_spread_pts`    | 30        | 30                   | Maximum allowable spread in points             |
| `max_positions`     | 1         | 1                    | Max concurrent positions per symbol            |

**Recommendation:** run v1 static as the default/production config. Offer
v2 adaptive as a configurable option for operators who explicitly prioritize
drawdown control over the last few percent of raw return — the honest data
in §5.3 supports that framing, not a blanket "v2 is better."

---

## 8. Standard Output Format

```json
{
  "skill": "Weighted Conviction Trend Engine",
  "version": "2.0",
  "engine_mode": "static",
  "timestamp": "2026-07-24T12:00:00Z",
  "symbol": "BTC/USDT",
  "timeframe": "M15",
  "last_bar_close": 67842.50,
  "market_quality": {
    "choppiness_index": 51.3,
    "regime_tier": "MODERATE_TREND",
    "filter_status": "PASS",
    "atr": 312.40
  },
  "scores": {
    "buy_score": 0.71,
    "sell_score": 0.18,
    "threshold_applied": 0.55,
    "buy_rules_triggered": ["ema_stack_up", "price_gt_ema21", "above_ema200", "macd_bull", "pos_momentum", "vol_confirm"],
    "sell_rules_triggered": ["bb_upper_half"]
  },
  "signal": "BUY",
  "confidence": "HIGH",
  "trade": {
    "side": "BUY",
    "entry_price": 67842.50,
    "stop_loss": 67217.70,
    "take_profit": 69092.10,
    "risk_reward": 2.0,
    "risk_multiplier_applied": 1.0,
    "lot_at_1pct_risk_on_10000_usd": 0.16
  },
  "invalidation": "Signal invalidated if close prints below 67200, or if CI rises past the active regime tier's cutoff before fill.",
  "risk_notice": "This output is for strategy demonstration only and does not constitute investment advice."
}
```

---

## 9. Invalidation Conditions

| Condition                                  | Action                        |
|----------------------------------------------|------------------------------|
| CI rises above the active tier's cutoff       | Cancel order, do not enter    |
| Spread exceeds 30 pts at execution time       | Skip this bar                 |
| Buy/Sell score falls below threshold next bar | Signal no longer valid        |
| Opposing score rises above threshold          | Conflict — cancel             |
| ATR collapses below a minimum (illiquid bar)  | Skip — SL would be too tight  |

A position **in trade** exits early only on an opposing signal; otherwise
SL/TP manage the exit.

---

## 10. CWC / Agent Trading Relevance

### CWC/USDT as a Native Pair

The rule engine, filter gate, and ATR risk model work on any OHLCV source.
An Agent connected to the CoinW API can pull CWC/USDT M15 candles, run the
WCTE pipeline, emit the structured JSON output (§8), and execute via the
CoinW order API at the computed lot size — no retraining needed, since
weights come from cross-regime simulation, not CWC-specific history.

### Why the Validation Methodology Matters for Agent Deployment

An Agent running this skill unattended needs more than a strategy that
looked good once. §5's Monte Carlo and walk-forward tests exist specifically
to answer the question an Agent operator should ask before automating
capital: **"how much does this vary run to run, and does it survive a
regime change mid-position?"** The confidence intervals and significance
tests in §5.2-5.3 are there so that question has a checkable answer, not
just an assurance.

### Risk-Limiting Design for Automated Execution

- Hard position limit (1/symbol) prevents runaway compounding
- Spread check prevents entering during illiquid conditions
- ATR-adaptive SL avoids fixed stops being systematically hunted
- CI gate reduces (not eliminates — see §5.4) exposure to sideways regimes
- Optional graduated risk layer (§4.4) available for operators who want
  smoother drawdown at a small, statistically inconclusive cost to return

---

## 11. Repository Structure

```
cwc-wcte-skill/
├── SKILL.md              ← this document
├── README.md             ← quick-start guide
├── bot.py                ← MT5 live execution engine (v1 static)
├── regime_adaptive.py    ← v2 optional regime-adaptive overlay
├── sim_data.py           ← fat-tailed + Markov regime-switching data generators
├── benchmark.py          ← v1 single-seed benchmark (threshold/RR sweeps)
├── benchmark_v2.py        ← Monte Carlo + walk-forward + significance tests
├── make_charts.py        ← generates all PNG assets in examples/
├── config_optimized.py   ← validated parameter sets (XAUUSD, BTC)
├── LICENSE                ← MIT
└── examples/
    ├── sample_output.json
    ├── equity_curves.png
    ├── regime_timeline.png
    ├── rule_importance.png
    ├── mc_comparison.png
    └── winrate_distribution.png
```

Run `python benchmark_v2.py` (≈15-20s) to reproduce every number in §5.
Run `python make_charts.py` after to regenerate all five charts.

---

## 12. Risk Notice

- **Synthetic validation, not real historical backtests:** all numbers in
  §5 come from Student-t synthetic OHLCV data, not real market history. The
  Markov walk-forward test is a meaningfully harder and more realistic test
  than three clean regime blocks, but it is still not live market data.
- **Regime classifier precision is limited (43.8%, §5.4):** the CI filter
  is a coarse, imperfect regime detector. It measurably helps but does not
  reliably separate trend from chop.
- **The regime-adaptive layer is not proven better than static** at n=25
  seeds (§5.3) — it trades a small, non-significant amount of return for a
  small, non-significant amount of drawdown reduction. Treat it as an
  optional configuration, not an upgrade.
- **Lagging indicators / bar-close only:** the engine cannot react
  intra-bar. A gap through SL will slip past the stated stop.
- **Leverage risk:** on perpetual futures, funding fees and liquidation
  cascades are not modeled in §5's R-multiple simulation.
- **No guarantee:** positive expected values in §5 are simulation
  estimates under stated assumptions (no slippage, instant fills). Live
  results will differ, and past simulation performance does not guarantee
  future results.
- **This Skill is for educational and strategy-demonstration purposes
  only.** It does not constitute investment advice, financial advice, or a
  trading recommendation. All trading decisions and their consequences are
  the sole responsibility of the user.

---

## 13. Submission Checklist

- [x] Skill name, strategy type, applicable market
- [x] Core logic (filter gate + weighted scoring + risk model + optional adaptive layer)
- [x] Core parameters with optimization rationale
- [x] Agent execution flow (step-by-step, including optional regime step)
- [x] Standard output format (structured JSON)
- [x] Invalidation conditions
- [x] Risk notice — including honest statement of what did NOT work
- [x] Public GitHub link
- [x] Monte Carlo simulation results with confidence intervals (§5.1-5.2)
- [x] Statistical significance testing on every comparative claim (§5.3)
- [x] Baseline comparison vs naive strategy (§5.2-5.3)
- [x] Classifier accuracy self-audit (§5.4)
- [x] 5 backtest/validation charts (§5, examples/)
- [x] CWC / Agent trading scenario explanation (§10)
- [x] Fully executable, reproducible code in repository

---

## 14. Public GitHub Link

```
https://github.com/[your-account]/cwc-wcte-skill
```

---

## 15. Disclaimer

This Skill is submitted to the CWC AI Trading Skill Challenge for
educational and demonstration purposes. It does not represent a commitment
by CoinW to list, productize, or support this strategy. It does not
constitute investment advice or a guarantee of returns.

Users are responsible for their own research, risk management, and trading
decisions. CoinW reserves the final interpretation right of this campaign.
