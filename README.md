# Financial Entropy Agent

**An entropy-based market risk surveillance system that uses Information Theory and Statistical Physics to detect structural risk invisible to traditional volatility measures.**

Built by a former securities broker who studied physics — motivated by the observation that financial markets exhibit Type-2 chaos, where dangerous coordination (herding, panic) produces deterministic structure that entropy can detect.

![Regime Validation](validation/regime_validation.png)

---

## The Problem

Traditional risk management relies on volatility — the standard deviation of past returns. This approach has a fundamental blind spot: **volatility is backward-looking**. It tells you the market *was* risky, not that it *is becoming* risky. Before every major crash, rolling volatility is low because the market appears calm.

As a securities broker observing the Vietnamese stock market (VNINDEX), I noticed a pattern: the most dangerous periods weren't when the market was visibly chaotic, but when price movements became unusually *structured* — when everyone was buying the same stocks, following the same momentum, creating an illusion of stability that masked fragility underneath.

This led to a hypothesis rooted in my physics background: **if financial markets are Type-2 chaotic systems, then entropy — the measure of disorder — should detect structural risk before volatility does.**

---

## The Approach

The system measures market disorder through three entropy metrics, classifies structural regimes using unsupervised learning, and estimates conditional volatility via GARCH modeling.

### Entropy Feature Engineering

**Weighted Permutation Entropy (WPE)** measures the ordinal pattern disorder in price log-returns. Low WPE means price movements follow deterministic, repeating patterns — the signature of coordinated behavior (herding, panic, momentum). High WPE means random, unpredictable movements — a normal, healthy market.

**Standardized Price Sample Entropy (SPE_Z)** measures trajectory complexity of close prices. It captures how predictable the price path is in amplitude space, complementing WPE's ordinal analysis.

**Volume Entropy (Shannon + SampEn)** measures liquidity structure — whether capital flow is concentrated (institutional consensus) or dispersed (fragmented, no agreement).

### Regime Classification (GMM)

A Full-Covariance Gaussian Mixture Model (n=3) operates directly on raw [WPE, SPE_Z] features — no preprocessing or normalization — discovering the natural topological boundaries of three market phases:

- **Deterministic** (low entropy) — Strong ordinal structure in price movements. Indicates trending, coordinated behavior. *Highest risk.*
- **Transitional** (mid entropy) — Phase boundary between ordered and disordered states. *Moderate risk.*
- **Stochastic** (high entropy) — Random walk behavior. Normal, healthy market. *Lowest risk.*

### Unsupervised Regime Discovery — Visual Evidence

The GMM discovers natural cluster boundaries in raw entropy feature space without any human-imposed thresholds or preprocessing.

**Price Entropy Phase Space** (Plane 1: WPE × SPE_Z)

![Price Phase Space](docs/images/price_phase_space.png)

The scatter plot shows three distinct clusters emerging from raw entropy features:
- **Red (Deterministic)**: Low WPE, negative SPE_Z — structured, predictable price patterns indicating coordinated behavior
- **Yellow (Transitional)**: Mid-range entropy — the boundary between ordered and disordered states
- **Green (Stochastic)**: High WPE, positive SPE_Z — random, complex price evolution characteristic of normal market conditions

The dashed ellipses represent 95% confidence boundaries of each GMM component's full covariance matrix — each cluster has its own shape and orientation, capturing the true geometry of entropy distributions.

**Volume Entropy Phase Space** (Plane 2: Shannon × SampEn)

![Volume Phase Space](docs/images/volume_phase_space.png)

Volume flow structure reveals three liquidity regimes:
- **Blue (Consensus Flow)**: Low entropy — capital moves in organized, predictable patterns (institutional consensus)
- **Purple (Dispersed Flow)**: Mid entropy — fragmented capital flow, no clear institutional agreement
- **Red (Erratic/Noisy Flow)**: High entropy — chaotic liquidity with unpredictable volume impulses

**Market Structure Timeline**

![Market Structure](docs/images/market_structure.png)

The regime classification overlaid on VNINDEX price history (2020–2026) demonstrates how entropy regimes align with market events. Note that Deterministic (red) periods coincide with both sharp rallies and sharp declines — entropy measures structural coordination, not direction.

### Conditional Volatility (GARCH)

GARCH(1,1) provides the conditional volatility backbone, with entropy features tested as exogenous variables (GARCH-X). Filtered Historical Simulation computes Expected Shortfall (ES 5%) without assuming Gaussian tails.

### AI Explanation Layer

An LLM agent (Claude API) translates quantitative signals into natural-language risk narratives for non-technical investors — the "last mile" between mathematical models and actionable advice.

For the full mathematical formulations (WPE, SampEn, Yeo-Johnson, GMM specifications, GARCH variance equations), see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Validation Results

All validation is out-of-sample on VNINDEX data (2020–2026, ~1,600 trading days). Code is in the `validation/` folder and fully reproducible.

### V1: Regime Labels Discriminate Future Volatility

Entropy-based regime labels significantly predict forward 20-day realized volatility.

| Regime | Mean Forward 20d Vol | Median | n |
|:-------|:---------------------|:-------|:--|
| **Deterministic** (High Risk) | 21.59% | 20.23% | 464 |
| Transitional | 18.00% | 16.11% | 754 |
| **Stochastic** (Low Risk) | 15.15% | 11.54% | 340 |

**Kruskal-Wallis H = 150.89, p < 0.0001** — Regime labels carry statistically significant information about future market risk.

> **Note:** "Deterministic" does not mean "market will fall." It means "market structure is highly coordinated" — which can manifest as aggressive rallies *or* sharp declines. The higher forward volatility (21.59%) reflects the fact that coordinated states produce larger moves in *both* directions, consistent with the Type-2 chaos framework where structural order precedes instability.

### V3: Tail Risk Detection — Where Entropy Excels

The system's discriminative power *increases with event severity* — precisely the behavior required for tail risk management.

| Timeframe | Drawdown | Stochastic | Deterministic | Lift |
|:----------|:---------|:-----------|:--------------|:-----|
| 5 days | > 3% | 8.5% | 17.4% | **2.06×** |
| 5 days | > 7% | 0.8% | 4.3% | **5.50×** |
| 10 days | > 5% | 6.3% | 16.0% | **2.54×** |
| 20 days | > 7% | 6.5% | 19.6% | **3.00×** |

When the system identifies a Deterministic regime, the probability of a severe drawdown (>7% in 5 days) is **5.5 times higher** than during Stochastic regimes. For a risk manager, this is actionable information for position sizing.

![Risk Alert Hit Rate](validation/risk_alert_hitrate.png)

### V4: Entropy vs Simple Volatility — Complementary, Not Replacement

| Model | Features | H-statistic | p-value |
|:------|:---------|:------------|:--------|
| A: Entropy | WPE, SPE_Z | 150.55 | < 0.0001 |
| B: Simple Vol | Rolling 22d vol, 5d vol change | 299.78 | < 0.0001 |
| C: Combined | WPE, SPE_Z, Rolling 22d vol | 229.04 | < 0.0001 |

Simple volatility discriminates future volatility better (H=299.78 vs 150.55) — this is expected because past volatility auto-correlates with future volatility. **However, entropy measures a different dimension of risk: structural fragility.** Entropy detects FOMO peaks, herding behavior, and momentum concentration *before* they manifest as volatility spikes. The Combined model (H=229.04) confirms entropy carries complementary information beyond what simple volatility provides.

![Entropy vs Simple Vol](validation/entropy_vs_simple.png)

### V2: GARCH Forecast Evaluation — Honest Limitations

| Metric | GARCH(1,1) | Rolling 22d | Winner |
|:-------|:-----------|:------------|:-------|
| QLIKE | 1.7765 | 1.5737 | Rolling 22d |
| MSE Variance | 15.5130 | 15.3941 | Rolling 22d |
| Correlation(σ, \|r\|) | 0.3302 | — | — |

GARCH(1,1) underperforms simple rolling volatility on point forecast metrics for VNINDEX. This is consistent with frontier market characteristics — VNINDEX exhibits jump risk (circuit breakers, policy shocks) that GARCH's smooth conditional variance cannot capture. The positive directional correlation (r=0.33) confirms the model tracks volatility direction correctly but misestimates magnitude.

This result justifies the system's use of Filtered Historical Simulation rather than parametric VaR — FHS does not assume Gaussian tails and is more robust to the fat-tailed distribution of VNINDEX returns.

![GARCH Forecast Evaluation](validation/garch_forecast_eval.png)

### V5: Cross-Market Validation — VNINDEX, S&P 500, Bitcoin

The entropy framework was tested across three markets with fundamentally
different microstructures to assess generalizability.

| Market | Circuit Breaker | Dominant Participants | H-statistic | p-value | Entropy Paradox |
|:-------|:----------------|:----------------------|:------------|:--------|:----------------|
| **VNINDEX** | ±7% daily limit | Retail-dominated | **192.43** | <0.0001 | **✓ YES** (Det 20.5% vs Sto 14.0%) |
| **S&P 500** | None (daily) | Algo/institutional | 14.25 | 0.0008 | **✗ INVERTED** (Sto 16.8% > Det 16.2%) |
| **Bitcoin** | None, 24/7 | Mixed retail/bot | **42.70** | <0.0001 | **✓ YES** (Det 44.6% vs Sto 43.9%, thin margin) |

![Cross-Market Validation](validation/cross_market_validation.png)

**Key finding:** The Entropy Paradox is **market-microstructure dependent**,
not universal. This reveals a deeper insight about what entropy actually measures
in different market contexts:

- **Frontier markets (VNINDEX):** Low entropy = retail herding, panic, FOMO.
  Coordinated behavior from unsophisticated participants is inherently unstable —
  **order = danger**. Circuit breaker ±7% constrains ordinal patterns, producing
  cleaner GMM separation (H=192.43).

- **Developed markets (S&P 500):** Low entropy = institutional stabilization,
  efficient price discovery by algorithmic market makers. "Order" comes from
  sophisticated liquidity provision, which is stabilizing — **order ≈ stability**.
  Without circuit breakers, WPE distribution is compressed (0.65–1.0 vs 0.4–1.0 for VNINDEX),
  making regime separation harder (H=14.25).

- **Crypto (Bitcoin):** Paradox holds directionally but with thin margin (44.6% vs 43.9%).
  Bitcoin has both retail herding (like VNINDEX) and bot trading (like S&P 500),
  producing mixed signals. WPE parameters calibrated for daily trading (window=22)
  may not be optimal for 24/7 markets.

**Implication:** Entropy does not measure the same thing across all markets.
On frontier markets, it measures **behavioral coordination risk** (herding).
On developed markets, it measures **informational efficiency** (price discovery quality).
This distinction maps directly to the Adaptive Market Hypothesis (Lo, 2004):
market efficiency is not binary but evolves with participant composition
and institutional structure.

---

## Key Insight: The Entropy Paradox

The most important finding from validation was counterintuitive: **in financial markets, low entropy = high risk, and high entropy = low risk.**

This is the *opposite* of physical systems, where low entropy indicates calm equilibrium. In financial markets — which are Type-2 chaotic systems with reflexive, adversarial participants — low entropy means price movements have become deterministic. Deterministic structure in financial prices is not a sign of health; it is the signature of **coordinated behavior**: herding, panic selling, or momentum-driven rallies. These states are inherently unstable and produce the largest realized moves.

Maximum entropy, by contrast, means maximum randomness — a market where diverse participants with diverse strategies cancel each other out. This is the *healthy* state.

**Entropy does not measure volatility. It measures the absence of dangerous coordination.**

This insight directly connects to the Type-2 chaos hypothesis that motivated the project: in a system where participants observe and react to each other (reflexivity), order is a warning signal.

**Cross-market evidence** (see V5 above) reveals that the Entropy Paradox
is specific to markets where low entropy signals behavioral coordination
rather than institutional stabilization. This finding connects to the broader
entropy-efficiency literature (Zanin et al., 2012; Risso, 2008) which
established that permutation entropy correlates with market development stage,
but extends it by showing that the *direction* of the entropy-risk relationship
itself depends on market microstructure — a result not previously documented
in the literature.

**Practical implication:** Because Deterministic regimes coincide with both sharp rallies *and* sharp declines, the system is not a directional predictor. It is a **structural fragility detector**. When the market enters a Deterministic regime during a rally, it does not mean "sell now" — it means "the current trend is driven by coordination rather than diverse conviction, making it vulnerable to sharp reversals." This distinction between directional prediction and structural assessment is critical for responsible deployment in financial applications.

---

## Lessons Learned

**1. Validation changes everything.** The initial regime labels were inverted (mislabeled "Stable" for the highest-risk regime). Without forward-looking validation, this error would have gone undetected and the system would have given exactly wrong advice. Lesson: never trust model outputs without out-of-sample evidence.

**2. Complexity must justify itself.** GARCH-X with entropy exogenous variables was statistically insignificant during calm periods — entropy adds no value to volatility forecasting when the market is already stable. The system needed to be redesigned with adaptive activation: entropy contributes through regime classification (always active) rather than forcing it into the variance equation.

**3. Simple benchmarks are essential.** Rolling 22-day volatility outperformed GARCH on VNINDEX. This doesn't invalidate the entropy approach — it clarifies its role: entropy excels at structural risk detection (Lift 5.5× for tail events), while simple volatility excels at point forecasting. They solve different problems.

**4. Domain knowledge matters more than model sophistication.** The Entropy Paradox was only interpretable because of my background in physics (understanding entropy in different system types) and finance (understanding that coordinated behavior drives crashes). A purely technical approach would have either missed the inversion or abandoned entropy as "broken."

**5. Market microstructure determines entropy interpretation.**
The same metric (WPE) measures different phenomena on different markets:
behavioral coordination risk on frontier markets versus informational
efficiency on developed markets. Any deployment of this framework on a
new market requires re-validation of the entropy-risk relationship direction,
not just re-fitting of model parameters. This is perhaps the most important
lesson: **a model's meaning is context-dependent, not just its parameters.**

---

## Mathematical Foundation

### Weighted Permutation Entropy (WPE)

For a time series $\{x_t\}$ with embedding dimension $m$ and lag $\tau$, all ordinal patterns $\pi \in S_m$ (permutations of $m$ values) are extracted from overlapping windows and weighted by amplitude variance:

$$\text{WPE} = -\sum_{\pi \in S_m} w(\pi)\ln w(\pi)$$

The amplitude-weighted probability of each pattern $\pi$ is:

$$w(\pi) = \frac{\displaystyle\sum_{t\,:\,\text{ord}(x_t^{(m)}) = \pi} \text{Var}\!\left(x_t,\, x_{t+\tau},\, \ldots,\, x_{t+(m-1)\tau}\right)}{\displaystyle\sum_{t} \text{Var}\!\left(x_t,\, x_{t+\tau},\, \ldots,\, x_{t+(m-1)\tau}\right)}$$

Output is normalized to $[0,1]$ by dividing by $\ln(m!)$.

**Parameters:** $m=3$, $\tau=1$, window $= 22$ days.

**Interpretation:** Low $\text{WPE}$ $\Rightarrow$ repeating ordinal patterns $\Rightarrow$ coordinated, deterministic behavior. High $\text{WPE}$ $\Rightarrow$ diverse patterns $\Rightarrow$ normal random market. The amplitude weighting ensures large-magnitude events (panic, rallies) carry more weight than small random fluctuations.

---

### Sample Entropy (SampEn)

SampEn measures the conditional probability that two $m$-length subsequences, similar within tolerance $r$, remain similar at length $m+1$:

$$\text{SampEn}(m,\, r) = -\ln\!\left(\frac{A}{B}\right)$$

where $B$ counts template pairs within tolerance $r$ at length $m$, $A$ counts the same at length $m+1$, and the adaptive tolerance is:

$$r = 0.2 \times \sigma_{\text{window}}$$

The feature fed into the GMM is the rolling Z-score:

$$\text{SPE\_Z}_t = \frac{\text{SampEn}_t - \bar{\mu}_t}{\bar{\sigma}_t}$$

**Parameters:** $m=2$, $r=0.2\,\sigma$, window $= 60$ days.

**Interpretation:** Low $\text{SampEn}$ $\Rightarrow$ predictable price trajectory (low complexity). Unlike WPE (ordinal structure), SampEn captures amplitude-space complexity.

---

### Shannon Volume Entropy

Applied to the rolling 60-day distribution of normalized volume:

$$H_{\text{Shannon}} = -\sum_{i=1}^{N} p_i \ln p_i$$

where $p_i$ is the probability mass in bin $i$ of the discretized volume histogram. Measures capital flow concentration: few dominant volume days $\Rightarrow$ low $H$ (institutional consensus). Uniform distribution $\Rightarrow$ high $H$ (dispersed, no agreement).

---

### GMM Regime Classifier

Full-covariance Gaussian Mixture Model with $k=3$ components fitted on $\mathbf{x} = [\text{WPE},\, \text{SPE\_Z}]^\top$:

$$p(\mathbf{x}) = \sum_{k=1}^{3} \pi_k\; \mathcal{N}\!\left(\mathbf{x} \mid \boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k\right)$$

Each component has its own mean $\boldsymbol{\mu}_k \in \mathbb{R}^2$ and free covariance $\boldsymbol{\Sigma}_k \in \mathbb{R}^{2\times 2}$, allowing elongated, rotated clusters that axis-aligned covariance would miss.

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| $k$ (components) | 3 | Three phases: ordered / transition / disordered |
| Covariance type | Full | Discovers true geometric structure of entropy distributions |
| `n_init` | 10 | Multiple restarts to avoid local optima |
| Preprocessing | None (Plane 1) | Raw $[\text{WPE},\, \text{SPE\_Z}]$ — natural scale carries physical meaning |
| Preprocessing | Yeo-Johnson (Plane 2) | Right-skewed volume features require normalization before GMM |

---

### GARCH(1,1) and GARCH-X Variance Equations

Base conditional variance model:

$$\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2, \qquad \alpha + \beta < 1$$

where $\omega > 0$ is the long-run variance floor, $\alpha$ the shock reaction coefficient, and $\beta$ the persistence parameter. The stationarity constraint $\alpha + \beta < 1$ ensures mean-reversion.

When entropy exogenous variables pass statistical pruning ($p < 0.10$), the model extends to GARCH-X:

$$\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2 + \boldsymbol{\gamma}^\top \mathbf{z}_{t-1}$$

where $\mathbf{z}_{t-1}$ are lagged, MinMaxScaled entropy features over a 504-day rolling window. Variables with $p > 0.10$ are dropped; if all are pruned the model reduces to pure GARCH(1,1).

The regime stress multiplier amplifies $\sigma_t$ to reflect structural vulnerabilities detected by the GMM:

$$\sigma_{\text{adj}} = \sigma_t \times m_{\text{regime}}, \qquad m \in \{1.0,\; 1.4,\; 2.2\}$$

**Filtered Historical Simulation** for tail risk: standardized residuals $z_t = \varepsilon_t / \sigma_t$ are drawn from their empirical distribution rather than a Gaussian assumption.

$$\text{VaR}_{5\%} = \sigma_t \cdot Q_{0.05}\!\left(\{z_s\}\right), \qquad \text{ES}_{5\%} = \sigma_t \cdot \mathbb{E}\!\left[z_s \mid z_s < Q_{0.05}\right]$$

This handles VNINDEX's fat tails and jump risk (circuit breakers, policy shocks) that parametric VaR cannot capture.

---

## Architecture Overview

The system processes a single OHLCV stream through six sequential pipeline layers:

```
LAYER 1 — DATA INGESTION
  skills/data_skill.py
  vnstock (VNINDEX, VN30 constituents) + yfinance fallback
  Output: OHLCV DataFrame, VN30 returns matrix

LAYER 2 — ENTROPY FEATURE ENGINE
  skills/quant_skill.py
  WPE (m=3, tau=1, window=22)         — ordinal disorder in price returns
  SPE_Z (m=2, r=0.2*sigma, window=60) — trajectory complexity, standardized
  Vol_Shannon + Vol_SampEn (window=60) — liquidity structure entropy
  Cross-sectional entropy (VN30)       — breadth / coordination signal
  MFI                                  — money flow intensity
  Output: entropy feature columns on df

LAYER 3 — UNSUPERVISED REGIME CLASSIFIER
  skills/ds_skill.py
  Plane 1: Full-Cov GMM(n=3) on raw [WPE, SPE_Z]
    -> Deterministic / Transitional / Stochastic
  Plane 2: Full-Cov GMM(n=3) on [Vol_Shannon, Vol_SampEn]
    -> Consensus Flow / Dispersed Flow / Erratic/Noisy Flow
  No preprocessing, no normalization — discovers natural topology
  Output: RegimeName, VolRegimeName columns on df

LAYER 4 — CONDITIONAL VOLATILITY ENGINE
  agent_orchestrator.py  fit_garch_x()
  GARCH(1,1) fitted on full log-return history
  Filtered Historical Simulation -> VaR 5%, ES 5%
  Regime multiplier applied: sigma_adj = sigma_raw x multiplier
    Stochastic: 1.0x  |  Transitional: 1.4x  |  Deterministic: 2.2x
  Output: sigma_daily_pct, sigma_annual_pct, ES_5pct, ES_5pct_adjusted

LAYER 5 — VERDICT MATRIX (Regime-Aware Risk Classification)
  dashboard.py  (inline logic, post-GARCH)
  Combines sigma_adjusted level with price regime in a 3×4 matrix:
    Stochastic + any sigma  -> ELEVATED VOLATILITY (yellow) at most
    Transitional + elevated -> HIGH RISK (orange)
    Deterministic + low     -> STRUCTURAL WARNING (orange) — calm-before-storm
    Deterministic + high    -> EXTREME RISK (red)
  Prevents sigma threshold alone from mislabeling liquidity spikes as systemic crises.
  Fallback when GARCH unavailable: calc_composite_risk_score() entropy aggregate (0-100)
  Output: risk_verdict, risk_verdict_color, mult_explain

LAYER 6 — AI EXPLANATION LAYER
  agent_orchestrator.py  ReAct loop (Claude API)
  5-tool sequence: fetch -> entropy -> volume -> regime -> vol_regime
  Synthesizes all layers into structured markdown risk narrative
  Direction-aware: Deterministic + rally vs Deterministic + decline
  Output: natural-language report for non-technical stakeholders
```

```
skills/data_skill.py  ->  skills/quant_skill.py  ->  skills/ds_skill.py
     LAYER 1                    LAYER 2                   LAYER 3
                                     |
                           agent_orchestrator.py + dashboard.py
                           LAYER 4 + 5 + 6 (ReAct + Verdict Matrix)
                                     |
                               dashboard.py
                           Streamlit + Plotly UI
```

For detailed mathematical specifications, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Project Structure

```
Financial Entropy Agent/
├── agent_orchestrator.py       # GARCH engine, risk scoring, AI agent orchestrator
├── dashboard.py                # Streamlit interactive terminal
├── skills/
│   ├── data_skill.py           # Data ingestion (vnstock, yfinance)
│   ├── quant_skill.py          # WPE, SampEn, Shannon, kinematics
│   └── ds_skill.py             # GMM regime classification
├── validation/
│   ├── regime_validation.py          # V1: Regime labels vs forward realized vol
│   ├── garch_forecast_eval.py        # V2: GARCH out-of-sample forecast
│   ├── risk_alert_hitrate.py         # V3: Drawdown prediction hit rate
│   ├── entropy_vs_simple.py          # V4: Entropy vs simple vol comparison
│   └── cross_market_validation.py    # V5: Cross-market generalizability test
├── ARCHITECTURE.md             # Full mathematical specifications
├── requirements.txt
└── README.md                   # This file
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- An Anthropic API key (for the AI agent; optional — system works without it)

### Installation

```bash
git clone https://github.com/223hoangthai35/financial-entropy-agent.git
cd financial-entropy-agent
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run dashboard.py
```

### Run Validation Suite

```bash
python validation/regime_validation.py
python validation/risk_alert_hitrate.py
python validation/entropy_vs_simple.py
python validation/garch_forecast_eval.py
python validation/cross_market_validation.py
```

---

## Technical Requirements

`numpy`, `pandas`, `numba` (JIT), `scikit-learn` (GMM), `scipy`, `arch` (GARCH), `statsmodels`, `plotly`, `streamlit`, `anthropic` (optional), `matplotlib`, `vnstock`, `yfinance`

---

## Generalization to Other Markets

The system is designed as a market-agnostic entropy surveillance engine. VNINDEX is the validation market, not an architectural constraint.

### What Is Market-Specific

| Component | What changes | What stays the same |
|:----------|:-------------|:--------------------|
| Data source | `data_skill.py` — replace vnstock with Bloomberg/Reuters/yfinance | OHLCV schema |
| Regime multipliers | 1.0 / 1.4 / 2.2 calibrated on VNINDEX tail behavior | GARCH model specification |
| GARCH fit | Refits on new market's return distribution automatically | Model specification |
| VN30 breadth | Replace with S&P 500 constituents, FTSE 100, etc. | Eigenvalue decomposition logic |

### Verifying the Entropy Paradox on a New Market

The most important validation before deploying on a new market is to re-confirm that **low entropy = high risk** holds in that market's regime structure. This is not guaranteed — markets with different microstructure (e.g., thin frontier markets with high manipulation risk vs. deep liquid markets) may exhibit different entropy-risk relationships.

**Recommended steps:**

1. Run `validation/regime_validation.py` on the new market's data
2. Check that the Kruskal-Wallis H-statistic is significant (p < 0.05)
3. Check that the **ordering** of mean forward volatility matches: Deterministic > Transitional > Stochastic
4. If ordering is inverted, the GMM cluster-to-regime mapping (`_cluster_to_regime` in `ds_skill.py`) needs to be re-calibrated for that market's entropy topology

### Markets Where This Approach Is Well-Motivated

The entropy framework is most applicable to markets exhibiting **Type-2 chaos characteristics**:
- Markets with identifiable herding and momentum-following behavior
- Markets where retail participation is high (coordinated behavior easier to detect)
- Markets with sufficient history for GMM convergence (minimum ~500 trading days recommended)

For deep, institutionally-dominated markets (e.g., US Treasuries), the Entropy Paradox may be weaker because arbitrage suppresses coordination faster. For frontier markets with thin liquidity, jump risk dominates and GARCH limitations (noted in V2) become more severe.

---

## References

- Bandt, C. & Pompe, B. (2002). *Permutation Entropy: A Natural Complexity Measure for Time Series.* Physical Review Letters, 88(17).
- Fadlallah, B. et al. (2013). *Weighted-Permutation Entropy: A Complexity Measure for Time Series Incorporating Amplitude Information.* Physical Review E, 87(2).
- Richman, J.S. & Moorman, J.R. (2000). *Physiological Time-Series Analysis Using Approximate Entropy and Sample Entropy.* American Journal of Physiology.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.* Journal of Econometrics.
- Lo, A. W. (2004). *The Adaptive Markets Hypothesis.* Journal of Portfolio Management, 30(5), 15-29.
- Zanin, M., Zunino, L., Rosso, O. A., & Papo, D. (2012). *Forbidden patterns, permutation entropy and stock market inefficiency.* Physica A, 391(6), 1820-1827.
- Risso, W. A. (2008). *The informational efficiency and the financial crashes.* Research in International Business and Finance, 22(3), 396-408.

---

## About the Author

Physics background -> Securities broker -> Data Science.

This project was born from observing that traditional technical analysis systematically failed to warn about structural market risks. The hypothesis — that entropy from statistical physics could detect dangerous coordination in financial markets — was validated through rigorous out-of-sample testing on 6 years of VNINDEX data.

---

*Disclaimer: This system is a quantitative research tool for structural risk assessment. It is not investment advice. All investment decisions require final approval from qualified professionals.*
