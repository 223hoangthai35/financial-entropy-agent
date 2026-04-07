# Financial Entropy Agent

### A Tri-Vector Phase Space Surveillance System for Systemic Risk Quantification

---

## Abstract

**Financial Entropy Agent** is an institutional-grade systemic risk surveillance engine that decodes market microstructure through the principles of **Symbolic Dynamics**, **Information Theory**, and **Statistical Physics**. The system replaces all heuristic-based Technical Analysis (TA) with continuous, mathematically rigorous models rooted in statistical mechanics.

The core innovation is the **Raw Entropy Phase Space** -- a 2D space defined by Weighted Permutation Entropy (structural order) and Standardized Price Sample Entropy (price predictability), where a Full-Covariance Gaussian Mixture Model discovers the natural topological boundaries of market regimes (Stable, Fragile, Chaos) without any PowerTransform preprocessing or human-imposed boundary constraints.

Kinematic indicators (velocity and acceleration of entropy) are strictly decoupled from the unsupervised learning pipeline and reserved as Explainable AI (XAI) trajectory descriptors for the autonomous LLM agent.

---

## Core Philosophy: The Paradigm Shift

Traditional financial analysis relies on pattern-matching heuristics: moving averages, RSI thresholds, support/resistance levels. These tools assume stationarity and normality in price series -- assumptions that are systematically violated by real markets.

This system operates on a fundamentally different premise:

| Dimension | Traditional TA | Financial Entropy Agent |
|:---|:---|:---|
| **Signal** | Price level, amplitude | Ordinal pattern entropy (WPE) + Price trajectory complexity (SPE_Z) |
| **Dynamics** | Trend-following | Kinematic velocity and acceleration of entropy (XAI only) |
| **Regime Detection** | Fixed thresholds | Raw Full-Covariance GMM in natural Phase Space |
| **Risk Scoring** | Rule-based if/else | Continuous tri-vector composite index (0-100) |
| **Normalization** | Z-score (assumes normality) | Yeo-Johnson PowerTransform (risk engine only) |
| **Threshold Calibration** | Expert guesses | Empirical quantiles (P75/P90) from rolling distribution |

---

## Methodology

This section presents the formal mathematical framework underpinning the Financial Entropy Agent.

### A. Theoretical Rationale for Feature Selection

The system employs three distinct entropy measures, each selected for specific mathematical properties that make it superior to conventional alternatives for its designated measurement domain.

#### A.1 Weighted Permutation Entropy (WPE) -- Structural Order

**Selected for:** Ordinal pattern analysis of price log-returns (Plane 1, X-axis).

**Rationale:** Standard Shannon Entropy applied to price data suffers from two fundamental limitations: (1) it is sensitive to binning parameters and (2) it discards all temporal correlation structure. Permutation Entropy (Bandt & Pompe, 2002) addresses both by analyzing the ordinal rank patterns of consecutive observations, capturing the **causal temporal structure** of the time series.

The amplitude-weighted extension (Fadlallah et al., 2013) further resolves a critical weakness of standard PE: identical ordinal patterns receive equal weight regardless of whether the underlying magnitude change is 0.01% or 5%. WPE incorporates amplitude information into the probability weighting, making it simultaneously sensitive to both **temporal order** and **volatility regime**.

**Mathematical superiority over Shannon Entropy:**

| Property | Shannon $H$ | Permutation Entropy $H_{PE}$ | Weighted PE $H_{WPE}$ |
|:---|:---|:---|:---|
| Temporal structure | None | Ordinal ranks | Ordinal ranks |
| Amplitude sensitivity | Via binning (arbitrary) | None | Weighted probabilities |
| Stationarity requirement | Strong | Weak | Weak |
| Robustness to outliers | Low | High | Moderate |
| Parameter dependence | Bin width | $m$, $\tau$ only | $m$, $\tau$ only |

#### A.2 Sample Entropy (SampEn) -- Price Predictability & Volume Complexity

**Selected for:** Trajectory complexity measurement on close prices (Plane 1, Y-axis as SPE_Z) and volume micro-structure (Plane 2 / Vector 2).

**Rationale:** Sample Entropy (Richman & Moorman, 2000) is chosen over Approximate Entropy ($ApEn$) for two critical reasons:

1. **No self-matching bias.** $ApEn$ counts self-matches (comparing a template vector against itself), which introduces a systematic bias that inflates entropy estimates for short time series. $SampEn$ excludes self-matches, providing an unbiased estimate of conditional probability.

2. **Relative consistency.** $ApEn$ is not consistent -- its value depends on the data length $N$ in a non-trivial way. $SampEn$ exhibits superior relative consistency, making cross-comparison between rolling windows meaningful.

**Why SampEn on close prices (not log-returns) for SPE_Z:** Log-returns discard absolute price level information. By computing SampEn directly on close prices, the metric captures **trajectory complexity** -- how predictable the price path is in amplitude space. This is orthogonal to WPE, which measures ordinal pattern disorder in log-returns.

#### A.3 Shannon Entropy -- Volume Distribution Analysis

**Selected for:** Concentration/dispersion measurement of volume flow (Vector 2).

**Rationale:** For volume analysis, the goal is to measure **how evenly capital is distributed** across a discrete set of bins within a rolling window. This is the classical domain of Shannon Entropy -- measuring the uniformity of a probability distribution over discrete categories. Unlike WPE (which requires temporal ordering) or SampEn (which requires continuous template matching), Shannon's formulation is optimal for answering: "Is volume concentrated in a few extreme bars, or evenly dispersed?"

$$H_{Shannon} = -\frac{1}{\ln(K)} \sum_{k=1}^{K} p_k \cdot \ln(p_k)$$

where $K$ is the number of histogram bins and $p_k$ is the fraction of volume in bin $k$. The normalization by $\ln(K)$ constrains $H \in [0, 1]$.

---

### B. Mathematical Formulations

#### B.1 Weighted Permutation Entropy (WPE)

Given a time series $\{x_t\}_{t=1}^{N}$, construct embedding vectors of dimension $m$ with delay $\tau$:

$$\mathbf{X}_t = (x_t, x_{t+\tau}, x_{t+2\tau}, \ldots, x_{t+(m-1)\tau})$$

Each embedding vector maps to a unique ordinal permutation $\pi$. The amplitude-weighted probability of permutation $\pi$ is:

$$\hat{p}_w(\pi) = \frac{\sum_{t : \text{ord}(\mathbf{X}_t) = \pi} w_t}{\sum_{t=1}^{N-(m-1)\tau} w_t}$$

where $w_t = \frac{1}{m} \sum_{k=0}^{m-1} |x_{t+k\tau} - \bar{\mathbf{X}}_t|$ is the amplitude weight (mean absolute deviation).

The Weighted Permutation Entropy is then:

$$H_{WPE} = -\frac{1}{\ln(m!)} \sum_{\pi} \hat{p}_w(\pi) \cdot \ln(\hat{p}_w(\pi))$$

**Output:** $H_{WPE} \in [0, 1]$ where 0 = perfect order, 1 = maximum disorder.

#### B.2 Sample Entropy (SampEn)

For a time series $\{x_t\}_{t=1}^{N}$, define template vectors of length $m$:

$$\mathbf{u}_i^{(m)} = (x_i, x_{i+1}, \ldots, x_{i+m-1}) \quad \text{for } i = 1, \ldots, N-m+1$$

Count template matches using the Chebyshev distance:

$$B_i^{(m)}(r) = \frac{1}{N-m} \sum_{\substack{j=1 \\ j \neq i}}^{N-m} \mathbb{1}\left[\max_{k=0,\ldots,m-1} |x_{i+k} - x_{j+k}| < r\right]$$

Define:

$$B^{(m)}(r) = \frac{1}{N-m+1} \sum_{i=1}^{N-m+1} B_i^{(m)}(r)$$

$$A^{(m)}(r) = \frac{1}{N-m} \sum_{i=1}^{N-m} A_i^{(m)}(r)$$

where $A_i^{(m)}$ counts matches of length $m+1$. Then:

$$SampEn(m, r, N) = -\ln\frac{A^{(m)}(r)}{B^{(m)}(r)}$$

**Global Z-Score Normalization (SPE_Z):**

$$SPE\_Z_t = \frac{SampEn_t - \mu_{SampEn}}{\sigma_{SampEn}}$$

where $\mu$ and $\sigma$ are computed over all valid rolling windows in the dataset.

#### B.3 WPE Kinematics (XAI Trajectory Indicators)

Discrete first and second differences of the WPE time series:

$$V_{WPE}(t) = WPE(t) - WPE(t-1)$$

$$a_{WPE}(t) = V_{WPE}(t) - V_{WPE}(t-1)$$

These are **strictly reserved for Explainable AI (XAI) narrative generation**. They are NOT used as inputs to the GMM classifier or the Composite Risk Engine.

#### B.4 Yeo-Johnson Power Transform

The Yeo-Johnson transformation (Yeo & Johnson, 2000) is applied **exclusively in the Composite Risk Scoring pipeline** (Module B) to correct left-skewness in entropy distributions before linear weighting:

$$\psi(x, \lambda) = \begin{cases}
\frac{(x+1)^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0, x \geq 0 \\
\ln(x+1) & \text{if } \lambda = 0, x \geq 0 \\
-\frac{(-x+1)^{2-\lambda} - 1}{2-\lambda} & \text{if } \lambda \neq 2, x < 0 \\
-\ln(-x+1) & \text{if } \lambda = 2, x < 0
\end{cases}$$

The optimal $\lambda$ is estimated via maximum likelihood on the 504-day rolling history.

**Where it is used:** Normalizing each of the three risk vectors (V1, V2, V3) before MinMaxScaler and weighted summation.

**Where it is NOT used:** GMM clustering on Plane 1 and Plane 2. Raw features are fed directly into the GMM to preserve natural topological boundaries.

#### B.5 Tri-Vector Composite Risk Equation

$$Risk_{sys} = \left(\sum_{i=1}^{3} w_i \cdot \overline{\text{MinMax}\left(\text{YeoJohnson}(\mathbf{V}_i)\right)}\right) \times 100$$

Where:

$$w_1 = 0.4 \text{ (Price Phase Space)}, \quad w_2 = 0.4 \text{ (Liquidity Depth)}, \quad w_3 = 0.2 \text{ (Structural Breadth)}$$

And each vector is processed as:

$$\text{Scaled}_i = \overline{\text{MinMax}_{[0,1]}\left(\text{YeoJohnson}_{\lambda_i}(\mathbf{V}_i^{hist})\right)}$$

With the current day's features transformed using the **history-fitted** parameters (no data leakage).

**Dynamic Risk Thresholds:**

$$\text{Elevated bound} = P_{75}\left(\{Risk_{sys,t}\}_{t=T-503}^{T}\right)$$

$$\text{Critical bound} = \max\left(P_{90}\left(\{Risk_{sys,t}\}_{t=T-503}^{T}\right), \text{ Elevated bound} + 3.0\right)$$

---

### C. Hyperparameters & Settings

#### C.1 Core Parameters

| Parameter | Symbol | Value | Domain | Rationale |
|:---|:---|:---|:---|:---|
| WPE embedding dimension | $m$ | 3 | Price WPE | Captures ordinal patterns of length 3 ($m! = 6$ possible permutations). Optimal for daily market data. |
| WPE delay | $\tau$ | 1 | Price WPE | Consecutive observations. No sub-sampling. |
| WPE rolling window | $w$ | 22 | Price WPE | ~1 trading month. Balances resolution vs. stability. |
| Price SampEn template length | $m$ | 2 | SPE_Z | Standard for financial SampEn. |
| Price SampEn tolerance | $r$ | $0.2 \cdot \sigma_{win}$ | SPE_Z | Adaptive threshold scaled to local volatility. |
| Price SampEn window | -- | 60 | SPE_Z | ~3 trading months. Sufficient points for robust SampEn. |
| Volume SampEn template length | $m$ | 2 | Volume | Consistent with price SampEn. |
| Volume SampEn tolerance | $r$ | 0.2 | Volume | Fixed tolerance for volume z-score series. |
| Volume entropy window | -- | 60 | Volume | Matched to price SampEn window. |
| Volume Global Z-Score window | -- | 252 | Volume | ~1 trading year. Macro-scale baseline. |

#### C.2 Unsupervised Learning Parameters

| Parameter | Value | Component | Rationale |
|:---|:---|:---|:---|
| GMM components | $n = 3$ | Plane 1 & 2 | Three market regimes: Stable, Fragile, Chaos (Price) / Consensus, Dispersed, Erratic (Volume). |
| Covariance type | `full` | Plane 1 & 2 | Each cluster has its own 2x2 covariance matrix. Captures true geometric cluster shapes. |
| GMM initialization runs | $n_{init} = 10$ | Plane 1 & 2 | Reduce sensitivity to random initialization. |
| Max EM iterations | 500 | Plane 1 & 2 | Convergence guarantee. |
| PowerTransform (Plane 1) | **NONE** | GMM input | Raw features preserve natural topology. |
| Label sorting (Plane 1) | Sum of centroid means | Regime naming | $\text{score}_k = \text{WPE}_k + \text{SPE\_Z}_k$. Lowest = Stable, Highest = Chaos. |
| Label sorting (Plane 2) | Centroid X-mean | Regime naming | Sorted by Shannon entropy centroid. |

#### C.3 Risk Engine Parameters

| Parameter | Value | Rationale |
|:---|:---|:---|
| Rolling window | 504 days (~2 trading years) | Captures at least one full market cycle. |
| PowerTransformer method | Yeo-Johnson | Handles both positive and negative values (required for SPE_Z, Global Z). |
| MinMaxScaler range | [0, 1] | Ensures all vectors are on comparable scale before weighting. |
| Weights | $w_1 = 0.4$, $w_2 = 0.4$, $w_3 = 0.2$ | Price and Volume are co-primary; Breadth is supplementary. |
| Elevated threshold | P75 of rolling 504-day scores | Top quartile indicates structural divergence. |
| Critical threshold | P90 (min separation: +3.0 from P75) | Top decile indicates phase transition. |

---

## Architecture Overview

```
                        FINANCIAL ENTROPY AGENT
                    =======================================

                    [ RAW OHLCV & VN30 DATA ]
                               |
                               +-------------------------------------------------+
                               |                                                 |
                  [ MODULE A: UNSUPERVISED GMM ]                [ MODULE B: COMPOSITE RISK ENGINE ]
                  (Dual-Plane Visual Diagnostics)               (Tri-Vector Mathematical Synthesis)
                               |                                                 |
                         +-----+-----+                             +-------------+-------------+
                         |           |                             |             |             |
                      PLANE 1     PLANE 2                      VECTOR 1      VECTOR 2      VECTOR 3
                      (Price)     (Volume)                  (Price Phase)    (Volume)     (VN30 Breadth)
                         |           |                             |             |             |
                     RAW WPE      Shannon                   WPE / |SPE_Z|  SampEn / GZ    CorrEnt / MFI
                     RAW SPE_Z    SampEn                         |             |             |
                         |           |                           +-------------+-------------+
                     Full GMM     Full GMM                                     |
                    (3 Regimes)  (3 Regimes)                       PowerTransformer (Yeo-Johnson)
                   (NO transform)                                              |
                                                                         MinMaxScaler [0, 1]
                         |                                                     |
                 V_WPE, a_WPE                             Weighted Sum: 0.4*V1 + 0.4*V2 + 0.2*V3
                 (XAI Overlay)                                                 |
                                                                Composite Risk Score (0-100)
                                                                               |
                                                                P75/P90 Rolling 504-day
                                                                               |
                                                             STABLE / ELEVATED / CRITICAL
```

For the full technical specification, see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Technical Requirements

- **Python**: 3.9+
- **Core Dependencies**: `numpy`, `pandas`, `numba` (JIT optimization), `scikit-learn` (GMM, PowerTransformer, MinMaxScaler), `scipy`
- **Dashboard**: `streamlit`, `plotly`
- **AI Orchestrator**: `anthropic` (requires `ANTHROPIC_API_KEY`)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Launch the Dark Quant Terminal
streamlit run dashboard.py
```

---

## Project Structure

```
Financial Entropy Agent/
|-- agent_orchestrator.py    # Tri-Vector Composite Risk Engine + ReAct Orchestrator
|-- dashboard.py             # Dark Quant Terminal (Streamlit + Plotly)
|-- ARCHITECTURE.md          # Detailed technical architecture document
|-- README.md                # This file (Research Proposal + Mathematics)
|-- skills/
|   |-- data_skill.py        # Data ingestion (vnstock, yfinance)
|   |-- quant_skill.py       # WPE, SampEn (Price + Volume), Shannon, EVD, Kinematics
|   |-- ds_skill.py          # Raw Full GMM (Plane 1) + PowerTransform Volume GMM (Plane 2)
```

---

## Institutional Use Cases & Practical Applications

This system is not a signal generator. It is a **macro-structural risk filter** designed to operate upstream of any trading strategy, providing a continuous, physics-based assessment of systemic market health.

### 1. Systemic Risk Filter for Dynamic Position Sizing

Quantitative funds and systematic portfolio managers can integrate the **Composite Risk Score (0-100)** into their position sizing and margin allocation frameworks.

| Composite Risk Score | Recommended Action |
|:---|:---|
| **0 - 40 (Low)** | Full allocation (100%). Market structure exhibits systemic coherence. Entropy is low, liquidity is orderly. |
| **40 - P75 (Moderate)** | Standard allocation. Monitor XAI trajectory (V_WPE, a_WPE) for directional entropy shifts. |
| **P75 - P90 (Elevated)** | Reduce gross exposure by 50%. Halt new position entries. Structural divergence detected across one or more vectors. |
| **> P90 (Critical)** | Maximum risk reduction. Initiate hedging protocols. Phase transition imminent -- top-decile extreme event in the rolling 504-day distribution. |

### 2. Detecting Liquidity Divergences (Trap Detection)

The decoupled multi-plane architecture enables the detection of **hollow rallies (bull traps)** and **false breakdowns**:

- **Hollow Rally**: Plane 1 (Price) shows "Stable" regime, but Plane 2 (Volume) simultaneously classifies as "Erratic/Dispersed". Price stability is not supported by consensus liquidity flow.
- **Capitulation Vacuum**: Plane 1 shows "Chaos" but `Vol_Global_Z` is negative. High price entropy is driven by illiquidity, not genuine institutional selling.
- **Climax Distribution**: High Composite Risk score but `Vol_Global_Z` is strongly positive. Excess liquidity is flowing into a structurally deteriorating market -- peak FOMO.

### 3. Measuring Internal Structural Health (Sector Rotation Detection)

Vector 3 (VN30 Breadth) provides a unique diagnostic layer invisible to index-level analysis:

- **Correlation Entropy < 40%**: The VN30's movement is dominated by a narrow set of heavyweight pillars. The "propped-up" structure is inherently fragile.
- **Correlation Entropy > 70%**: The component stocks are moving in highly independent directions. Aggressive internal sector rotation or capital flight.

---

## Academic References

- Bandt, C. & Pompe, B. (2002). *Permutation Entropy: A Natural Complexity Measure for Time Series*. Physical Review Letters, 88(17).
- Fadlallah, B. et al. (2013). *Weighted-Permutation Entropy: A Complexity Measure for Time Series Incorporating Amplitude Information*. Physical Review E, 87(2).
- Richman, J.S. & Moorman, J.R. (2000). *Physiological Time-Series Analysis Using Approximate Entropy and Sample Entropy*. American Journal of Physiology.
- Yeo, I.K. & Johnson, R.A. (2000). *A New Family of Power Transformations to Improve Normality or Symmetry*. Biometrika, 87(4).
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

---

**Disclaimer**: This system is designed as a quantitative research tool for institutional systemic risk surveillance. It is not a trading signal generator. All investment decisions based on its output require final approval from qualified human analysts.
