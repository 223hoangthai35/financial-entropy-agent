# Financial Entropy Agent: Dual-Plane Systemic Risk Engine

## Abstract

The **Financial Entropy Agent** is a Multi-Modal Systemic Risk Engine that monitors financial
market structural integrity through two independent, unsupervised observation planes. Moving beyond
single-variable forecasting or lagging technical indicators, this system employs concepts from
**Symbolic Dynamics**, **Information Theory**, and **Non-linear Physics** to decouple market noise
from intrinsic structural regimes.

The architecture observes the market through two lenses simultaneously:

- **Plane 1 -- Price Dynamics**: Measures the *physical chaos* of price action via Weighted
  Permutation Entropy and Annualized Volatility.
- **Plane 2 -- Liquidity Structure (Macro-Micro Fusion)**: Fuses **Global Z-Scores** (to measure
  absolute macro-scale systemic liquidity) with **Rolling Z-Scores + Entropy** (to measure
  localized micro-structural trading behavior without structural break bias).

An autonomous AI Agent (Anthropic Claude) acts as the **Cross-Plane Reasoning Engine**, synthesizing
both planes to identify systemic conditions that are invisible from a single observation space --
such as Smart Money accumulation masked by price chaos, or trend exhaustion hidden beneath stable
prices.

---

## 1. Architecture Overview

```
+-------------------------------------------------+
|             agent_orchestrator.py               |
|         Cross-Plane Reasoning Engine            |
|       (Macro-Micro Fusion & ReAct Loop)         |
+-------------------------------------------------+
         /                |                \
+-------------+   +---------------+   +---------------+
| data_skill  |   |  quant_skill  |   |   ds_skill    |
+-------------+   +---------------+   +---------------+
| Market Data |   | PE, V, a (P1) |   | GMM Clusters  |
| VN30 fetch  |   | Vol Pipeline  |   | Regime Map    |
+-------------+   +-------+-------+   +---------------+
          |               |                   |
          |       [PLANE 1: PRICE]     [PLANE 2: VOLUME]
          |       Physics Dynamics     Macro-Micro Fusion
          |               |            /             \
          v               v           v               v
    [Index Data]    [WPE, V, a]   [Macro Z]   [Rolling Z + Ent]
          |               |           |               |
          |               |           |               |
    +=====v===============v===+   +===v===============v=====+
    |         PLANE 1         |   |         PLANE 2         |
    |     Price Dynamics      |   |     Liquidity Space     |
    +=========================+   +=========================+
                  \                         /
                   \                       /
             +=============================+
             |    CROSS-PLANE SYNTHESIS    |
             | (Price Chaos + Liquidity F.) |
             +=============================+
```

> **Macro-Micro Fusion (Plane 2):** Plane 2 utilizes a dual-path preprocessing architecture.
> Raw volume is `log1p`-transformed, then split into: **(A) Global Z-Score** for absolute
> macro-scale measurement (e.g., FDI inflows, systemic liquidity drought), and **(B) 252-day
> Rolling Z-Score** which feeds the Shannon/SampEn entropy calculations. This ensures entropy
> metrics capture localized trading behavior resistant to long-term structural breaks.


| Observation Plane | X-Axis | Y-Axis | Measures |
|---|---|---|---|
| **Plane 1: Price Dynamics** | Weighted Permutation Entropy (WPE) | Annualized Volatility | Physical Chaos |
| **Plane 2: Liquidity Structure** | Volume Shannon Entropy | Volume Sample Entropy | Capital Flow Structure |

---

## 2. Mathematical Foundations and Hyperparameters

### 2.1 Weighted Permutation Entropy (WPE) -- Plane 1

Permutation Entropy measures the structural orderliness of a time series by analyzing the frequency
distribution of **ordinal patterns** (rank sequences) within sliding windows. Unlike amplitude-based
entropy measures, PE captures the *temporal order* of data points, making it fundamentally appropriate
for non-stationary financial time series where absolute price levels are less informative than
relative movements.

**Weighted Permutation Entropy** extends this by incorporating the variance (amplitude) of each
pattern, preventing information loss from rank-only analysis:

$$H_{WPE} = -\frac{1}{\ln(d!)} \sum_{i=1}^{d!} p_i^{(w)} \ln\left(p_i^{(w)}\right)$$

Where:
- $d = 3$ -- Embedding dimension. Defines the length of ordinal patterns extracted from the time
  series. $d = 3$ yields $3! = 6$ possible permutations, balancing granularity against statistical
  reliability.
- $\tau = 1$ -- Time delay between consecutive data points. Standard for daily granularity.
- $p_i^{(w)}$ -- Variance-weighted frequency of the $i$-th ordinal pattern. Weights proportional
  to the variance within each pattern window ensure that high-amplitude events contribute more to
  the entropy estimate.
- Rolling window: **22 trading days** (approximately 1 calendar month).

**Interpretation**: $H_{WPE} \to 0$ implies deterministic, trending structure. $H_{WPE} \to 1$
implies maximum disorder -- random, structureless price action.

### 2.2 Market Fragility Index (MFI)

The MFI combines entropy with statistical complexity to quantify how close the market is to a
structurally fragile state:

$$\text{MFI} = H_{WPE} \times (1 - C_{JS})$$

Where $C_{JS}$ is the Jensen-Shannon Statistical Complexity -- a measure of the distance between
the observed ordinal distribution and a uniform (maximum entropy) distribution. High MFI indicates
simultaneously high disorder and low complexity, the signature of structural fragility.

> *Note: While the theoretical 2D Complexity-Entropy Causality Plane (CECP) boundaries are not
> visualized on the dashboard (which prioritizes Volatility for Plane 1), MFI serves as a 1D
> dimensional reduction of that space. It successfully embeds the structural complexity ($C_{JS}$)
> into a single actionable metric.*

### 2.3 Kinematic Physics Vectors -- Plane 1

Beyond the static entropy value, the system computes the **first and second derivatives** of
Permutation Entropy to capture dynamic momentum -- analogous to velocity and acceleration in
classical mechanics:

$$V = \frac{dE}{dt} \approx \Delta_3 H_{WPE} = H_{WPE}(t) - H_{WPE}(t-3)$$

$$a = \frac{d^2E}{dt^2} \approx \Delta_3 V = V(t) - V(t-3)$$

| Vector | Formula | Interpretation |
|---|---|---|
| **V (Velocity)** | `WPE.diff(3)` | Direction of entropy change. $V > 0$: chaos expanding. $V < 0$: order forming. |
| **a (Acceleration)** | `V.diff(3)` | Force of momentum. $a > 0$: trend accelerating (exploding). $a < 0$: momentum fading (exhausting). |

**Why 3-day momentum**: A 1-day diff captures excessive noise from daily microstructure artifacts.
A 3-day window filters noise while remaining responsive to genuine structural shifts. NaN values
from the differencing operation are filled with `0` to prevent breaking downstream GMM clustering
and JSON serialization.

**Kinematic Heuristic Matrix**:

| V | a | Interpretation |
|---|---|---|
| $V > 0$ | $a > 0$ | Chaos accelerating. Structural breakdown in progress. |
| $V > 0$ | $a < 0$ | Chaos expanding but decelerating. Peak disorder may be near. |
| $V < 0$ | $a < 0$ | Order forming and accelerating. Recovery gaining momentum. |
| $V < 0$ | $a > 0$ | Order forming but decelerating. Recovery may stall. |

### 2.4 Volume Sample Entropy (SampEn) -- Plane 2

Sample Entropy quantifies the **regularity of volume impulse patterns**. It measures the conditional
probability that two sequences of $m$ consecutive data points that are similar (within tolerance $r$)
will remain similar when extended to $m+1$ points.

$$\text{SampEn}(m, r, N) = -\ln\left(\frac{A}{B}\right)$$

Where:
- $A$ = Number of template matches for sequences of length $m+1$
- $B$ = Number of template matches for sequences of length $m$

**Hyperparameters and Justification**:

| Parameter | Value | Justification |
|---|---|---|
| $m$ (Embedding Dimension) | **2** | Captures 2-day micro-structural patterns in volume impulses. Higher $m$ would require exponentially more data ($N \geq 10^m$) for convergence. |
| $r$ (Tolerance) | **$0.2$** (fixed) | Since the input to SampEn is a **Rolling Z-Score** (unit variance by construction), the tolerance is fixed at $0.2$ standard deviations. This eliminates the need for adaptive `0.2 * std(x)` scaling. |
| Window | **60 trading days** | Satisfies the convergence requirement $N \geq 10^m = 100$ at the practical minimum. 60 trading days (~3 calendar months) balances statistical reliability against responsiveness to regime shifts. |
| Pre-processing | **Rolling Z-Score of $\log(1 + V)$** | Raw volume is first `log1p`-transformed to stabilize heavy tails, then converted to a 252-day Rolling Z-Score. This ensures entropy measures capture *relative structural behavior* rather than absolute magnitude, and resists contamination from long-term structural breaks. |

**Interpretation**:
- **Low SampEn**: Volume impulses are regular and predictable -- characteristic of institutional,
  algorithmic flow (Consensus Flow).
- **High SampEn**: Volume impulses are irregular and unpredictable -- characteristic of fragmented
  retail activity or panic-driven trading (Erratic Flow).

**Performance Note**: The $O(N^2)$ complexity of the template-matching algorithm necessitates
`@numba.njit` JIT compilation for real-time dashboard responsiveness.

### 2.5 Volume Shannon Entropy -- Plane 2

Shannon Entropy measures the **concentration or dispersion** of volume distribution:

$$H_{Shannon} = -\sum_{i=1}^{k} p_i \log_2(p_i) \quad \text{(normalized to [0, 1])}$$

Where $p_i$ is the probability of volume falling in the $i$-th histogram bin.

**Hyperparameters and Justification**:

| Parameter | Value | Justification |
|---|---|---|
| Binning Strategy | **`bins='auto'`** | Delegates to NumPy's automatic bin selection (Freedman-Diaconis or Sturges rule). This is critical for financial volume data: hardcoded bins (e.g., `bins=10`) create numerous empty bins due to the heavy-tailed distribution, artificially deflating entropy. The Freedman-Diaconis rule adapts bin width based on the Interquartile Range, naturally handling outliers and volume spikes. |
| Normalization | **$H / H_{max}$** | Divides by $\log_2(k)$ to normalize to $[0, 1]$ range, enabling cross-comparison regardless of the number of bins. |

**Interpretation**:
- **Low Shannon**: Volume is concentrated in a narrow range -- institutional, consensus-driven flow.
- **High Shannon**: Volume is dispersed across many levels -- fragmented, multi-source activity.

### 2.6 Cross-Sectional Correlation Entropy (VN30)

Measures the structural fragmentation across the VN30 blue-chip basket:

1. Compute the Pearson Correlation Matrix of VN30 component daily returns.
2. Perform Eigenvalue Decomposition (EVD).
3. Convert eigenvalues to a probability distribution and compute Shannon Entropy.
4. Scale to $[0, 100]$.

**Interpretation**: Low entropy = centralized capital consensus (blue chips move together).
High entropy = extreme fragmentation (decorrelated sector rotation).

---

## 3. Unsupervised Learning: Gaussian Mixture Models

Both observation planes employ **Gaussian Mixture Models (GMM)** with $n = 3$ components to
discover hidden market regimes without human-labeled bias.

### 3.1 Price Regime Classification (Plane 1)

| Features | $[H_{WPE}, C_{JS}, \text{MFI}]$ |
|---|---|
| **Preprocessing** | `StandardScaler` (zero mean, unit variance) |
| **Covariance Type** | `full` (no assumption on feature independence) |
| **Label Mapping** | Sort clusters by mean feature value (ascending) |

| Regime | Characteristics |
|---|---|
| **Stable Growth** | Low WPE, high complexity, low MFI. Structured, deterministic market. |
| **Fragile Growth** | Moderate WPE, declining complexity. Ordinal pattern breakdown beginning. |
| **Chaos/Panic** | High WPE, low complexity, high MFI. Maximum structural fragmentation. |

### 3.2 Volume Regime Classification (Plane 2)

| Features | [Vol<sub>Shannon</sub>, Vol<sub>SampEn</sub>] |
|---|---|
| **Preprocessing** | `StandardScaler` |
| **Label Mapping** | Sort clusters by **sum of centroid coordinates** (Shannon + SampEn) |

| Regime | Characteristics |
|---|---|
| **Consensus Flow** | Low Shannon + Low SampEn. Volume concentrated and regular. Institutional. |
| **Dispersed Flow** | Moderate Shannon + SampEn. Transitional, mixed participation. |
| **Erratic/Noisy Flow** | High Shannon + High SampEn. Fragmented and irregular. Retail/Panic. |

---

## 4. The Agent Orchestrator: Cross-Plane Synthesis

> **Disclaimer (Human-in-the-Loop):** The Agent is designed as a structural telemetry tool. It provides the "map" (Entropy states) and the "vehicle's dashboard" (Kinematic vectors V and a) to interpret complex market dynamics. However, the Agent is strictly an analytical observer. The final decision to act upon these conclusions rests entirely on human judgment and execution strategy.

The LLM Agent (Anthropic Claude) operates as the **Cross-Plane Reasoning Engine** using a
**ReAct (Reasoning and Acting) Loop** with 5 sequential tool calls:

```
[1] fetch_market_data       -->  OHLCV data
[2] compute_entropy_metrics -->  Plane 1: WPE, MFI, V (dE/dt), a (d2E/dt2)
[3] compute_volume_entropy  -->  Plane 2: Shannon, SampEn
[4] predict_market_regime   -->  Price Regime (GMM)
[5] predict_volume_regime   -->  Volume Regime (GMM)
[6] Cross-Plane Synthesis   -->  Unified systemic conclusion
```

### Cross-Plane Synthesis Matrix

The Agent cross-validates Price Physics against Liquidity Structure to prevent false positives:

| Price Plane | Volume Plane | Synthesis | Rationale |
|---|---|---|---|
| Fragile / Chaos | Consensus Flow | **STRUCTURAL ACCUMULATION** | Price chaos is a surface phenomenon. Highly organized institutional liquidity absorbs supply. Smart Money accumulation. |
| Fragile / Chaos | Erratic/Noisy Flow | **CRITICAL BREAKDOWN** | Both planes confirm systemic instability. Fragmented liquidity amplifies price chaos. No structural floor. |
| Stable Growth | Erratic/Noisy Flow | **TREND EXHAUSTION** | Stable prices mask deteriorating liquidity structure. Volume fragmentation precedes price correction. |
| *Other* | *Other* | **SYSTEM COHERENT** | Both planes structurally aligned. No cross-plane divergence. |

**Why Cross-Plane matters**: A single-plane observation would classify "Fragile + Consensus" identically
to "Fragile + Erratic" (both show price stress). Only by observing the volume plane can we
distinguish **accumulation** (bullish) from **breakdown** (bearish) -- the difference between
a buying opportunity and a systemic crisis.

---

## 5. Dashboard: Streamlit Terminal

The dashboard provides three integrated views:

1. **All-in-One Structural Telemetry**: Candlestick chart with WPE overlay + Cross-Sectional
   Entropy / MFI subplot. Regime backgrounds color-coded.
2. **Dual-Plane DS Proof**: Side-by-side GMM scatter plots proving both Price and Volume regimes
   are mathematically discovered without human labels.
3. **Cross-Plane Agent Diagnostic**: Markdown-formatted diagnostic table with structured
   Telemetry Module output, analysis sections, and mandatory conclusion.

---

## 6. Resilience and Portability

- **Dual-Pipeline Routing**: Cloud API (`vnstock`, `yfinance`) with automatic local file fallback
  (CSV / XLSX upload).
- **Numba JIT Compilation**: Critical $O(N^2)$ algorithms (SampEn) and iterative rolling windows
  (WPE) are JIT-compiled for real-time dashboard performance.

---

## 7. How to Run

**Prerequisites:**
- Python 3.9+
- Anthropic API Key (optional -- mock orchestrator available for testing)

**Installation:**

```bash
pip install -r requirements.txt
```

Dependencies: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `numba`, `vnstock`,
`yfinance`, `anthropic`.

**Execution:**

```bash
streamlit run dashboard.py
```

**Agent Orchestrator (standalone test):**

```bash
python agent_orchestrator.py
```

---

## 8. Project Structure

```
Financial Entropy Agent/
|-- agent_orchestrator.py           # Cross-Plane Reasoning Engine (5 tools + synthesis)
|-- dashboard.py                    # Streamlit UI: Dual scatter + agent diagnostic
|-- architecture.md                 # Detailed architecture blueprint
|-- README.md                       # <<< This file
|-- skills/
|   |-- data_skill.py               # Data ingestion (vnstock, yfinance, local file)
|   |-- quant_skill.py              # WPE, MFI, Shannon, SampEn, Cross-Sectional
|   |-- ds_skill.py                 # Dual GMM: Price + Volume regime classifiers
```

---

## References

- Bandt, C. & Pompe, B. (2002). *Permutation Entropy: A Natural Complexity Measure for Time Series.* Physical Review Letters, 88(17).
- Fadlallah, B. et al. (2013). *Weighted-permutation entropy: A complexity measure for time series incorporating amplitude information.* Physical Review E, 87(2).
- Richman, J. S. & Moorman, J. R. (2000). *Physiological time-series analysis using approximate entropy and sample entropy.* American Journal of Physiology, 278(6).
- Lopez-Ruiz, R., Mancini, H. L. & Calbet, X. (1995). *A statistical measure of complexity.* Physics Letters A, 209(5-6).
- Rosso, O. A. et al. (2007). *Distinguishing noise from chaos.* Physical Review Letters, 99(15).
