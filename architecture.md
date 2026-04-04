# Financial Entropy Agent -- Dual-Plane Architecture Blueprint

> **Objective**: The Multi-Modal Systemic Risk Engine observes the market through **two independent Unsupervised Learning planes** -- Price Dynamics and Liquidity Structure. The Agent Orchestrator (Anthropic Claude) acts as a **Cross-Plane Reasoning Engine**, confirming Price Physics with Liquidity Structure via the **ReAct** loop and **Tool Use** protocol.

---

## 1. Architecture Overview -- Dual-Plane Engine

```text
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

### Dual-Plane Definitions

| Plane | X-Axis | Y-Axis | Measurement | Purpose |
|---|---|---|---|---|
| **Plane 1: Price Dynamics** | Weighted Permutation Entropy (WPE) | Annualized Volatility | "Physical Chaos" + Kinematic Vectors (V, a) | Measures the degree of structural disorder in price dynamics. WPE evaluates the randomness of ordinal patterns; Volatility quantifies amplitude fluctuations. V = dE/dt (direction), a = d²E/dt² (momentum force). |
| **Plane 2: Liquidity Structure (Macro-Micro Fusion)** | Volume Shannon Entropy | Volume Sample Entropy | Macro-Micro Liquidity Status | Measures the structural integrity of capital flow using a dual-path architecture. The Macro component utilizes a Global Z-Score to evaluate absolute systemic liquidity. The Micro component applies a 252-day Rolling Z-Score before entropy computation to isolate localized structural regimes (Consensus vs. Erratic) while neutralizing structural break biases. |

### Agent Role: Cross-Plane Reasoning Engine

The Agent does not simply analyze each plane in isolation; it performs a **cross-plane synthesis** to detect systemic divergences that are invisible from a single dimensional perspective:

| Price Plane | Volume Plane (Micro) | Macro Condition | Conclusion | Rationale |
|---|---|---|---|---|
| Fragile/Chaos | Consensus Flow | Any | **STRUCTURAL ACCUMULATION** | Price chaos is contained by highly organized liquidity, indicating institutional absorption (Smart Money). |
| Fragile/Chaos | Erratic/Noisy Flow | Moderate/Low Z | **CRITICAL BREAKDOWN** | Price chaos is amplified by fragmented liquidity. Elevated systemic risk with no structural support. |
| Fragile/Chaos (V>0, a>0) | Erratic/Noisy Flow | Extreme High Z (>2.0) | **CLIMAX DISTRIBUTION** | Massive capital inflow combined with critical price fragility and highly fragmented behavior. A classic bubble burst or blow-off top signature. |
| Stable Growth | Erratic/Noisy Flow | Any | **TREND EXHAUSTION** | Stable price trends mask an underlying localized breakdown in liquidity structure. |
| *Other* | *Other* | Any | **SYSTEM COHERENT** | Both planes remain structurally synchronized. No cross-plane divergence detected. |

---

## 2. Design Principles

| Principle | Description |
|---|---|
| **Separation of Concerns** | Each skill is exclusively responsible for a single domain (Data / Math / ML). |
| **DRY** | Strict adherence to DRY; functions are defined in the canonical skill module and imported elsewhere. |
| **Vectorized First** | Array operations must utilize vectorized `numpy` implementations. Loops are strictly limited and require `@numba.jit(nopython=True)`. |
| **Type Safety** | Comprehensive type hints are enforced for all function signatures. |
| **Testable** | Every module concludes with an `if __name__ == "__main__":` block containing verification routines. |
| **Dual-Plane Independence** | The analytical classifiers (Price GMM, Volume GMM) execute with strict independence. The Agent remains the exclusive juncture for cross-plane synthesis. |

---

## 3. Module Details

### 3.1 `skills/data_skill.py` -- Data Ingestion Layer

**Responsibility**: Real-time market data acquisition, failover handling, and output normalization.

| Function | Input | Output | Notes |
|---|---|---|---|
| `fetch_vnindex(ticker, start, end)` | `ticker: str`, `start: str`, `end: str` | DataFrame OHLCV + DatetimeIndex | `vnstock` API. Columns: `Open, High, Low, Close, Volume`. |
| `fetch_vn30_returns(start, end)` | `start: str`, `end: str` | DataFrame pct_change returns VN30 | `yfinance`. `ffill()` applied prior to `pct_change()`. |
| `load_local_file(path)` | `path: str` | DataFrame OHLCV | Fallback mechanism. Supports `.csv` and `.xlsx`. |
| `get_latest_market_data(...)` | params | DataFrame OHLCV | Convenience wrapper: API priority with local fallback. |

---

### 3.2 `skills/quant_skill.py` -- Quantitative Physics Engine

**Responsibility**: Computation of Symbolic Dynamics (WPE, Complexity, MFI), Volume Entropy (Shannon, SampEn) via Macro-Micro Fusion, and Cross-Sectional Entropy.

#### 3.2.1 Plane 1: Price Entropy

| Function | Input | Output | Notes |
|---|---|---|---|
| `_calc_wpe_complexity_jit(x, m, tau)` | `np.ndarray`, `int`, `int` | `(H, C)` | Numba JIT. WPE + Jensen-Shannon Complexity. |
| `calc_rolling_wpe(log_returns, m, tau, window)` | arrays + params | `(wpe_arr, c_arr)` | Rolling window evaluation. Numba JIT. |
| `calc_wpe_complexity(x, m, tau)` | `np.ndarray` | `(float, float)` | Public wrapper for single arrays. |
| `calc_mfi(wpe, complexity)` | `np.ndarray`, `np.ndarray` | `np.ndarray` | MFI = WPE * (1 - C). Fully vectorized. |

**Kinematic Vectors (Processed directly in the pipeline)**:

| Vector | Formula | Parameters | Interpretation |
|---|---|---|---|
| `PE_Velocity` (V) | `df['WPE'].diff(3)` | 3-day momentum | Direction of entropy change. V > 0: chaos expanding. V < 0: structural order forming. |
| `PE_Acceleration` (a) | `PE_Velocity.diff(3)` | 3-day momentum | Momentum force. a > 0: trend is accelerating. a < 0: momentum is exhausting. |

**Handling NaNs**: Computed as `fillna(0)` to maintain JSON serialization and prevent GMM clustering destabilization.

#### 3.2.2 Plane 2: Volume Entropy (Macro-Micro Fusion)

| Function | Input | Output | Notes |
|---|---|---|---|
| `_calc_sample_entropy_jit(x, m, r)` | `np.ndarray`, `int`, `float` | `float` | Numba JIT. O(N²). SampEn = -ln(A/B). |
| `calc_sample_entropy(x, m, r)` | `np.ndarray` | `float` | Public wrapper. Adaptively configures r if unspecified. |
| `calc_shannon_entropy_hist(x, bins)` | `np.ndarray`, `str|int` | `float [0,1]` | Histogram Shannon Entropy. `bins='auto'` utilizes Freedman-Diaconis rule. |
| `calc_rolling_volume_entropy(volume, window, z_window)` | `np.ndarray`, `int=60`, `int=252` | `(shannon, sampen, global_z, rolling_z)` | Rolling wrapper implementing the dual-path Macro-Micro Fusion architecture. |

**Parameter Justification**:
- `bins='auto'`: Logarithmic volume distributions inherently possess heavy tails. Static bin constraints create empty subsets. The Freedman-Diaconis rule dynamically calibrates subset dispersion iteratively.
- `window=60`: SampEn with $m=2$ ideally requires $N \geq 10^m = 100$. A 60-day window is the minimal viable baseline balancing statistical convergence with practical regime responsiveness.
- **Macro-Micro Architecture**: Follows `log1p(volume)` transformation. Path A utilizes a Global Z-Score for absolute macro-scale systemic assessment. Path B computes a 252-day Rolling Z-score to eliminate structural break contamination. Entropy components (Shannon, SampEn) execute exclusively on Path B. SampEn tolerance $r$ revolves strictly at $0.2 \sigma$ since the rolling z-score vector already maintains unit variance.

#### 3.2.3 Cross-Sectional Correlation Entropy (VN30)

| Function | Input | Output | Notes |
|---|---|---|---|
| `calc_correlation_entropy(df_returns, window)` | `pd.DataFrame`, `int` | `pd.Series` (0-100) | Executed via Eigenvalue Decomposition over the Pearson Correlation Matrix. |

---

### 3.3 `skills/ds_skill.py` -- Data Science / ML Layer (Dual GMM)

**Responsibility**: Unsupervised Regime Classification across **both dimensions entirely**.

#### 3.3.1 Price Regime (Plane 1)

| Component | Description |
|---|---|
| `REGIME_NAMES` | `{0: "Stable Growth", 1: "Fragile Growth", 2: "Chaos/Panic"}` |
| `RegimeClassifier` | GMM `n_components=3`, `covariance_type='full'`. Mapping rule: Sorting executed sequentially by mean algorithmic feature value. |
| `fit_predict_regime(features)` | Functional API resolving -> `(labels, classifier)` |

#### 3.3.2 Volume Regime (Plane 2)

| Component | Description |
|---|---|
| `VOLUME_REGIME_NAMES` | `{0: "Consensus Flow", 1: "Dispersed Flow", 2: "Erratic/Noisy Flow"}` |
| `VolumeRegimeClassifier` | GMM `n_components=3`, `covariance_type='full'`. Mapping rule: Sorted recursively against sum(Shannon + SampEn). Lowest combined entropy translates to Consensus, while highest indicates Erratic variance. |
| `fit_predict_volume_regime(features)` | Functional API resolving -> `(labels, classifier)` |

**Volume Regime Semantics**:
- **Consensus Flow**: Low Shannon + Low SampEn. Concentration is high, and impulses are highly regular. Indicative of institutional / Smart Money accumulation or distribution dynamics.
- **Dispersed Flow**: High Shannon + Moderate SampEn. Transitional phase characterized by increasing participation divergence.
- **Erratic/Noisy Flow**: High Shannon + High SampEn. Systemic volumetric variance coupled with irregularity. Defines retail fragmentation or catastrophic panic events.

---

### 3.4 `agent_orchestrator.py` -- Cross-Plane Reasoning Engine

> **Disclaimer (Human-in-the-Loop):** The Agent is engineered exclusively as a structural telemetry diagnostic tool. It maps entropy states and kinematics vectors; however, the agent resolves its analytical synthesis autonomously. Any action based upon these analytical diagnostics naturally requires secondary human vetting matching current risk horizons.

**Responsibility**: Core Dual-Plane execution matrix. Manages iterative data sequencing through all 5 primary tools, synthesizes kinematic vectors against GMM regime classifications, and produces the unified Cross-Plane analytical report incorporating Macro-Micro Fusion heuristics.

#### Tool Execution Framework

```text
[1] fetch_market_data      -> Extracts comprehensive OHLCV inputs
[2] compute_entropy_metrics -> Calculates Plane 1 Physics (WPE, C, MFI, Volatility, V, a)
[3] compute_volume_entropy  -> Calculates Plane 2 (Shannon, SampEn, Global Z, Rolling Z)
[4] predict_market_regime   -> Instantiates Plane 1 GMM classifier output
[5] predict_volume_regime   -> Instantiates Plane 2 GMM classifier output
[6] Cross-Plane Synthesis   -> Executes unified diagnostic logic integrating Macro Z & Micro Entropy
```

#### Cross-Plane Synthesis Protocol (Source Implementation)

```python
def _cross_plane_synthesis(price_regime: str, volume_regime: str) -> tuple[str, str]:
    # Logical heuristics expanding upon the matrix defined above
    p = price_regime.upper()
    v = volume_regime.upper()
    
    # ... Validation for chaos, stability, and consensus flags ...
    
    if p_fragile_chaos and v_consensus:
        return "STRUCTURAL ACCUMULATION"
    elif p_fragile_chaos and v_erratic:
        return "CRITICAL BREAKDOWN" # Agent extends this evaluating Climax Distribution logic
    elif p_stable and v_erratic:
        return "TREND EXHAUSTION"
    else:
        return "SYSTEM COHERENT"
```

#### Standardized Diagnostic Output Blueprint

```text
==================================================
  DUAL-PLANE DIAGNOSTIC REPORT (MACRO-MICRO FUSION)
==================================================

  PLANE 1 -- PRICE DYNAMICS
  REGIME          : [FRAGILE GROWTH]
  MFI             : 0.8612
  PE Velocity (V) : +0.0312 (chaos expanding)
  PE Accel (a)    : -0.0045 (momentum fading)

  PLANE 2 -- LIQUIDITY STRUCTURE (MACRO-MICRO FUSION)
  MICRO REGIME    : [CONSENSUS FLOW]
  Macro Z (Global): -0.45
  Vol Shannon     : 0.8234
  Vol SampEn      : 1.4521

  CROSS-PLANE SYNTHESIS
  CONCLUSION      : [STRUCTURAL ACCUMULATION]
  SYSTEMIC RISK   : [MODERATE]

  [Agent contextual rationale output follows...]
==================================================
```

---

## 4. Operational Data Flow Pipeline

```text
[1] Query Iteration: "Analyze VNINDEX with Cross-Plane synthesis. Is it structurally sound?"
                 |
[2] Orchestrator Evaluation: Requires sequential dispatch of data retrieval -> physics generation -> dual GMM analysis
                 |
[3] ACT: fetch_market_data(ticker="VNINDEX", start="2024-01-01")
                 |
[4] ACT: compute_entropy_metrics()    -- Solves Plane 1 matrices
    ACT: compute_volume_entropy()     -- Solves Plane 2 Macro-Micro constraints
                 |
[5] ACT: predict_market_regime()      -- Extracts Plane 1 specific labels
    ACT: predict_volume_regime()      -- Extracts Plane 2 specific labels
                 |
[6] LOGICAL SYNTHESIS: Macro-Micro Synthesis & Cross-Plane Matrix
    e.g., Price ("Fragile Growth", V>0, a>0) + Volume ("Erratic Flow" @ Macro Z = +2.4)
    -> Evaluated State: CLIMAX DISTRIBUTION 
                 |
[7] RESPONSIVE DISCONNECT: 
    "The Dual-Plane Engine confirms CLIMAX DISTRIBUTION. While Plane 1 detects accelerating physical 
    price chaos (V>0, a>0), Plane 2's Macro-Micro Fusion critically indicates that extreme systemic 
    liquidity (Macro Z: +2.40) is currently operating under a fragmented micro-structural regime 
    (Erratic/Noisy Flow). This divergence highlights a volatile distribution signature rather than 
    accumulation. Systemic risk evaluates to CRITICAL."
```

---

## 5. Repository Structure Architecture

```text
Financial Entropy Agent/
|-- agent_orchestrator.py       # Cross-Plane Reasoning Engine (React + Macro-Micro heuristics)
|-- dashboard.py                # Streamlit UI: Realtime Dual Scatter charting & Agent output
|-- architecture.md             # <<< Canonical Architectural Guide (this file)
|-- README.md                   # Macro-theory and end-user documentation
|-- skills/
|   |-- data_skill.py           # Ingestion layer (vnstock, yfinance, local normalization)
|   |-- quant_skill.py          # Arithmetic & Physics: WPE, MFI, Shannon, SampEn, EVD
|   |-- ds_skill.py             # Dual GMM instances: RegimeClassifier + VolumeRegimeClassifier
|-- _reference_VSE/             # Legacy references (Streamlit monolithic codebase)
|-- .agents/
|   |-- workflows/              # Defined system behavior pathways
```

---

## 6. System Dependency Topography

| Package | Version Baseline | Target Module | Functionality Baseline |
|---|---|---|---|
| `numpy` | $\geq$ 1.24 | `quant_skill`, `ds_skill` | Array generation, vectorized math, and advanced linear algebra |
| `pandas` | $\geq$ 2.0 | `data_skill`, `quant_skill`, `dashboard` | DataFrame standardization and index integrity |
| `numba` | $\geq$ 0.58 | `quant_skill` | JIT acceleration (Mandatory for WPE rolling & $O(N^2)$ SampEn computation) |
| `vnstock` | $\geq$ 2.0 | `data_skill` | Primary local entity for VN-Index OHLCV fetch mechanics |
| `yfinance` | $\geq$ 0.2 | `data_skill` | Secondary resolution protocol (and VN30 component acquisition) |
| `scikit-learn` | $\geq$ 1.3 | `ds_skill` | GaussianMixture protocols and robust StandardScaling structures |
| `anthropic` | $\geq$ 0.30 | `agent_orchestrator` | Core AI reasoning protocols and Tool Use pipeline handling |
| `streamlit` | $\geq$ 1.30 | `dashboard` | Local execution framework |
| `plotly` | $\geq$ 5.18 | `dashboard` | Integrated high-fidelity graphing engines |

---

## 7. Core Developer Logistics & Code Guidelines

1. **NO RE-DEFINITIONS**: Functions maintaining quantitative value strictly remain in their canonical skill folders. Code reuse implies immediate import methodology (`from skills.quant_skill import ...`).
2. **DOMAIN INTEGRITY**: Arithmetic arrays operate inside `quant_skill.py`. ML algorithms remain inside `ds_skill.py`.
3. **VECTORIZATION PRIORITY**: Core system processing operates on `numpy` arrays. Standard algorithms must iterate efficiently without traditional loop protocols unless explicitly cached by `@numba.jit(nopython=True)`.
4. **TYPE ADHERENCE**: Python strict type hinting validates standard function boundaries. Associated docstrings strictly cover inputs, outputs, and fundamental mathematical context.
5. **VERIFIABILITY**: System execution necessitates that each module functions robustly standalone via an inner `if __name__ == "__main__":` validation loop.
6. **PLANE SYMMETRY**: Regimes inherently track independently. The Agent mechanism exists purely to cross-synthesize states, it does not instruct algorithmic models.
