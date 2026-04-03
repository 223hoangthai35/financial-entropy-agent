# Financial Entropy Agent: System Architect

## 1. Project Overview & Motivation
**Financial Entropy Agent** is a high-end quantitative finance terminal designed to monitor systemic risk and structural complexity in financial markets through the lens of Chaos Theory and Symbolic Dynamics. 

Traditional technical analysis heavily relies on lagging price indicators (like Moving Averages, RSI, or MACD), which only react after a structural breakdown has occurred. Furthermore, traditional probability-based entropy fails to capture the temporal order of market data, making it sub-optimal for non-stationary financial markets. The Financial Entropy Agent proposes a paradigm shift: adopting **Ordinal Pattern Analysis** to decouple "Market Noise" from intrinsic "Regimes." By measuring the thermodynamic entropy of capital dispersion and ordinal permutations, this system flags fragile market conditions before they manifest as catastrophic price collapses.

## 2. Technical Architecture: 1. ALL-IN-ONE STRUCTURAL TELEMETRY

### Permutation Entropy (PE) & Ordinal Analysis for VNINDEX
Permutation Entropy (PE) is employed as our primary diagnostic tool because it accurately measures the structural orderliness and deterministic characteristics of a time series based on its ordinal sequence. Because it maps values to ordinal rankings rather than raw amplitudes, PE captures the **temporal order** of market data, making it fundamentally superior to probability-based structural entropy for non-stationary financial markets.

When a market trends sustainably, its price patterns become highly structured, driving Entropy toward zero. Conversely, when capital fragments and "smart money" exits—leaving only random, uncoordinated retail noise—Entropy spikes toward the state of maximum chaos.

**The Formula:**
We calculate the Weighted Permutation Entropy ($H_{WPE}$) to capture both structural sequential patterns and amplitude variations:

$$
H_{WPE} = - \frac{1}{\ln(d!)} \sum_{i=1}^{d!} p^{(w)}_i \ln(p^{(w)}_i)
$$

Where:
- $d$ represents the **embedding dimension** (e.g., $d=3$), determining the length of the ordinal patterns extracted.
- $\tau$ is the **time delay** between data points (standard $\tau=1$ for daily granularity).
- $p^{(w)}_i$ is the variance-amplified frequency of the $i$-th ordinal pattern.

### Cross-Sectional Entropy for VN30
While VNINDEX captures the global state, the VN30 basket provides insight into the underlying structural pillars (blue-chip components). We construct a Pearson Correlation matrix of the largest 30 stocks and calculate structural fragmentation through Eigenvalue Decomposition (EVD).

The Cross-Sectional Entropy measures whether capital is flowing in a centralized consensus (Low Entropy) or dispersing erratically across sectors (High Entropy). A breakdown in Cross-Sectional consensus acts as the earliest warning of systemic fragility.

## 3. Machine Learning Logic: 2. UNSUPERVISED LEARNING DS PROOF

### Market Volatility (Ann. Std)
- **Definition:** The Annualized Standard Deviation of Log-Returns.
- **Formula:** 
  $$ \sigma_{ann} = \sigma_{daily} \times \sqrt{252} $$
- **Significance:** Measures the magnitude of price action and systemic turbulence. 

### GMM Clustering (The DS Core)
To move beyond subjective analysis, we apply a **Gaussian Mixture Model (GMM)**. The algorithm maps the historical trading sessions onto a 2D plane based on Entropy and Volatility parameters. Completely untethered from human labeling, this unsupervised learning approach probabilistically clusters the market space into mathematically distinct states:
1. **Stable Growth (Green):** Low Entropy, clustered volatility, structural deterministic consensus.
2. **Fragile Growth (Yellow):** Anomalous divergence—nominal surges in price accompanied by an ordinal pattern breakdown.
3. **Chaos/Panic (Red):** Maximum complexity saturation, high dynamic complexity, and structural fragmentation.

## 4. The Brain: 3. 🤖 AGENT ORCHESTRATOR DIAGNOSTIC

### ReAct Loop Integration
"Financial Entropy Agent" features a built-in **AI Orchestrator** leveraging Anthropic's Tool Use protocol. Instead of statically displaying the results, the Orchestrator, operating as a Financial Entropy Expert, runs an autonomous ReAct (Reasoning and Acting) loop. It fetches the required market vectors, calculates the thermodynamic physics parameters, triggers the ML clustering, and compiles the final diagnostic report.

### Diagnostic Logic (Explainable AI - XAI)
The Agent functions as an Explainable AI layer overlaying the Physics Engine. It doesn't merely spit out a classification label. By dissecting the variance between global entropy (VNINDEX) and structural component fragmentation (VN30), the Agent formulates a sophisticated, human-readable justification for the current Regime state—specifically highlighting structural vulnerabilities, ordinal pattern breakdowns, and dynamic complexity shifts.

## 5. Resilience & Portability
Engineered for ultimate institutional resilience:
- **Dual-Pipeline Routing:** The terminal seamlessly integrates cloud-based API fetches (via `vnstock` and `yfinance`). 
- **Decentralized Manual Failover:** Should API routes fail due to geographic constraints or Cloud IP bans, the terminal accepts local CSV / Excel files via an Upload Data Pipeline. The physics engines automatically recalibrate to the manual data feed, ensuring uninterrupted analysis capabilities.

## 6. How to Run

**Prerequisites:** 
- Python 3.9+
- An API Key from Anthropic (Optional, for full Agent capabilities)

**Installation:**
```bash
pip install -r requirements.txt
```
*(Dependencies include: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `numba`, `vnstock`, `yfinance`, `anthropic`)*

**Execution:**
Start the terminal from the base directory using:
```bash
streamlit run dashboard.py
```
