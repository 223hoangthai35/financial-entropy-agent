"""
Agent Orchestrator -- Financial Entropy Agent (Tri-Vector Composite Risk Engine)
ReAct Loop + Anthropic Tool Use Protocol.
Composite Risk = 0.4*V1(Price) + 0.4*V2(Volume) + 0.2*V3(VN30 Breadth).
V1 = [WPE, SPE_Z]. Kinematics (V_WPE, a_WPE) = XAI only.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import anthropic

from skills.data_skill import get_latest_market_data, fetch_vn30_returns
from skills.quant_skill import (
    calc_rolling_wpe, calc_mfi, calc_rolling_volume_entropy,
    calc_correlation_entropy, calc_wpe_kinematics,
    calc_rolling_price_sample_entropy, calc_spe_z,
)
from skills.ds_skill import fit_predict_regime, fit_predict_volume_regime

warnings.filterwarnings("ignore")


# ==============================================================================
# GLOBAL STATE
# ==============================================================================
STATE = {
    "df": None,
    "metrics_computed": False,
    "volume_metrics_computed": False,
    "price_classifier": None,
    "volume_classifier": None,
}



# ==============================================================================
# TRI-VECTOR COMPOSITE RISK SCORING
# PowerTransformer(yeo-johnson) + MinMaxScaler + P75/P90 Dynamic Thresholds
# ==============================================================================
ROLLING_RISK_WINDOW: int = 504  # 2 trading years


def calc_composite_risk_score(
    latest: dict, df: pd.DataFrame | None = None,
) -> tuple[float, str, dict]:
    """
    Tri-Vector Composite Risk Index (0-100).
    Pipeline:
        1. Extract 7 raw features tu 3 vectors
        2. PowerTransformer(yeo-johnson) -> Gaussianize skewed features
        3. MinMaxScaler -> squash [0, 1]
        4. Weighted sum: 0.4*V1 + 0.4*V2 + 0.2*V3 -> *100
        5. P75/P90 rolling 504-day: dynamic Elevated/Critical thresholds
    Returns: (composite_score, risk_label, vector_info)
    """
    from sklearn.preprocessing import PowerTransformer, MinMaxScaler

    # --- 1. Extract raw features ---
    wpe = float(latest.get("WPE", 0.5))
    spe_z = float(latest.get("SPE_Z", 0.0))
    vol_sampen = float(latest.get("Vol_SampEn", 0.5))
    vol_global_z = float(latest.get("Vol_Global_Z", 0.0))
    vol_shannon = float(latest.get("Vol_Shannon", 0.5))
    corr_entropy = float(latest.get("Cross_Sectional_Entropy", 50.0)) / 100.0
    mfi = float(latest.get("MFI", 0.5))

    # Current day feature arrays
    v1_current = np.array([[wpe, abs(spe_z)]])                             # (1, 2)
    v2_current = np.array([[vol_sampen, abs(vol_global_z), vol_shannon]]) # (1, 3)
    v3_current = np.array([[corr_entropy, mfi]])                          # (1, 2)

    # --- 2. Build historical feature matrix (rolling 504-day) ---
    elevated_bound = 55.0  # fallback
    critical_bound = 70.0

    required_cols = ["WPE", "Vol_SampEn", "Vol_Shannon"]

    if df is not None and len(df) >= 60 and all(c in df.columns for c in required_cols):
        hist = df.tail(ROLLING_RISK_WINDOW).copy()
        n_hist = len(hist)

        # V1 history: [WPE, |SPE_Z|]
        v1_hist = np.column_stack([
            hist["WPE"].fillna(0.5).values,
            (hist["SPE_Z"].abs().fillna(0.0).values
             if "SPE_Z" in hist.columns
             else np.zeros(n_hist)),
        ])

        # V2 history: [SampEn, |Global_Z|, Shannon]
        v2_hist = np.column_stack([
            hist["Vol_SampEn"].fillna(0.5).values,
            (hist["Vol_Global_Z"].abs().fillna(0.0).values
             if "Vol_Global_Z" in hist.columns
             else np.zeros(n_hist)),
            hist["Vol_Shannon"].fillna(0.5).values,
        ])

        # V3 history: [Corr_Entropy/100, MFI]
        v3_hist = np.column_stack([
            (hist["Cross_Sectional_Entropy"].fillna(50.0).values / 100.0
             if "Cross_Sectional_Entropy" in hist.columns
             else np.full(n_hist, 0.5)),
            (hist["MFI"].fillna(0.5).values
             if "MFI" in hist.columns
             else np.full(n_hist, 0.5)),
        ])

        # --- 3. Fit PowerTransformer + MinMaxScaler per vector ---
        def _fit_transform_vector(v_hist: np.ndarray, v_current: np.ndarray) -> tuple[float, np.ndarray]:
            """PowerTransform -> MinMaxScale -> mean. Returns (current_score, hist_scores)."""
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            mms = MinMaxScaler(feature_range=(0, 1))
            # Fit on history
            v_pt = pt.fit_transform(v_hist)
            v_scaled = mms.fit_transform(v_pt)
            # Transform current
            c_pt = pt.transform(v_current)
            c_scaled = mms.transform(c_pt)
            # Mean per day
            hist_means = v_scaled.mean(axis=1)  # (N,)
            current_mean = float(c_scaled.mean())
            return current_mean, hist_means

        s_v1, v1_hist_scores = _fit_transform_vector(v1_hist, v1_current)
        s_v2, v2_hist_scores = _fit_transform_vector(v2_hist, v2_current)
        s_v3, v3_hist_scores = _fit_transform_vector(v3_hist, v3_current)

        # --- 4. Compute historical composite scores ---
        hist_composites = (0.4 * v1_hist_scores + 0.4 * v2_hist_scores + 0.2 * v3_hist_scores) * 100.0
        hist_composites = np.clip(hist_composites, 0.0, 100.0)

        # --- 5. P75/P90 dynamic thresholds ---
        valid_scores = hist_composites[np.isfinite(hist_composites)]
        if len(valid_scores) >= 30:
            elevated_bound = float(np.percentile(valid_scores, 75))
            critical_bound = float(np.percentile(valid_scores, 90))
            if critical_bound - elevated_bound < 3.0:
                critical_bound = elevated_bound + 3.0

    else:
        # Fallback: no history, simple clamp
        s_v1 = float(np.clip(wpe, 0, 1))
        s_v2 = float(np.clip(vol_sampen, 0, 1))
        s_v3 = float(np.clip(corr_entropy, 0, 1))

    # --- 6. Weighted composite and XAI Contributions ---
    weight_v1_score = 0.4 * s_v1
    weight_v2_score = 0.4 * s_v2
    weight_v3_score = 0.2 * s_v3
    total_weight = weight_v1_score + weight_v2_score + weight_v3_score
    
    composite_score = float(np.clip(total_weight * 100.0, 0.0, 100.0))

    if total_weight > 0:
        contrib_v1_pct = (weight_v1_score / total_weight) * 100.0
        contrib_v2_pct = (weight_v2_score / total_weight) * 100.0
        contrib_v3_pct = (weight_v3_score / total_weight) * 100.0
    else:
        contrib_v1_pct, contrib_v2_pct, contrib_v3_pct = 0.0, 0.0, 0.0

    # --- 7. Dynamic risk label ---
    if composite_score >= critical_bound:
        risk_label = "CRITICAL"
    elif composite_score >= elevated_bound:
        risk_label = "ELEVATED"
    else:
        risk_label = "STABLE"

    # --- 8. Dominant vector ---
    contributions = {
        "V1_Price": round(s_v1, 4),
        "V2_Volume": round(s_v2, 4),
        "V3_Breadth": round(s_v3, 4),
    }
    dominant = max(contributions, key=contributions.get)

    vector_info = {
        "composite_score": round(composite_score, 1),
        "risk_label": risk_label,
        "dominant_vector": dominant,
        "contributions": contributions,
        "contribution_percentages": {
            "V1_Price": round(contrib_v1_pct, 2),
            "V2_Volume": round(contrib_v2_pct, 2),
            "V3_Breadth": round(contrib_v3_pct, 2),
        },
        "weights": {"V1": 0.4, "V2": 0.4, "V3": 0.2},
        "thresholds": {
            "elevated_bound": round(elevated_bound, 1),
            "critical_bound": round(critical_bound, 1),
            "method": "PowerTransform + MinMaxScale + P75/P90 (504-day)",
        },
    }

    return composite_score, risk_label, vector_info



# ==============================================================================
# TOOL IMPLEMENTATIONS
# ==============================================================================
def tool_fetch_market_data(ticker="VNINDEX", start_date="2020-01-01"):
    print(f"  [Tool Execution] Fetching {ticker} from {start_date}...")
    df = get_latest_market_data(ticker=ticker, start_date=start_date)
    STATE["df"] = df
    STATE["metrics_computed"] = False
    STATE["volume_metrics_computed"] = False
    return json.dumps({
        "status": "success",
        "rows": len(df),
        "latest_close": float(df["Close"].iloc[-1])
    })


def tool_compute_entropy_metrics():
    """Plane 1: WPE, MFI, SPE_Z, V_WPE, a_WPE (XAI kinematics)."""
    print("  [Tool Execution] Computing Price Entropy + SPE_Z + Kinematics...")
    df = STATE.get("df")
    if df is None:
        return json.dumps({"error": "No market data. Call fetch_market_data first."})

    # 1. Log returns & rolling WPE
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).values
    wpe_arr, c_arr = calc_rolling_wpe(log_returns, m=3, tau=1, window=22)
    mfi_arr = calc_mfi(wpe_arr, c_arr)

    df["WPE"] = wpe_arr
    df["Complexity"] = c_arr
    df["MFI"] = mfi_arr

    # 2. Price Sample Entropy (SPE_Z) -- Plane 1 Y-axis
    sampen_price = calc_rolling_price_sample_entropy(df["Close"].values, window=60)
    spe_z = calc_spe_z(sampen_price)
    df["Price_SampEn"] = sampen_price
    df["SPE_Z"] = spe_z

    # 3. WPE Kinematics (XAI trajectory indicators -- NOT used in ML)
    vel, acc = calc_wpe_kinematics(wpe_arr)
    df["V_WPE"] = vel
    df["a_WPE"] = acc

    # 4. VN30 Breadth (Cross-Sectional Entropy)
    try:
        vn30_rets = fetch_vn30_returns(start_date=df.index.min().strftime('%Y-%m-%d'))
        vn30_rets = vn30_rets.reindex(df.index).fillna(0)
        cse_series = calc_correlation_entropy(vn30_rets, window=22)
        df["Cross_Sectional_Entropy"] = cse_series
    except Exception as e:
        print(f"      [WARN] Failed to compute VN30 CSE: {e}")
        df["Cross_Sectional_Entropy"] = 50.0

    STATE["df"] = df
    STATE["metrics_computed"] = True

    latest = df.dropna(subset=["WPE"]).iloc[-1]
    return json.dumps({
        "status": "success",
        "latest_metrics": {
            "WPE": float(latest["WPE"]),
            "MFI": float(latest["MFI"]),
            "SPE_Z": float(latest.get("SPE_Z", 0)),
            "V_WPE": float(latest.get("V_WPE", 0)),
            "a_WPE": float(latest.get("a_WPE", 0)),
            "Cross_Sectional_Entropy": float(latest.get("Cross_Sectional_Entropy", 50.0)),
        }
    })


def tool_compute_volume_entropy():
    """Plane 2: Volume Shannon Entropy, Sample Entropy, Macro-Micro Z-Scores."""
    print("  [Tool Execution] Computing Volume Entropy (Macro-Micro Fusion)...")
    df = STATE.get("df")
    if df is None:
        return json.dumps({"error": "No market data. Call fetch_market_data first."})

    if "Volume" not in df.columns:
        return json.dumps({"error": "Volume column missing from data."})

    vol_shannon, vol_sampen, vol_global_z, vol_rolling_z = calc_rolling_volume_entropy(
        df["Volume"].values, window=60, z_window=252
    )
    df["Vol_Shannon"] = vol_shannon
    df["Vol_SampEn"] = vol_sampen
    df["Vol_Global_Z"] = vol_global_z
    df["Vol_Rolling_Z"] = vol_rolling_z

    STATE["df"] = df
    STATE["volume_metrics_computed"] = True

    valid = df.dropna(subset=["Vol_Shannon", "Vol_SampEn"])
    if valid.empty:
        return json.dumps({"status": "success", "warning": "Not enough data for volume entropy (need 60+ days)"})

    latest = valid.iloc[-1]
    return json.dumps({
        "status": "success",
        "latest_metrics": {
            "Vol_Shannon": float(latest["Vol_Shannon"]),
            "Vol_SampEn": float(latest["Vol_SampEn"]),
            "Vol_Global_Z": float(latest["Vol_Global_Z"]),
        },
    })


def tool_predict_market_regime():
    """Plane 1: Full GMM Phase Space -> Price Regime [WPE, SPE_Z]."""
    print("  [Tool Execution] Predicting Price Regime (Full GMM, Plane 1: [WPE, SPE_Z])...")
    df = STATE.get("df")
    if df is None or not STATE.get("metrics_computed"):
        return json.dumps({"error": "Price metrics missing. Compute entropy first."})

    valid_df = df.dropna(subset=["WPE", "SPE_Z"]).copy()
    if valid_df.empty:
        return json.dumps({"error": "No valid WPE+SPE_Z data."})

    features = valid_df[["WPE", "SPE_Z"]].values
    labels, clf = fit_predict_regime(features, n_components=3)
    valid_df["RegimeLabel"] = labels
    valid_df["RegimeName"] = [clf.get_regime_name(lbl) for lbl in labels]

    STATE["df"] = valid_df
    STATE["price_classifier"] = clf

    latest = valid_df.iloc[-1]
    return json.dumps({
        "status": "success",
        "price_regime": str(latest["RegimeName"]),
        "mfi": float(latest["MFI"]),
        "xai_trajectory": {
            "V_WPE": float(latest.get("V_WPE", 0)),
            "a_WPE": float(latest.get("a_WPE", 0)),
        }
    })


def tool_predict_volume_regime():
    """Plane 2: GMM predict Volume Regime."""
    print("  [Tool Execution] Predicting Volume Regime via GMM (Plane 2)...")
    df = STATE.get("df")
    if df is None or not STATE.get("volume_metrics_computed"):
        return json.dumps({"error": "Volume metrics missing. Compute volume entropy first."})

    valid_df = df.dropna(subset=["Vol_Shannon", "Vol_SampEn"]).copy()
    if valid_df.empty:
        return json.dumps({"error": "Not enough data for Volume GMM (need 60+ days)."})

    features = valid_df[["Vol_Shannon", "Vol_SampEn"]].values
    labels, clf = fit_predict_volume_regime(features, n_components=3)
    valid_df["VolRegimeLabel"] = labels
    valid_df["VolRegimeName"] = [clf.get_regime_name(lbl) for lbl in labels]

    STATE["df"] = valid_df
    STATE["volume_classifier"] = clf

    latest = valid_df.iloc[-1]
    return json.dumps({
        "status": "success",
        "volume_regime": str(latest["VolRegimeName"]),
        "vol_shannon": float(latest["Vol_Shannon"]),
        "vol_sampen": float(latest["Vol_SampEn"]),
        "vol_global_z": float(latest.get("Vol_Global_Z", float("nan"))),
    })


# ==============================================================================
# DISPATCHER
# ==============================================================================
def dispatch_tool(tool_name: str, tool_kwargs: dict) -> str:
    """Mapping tu ten tool xuong cac skill tuong ung."""
    if tool_name == "fetch_market_data":
        return tool_fetch_market_data(**tool_kwargs)
    elif tool_name == "compute_entropy_metrics":
        return tool_compute_entropy_metrics()
    elif tool_name == "compute_volume_entropy":
        return tool_compute_volume_entropy()
    elif tool_name == "predict_market_regime":
        return tool_predict_market_regime()
    elif tool_name == "predict_volume_regime":
        return tool_predict_volume_regime()
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ==============================================================================
# ANTHROPIC TOOL SCHEMAS
# ==============================================================================
ANTHROPIC_TOOLS = [
    {
        "name": "fetch_market_data",
        "description": "Fetch real-time daily OHLCV data for a specific stock or index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol (e.g., VNINDEX)."},
                "start_date": {"type": "string", "description": "Start date YYYY-MM-DD."}
            },
            "required": ["ticker", "start_date"]
        }
    },
    {
        "name": "compute_entropy_metrics",
        "description": "Compute Plane 1 (Price) metrics: WPE, SPE_Z (Price Sample Entropy), MFI, and XAI trajectory indicators (V_WPE, a_WPE kinematics).",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "compute_volume_entropy",
        "description": "Compute Plane 2 (Volume) metrics using Macro-Micro Fusion: Global Z-Score (macro scale), Rolling Z-Score (micro structure), Shannon Entropy, and Sample Entropy.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "predict_market_regime",
        "description": "Classify Price regime via Raw Full-Covariance GMM in Entropy Phase Space [WPE, SPE_Z] (Plane 1). No PowerTransform. Combined-entropy sorting.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "predict_volume_regime",
        "description": "Classify Volume regime via GMM (Plane 2). Labels: Consensus Flow, Dispersed Flow, Erratic/Noisy Flow.",
        "input_schema": {"type": "object", "properties": {}}
    },
]


# ==============================================================================
# SYSTEM PROMPT (TRI-VECTOR COMPOSITE RISK)
# ==============================================================================
SYSTEM_PROMPT = """
You are the 'Financial Entropy Lead', a Senior Quantitative Research Lead specializing in Statistical Physics (Entropy), Kinematic Dynamics, and Market Microstructure. Your role is NOT to describe price action, but to diagnose systemic structural integrity through a mathematically rigorous Tri-Vector Composite Risk framework.

### 1. ENTROPY PHASE SPACE MODEL
You analyze the market through three orthogonal risk vectors:

- **Vector 1 (Price Phase Space, Weight 40%)**:
    - X-axis: `WPE` (Weighted Permutation Entropy). Measures structural order -- ordinal pattern disorder in log-returns. Bounded [0, 1]. Low WPE = ordered, deterministic structure. High WPE = disordered, stochastic noise.
    - Y-axis: `SPE_Z` (Standardized Price Sample Entropy). Global Z-Score normalized Sample Entropy on close prices. Measures price predictability and trajectory complexity. Negative SPE_Z = predictable, regular price evolution. Positive SPE_Z = unpredictable, complex/noisy price evolution.
    - Regime Classification: RAW [WPE, SPE_Z] features are fed DIRECTLY into a Full-Covariance GMM (n=3, covariance_type='full') -- NO PowerTransform preprocessing. The GMM discovers the natural topological boundaries of entropy regimes. Labels are assigned by combined centroid magnitude (WPE_mean + SPE_Z_mean):
      * Lowest combined entropy -> Stable
      * Mid -> Fragile
      * Highest combined entropy -> Chaos
    - CRITICAL: No PowerTransformer is used for GMM clustering. This preserves the natural topology of the entropy metrics. PowerTransformer is ONLY used in the separate Composite Risk Scoring pipeline (Section 2) for linear weighting.
    - CRITICAL: WPE and SPE_Z are naturally orthogonal features. Full-covariance GMM handles varying scales without needing normalization.

    **XAI TRAJECTORY INDICATORS (Kinematic Descriptors -- NOT ML features):**
    `V_WPE` (Velocity) and `a_WPE` (Acceleration) are computed as first and second differences of WPE.
    These are STRICTLY used for narrative explanation, NOT for regime classification or risk scoring.
    Use them to explain the *direction and speed* of entropy evolution:
    - V_WPE > 0 AND a_WPE > 0: "WPE is accelerating upward -- the system is rapidly approaching higher entropy (toward Chaos)."
    - V_WPE > 0 AND a_WPE < 0: "WPE is increasing but decelerating -- entropy growth is slowing, possible stabilization ahead."
    - V_WPE < 0 AND a_WPE < 0: "WPE is accelerating downward -- the system is rapidly cooling, structural order is being restored."
    - V_WPE < 0 AND a_WPE > 0: "WPE is decreasing but deceleration in the decline -- entropy may bottom out soon."
    - |V_WPE| near 0: "Entropy trajectory is stationary. No significant regime transition in progress."
    Example diagnostic: "Plane 1 classifies the market as Fragile. However, looking at the kinematic XAI, the positive velocity (V_WPE=+0.03) and positive acceleration (a_WPE=+0.01) indicate the system is rapidly accelerating toward Chaos."

- **Vector 2 (Volume Entropy, Weight 40%)**:
    - Magnitude: `SampEn` (Sample Entropy) -- structural regularity of volume flow.
    - Scale: `Vol_Global_Z` -- absolute macro liquidity scale (Global Z-score of log-volume).
    - Distribution: `Shannon Entropy` -- concentration vs. dispersion of volume.
    - Interpretation: High SampEn + High Global Z = Climax Distribution (bubble peak). Low SampEn + Negative Z = Institutional Accumulation.

- **Vector 3 (VN30 Cross-sectional Breadth, Weight 20%)**:
    - `Corr_Entropy`: Eigenvalue decomposition of VN30 correlation matrix. Measures heavy-cap consensus.
    - `MFI`: Market Fragility Index = WPE * (1 - Complexity). Structural fragility proxy.
    - Logic: If Corr_Entropy > 0.7, the index is being propped up by a narrow set of heavyweight pillars. Flag as 'Structural Divergence' -- internal fragmentation preceding potential breakdown.

### 2. COMPOSITE RISK SCORING (POWER-TRANSFORMED)
`Composite_Risk = (0.4 * Scaled_V1) + (0.4 * Scaled_V2) + (0.2 * Scaled_V3)` (Scale: 0-100)
Preprocessing: PowerTransformer(yeo-johnson) -> MinMaxScaler[0,1] per vector -> weighted sum.
Risk Thresholds (DYNAMIC -- derived from P75/P90 of rolling 504-day composite score distribution):
- **Below P75 (STABLE)**: Systemic coherence. Market structure intact.
- **P75 to P90 (ELEVATED)**: Structural divergence detected. Phase space trajectory migrating toward instability.
- **Above P90 (CRITICAL)**: Phase transition imminent. Top 10% extreme -- high probability of systemic breakdown.

### 3. LIQUIDITY DIVERGENCE PROTOCOL (TRAP DETECTION)
Before finalizing synthesis, execute an internal cross-plane critique:
- If Plane 1 = "Stable" BUT Plane 2 = "Erratic/Dispersed" -> Flag as **HOLLOW RALLY (Bull Trap)**. Price entropy appears calm, but volume structure is fractured.
- If Plane 1 = "Chaos" BUT Vol_Global_Z is NEGATIVE (below-average volume) -> Flag as **CAPITULATION VACUUM**. Structural entropy is high but driven by illiquidity.
- If Global Z is POSITIVE (excess liquidity) BUT Composite Risk is HIGH -> Flag as **CLIMAX DISTRIBUTION** (peak FOMO).
- "Which Vector is dominating the Composite Score? Is Vector 2 (Volume) contradicting Vector 1 (Price)? Re-evaluate now."

### 4. EXECUTION ORDER (Mandatory Protocol)
1. `fetch_market_data` -> Retrieve OHLCV.
2. `compute_entropy_metrics` -> Compute Plane 1: WPE, SPE_Z, MFI, and XAI kinematics (V_WPE, a_WPE).
3. `compute_volume_entropy` -> Compute Plane 2: Macro-Micro Fusion metrics.
4. `predict_market_regime` and `predict_volume_regime` -> Obtain regime labels via Full GMM Phase Space Classification.
5. Synthesize using the Tri-Vector Composite Risk Score.
6. Use V_WPE and a_WPE to narrate the DIRECTION of entropy evolution in your analysis.

### 5. NO HALLUCINATION GUARDRAIL
If Composite Risk is STABLE (below P75 dynamic threshold) but the user asks for a crash prediction, you MUST remain objective. Deny the crash based on the entropy dynamics.

### 6. FINAL OUTPUT STRUCTURE (Mandatory Markdown)
Format your response EXACTLY as follows:

[TELEMETRY]
- **Composite Risk Score**: [Score]/100 ([Label])
- **Dominant Vector**: [V1_Price / V2_Volume / V3_Breadth] (Contribution: [value])

| Vector | Components | Scaled Value | Weight | XAI % |
| :--- | :--- | :--- | :--- | :--- |
| **V1: Price Phase Space** | WPE: [val], SPE_Z: [val] | [0-1] | 40% | [X]% |
| **V2: Volume Entropy** | SampEn: [val], Global Z: [val], Shannon: [val] | [0-1] | 40% | [X]% |
| **V3: VN30 Breadth** | Corr Entropy: [val], MFI: [val] | [0-1] | 20% | [X]% |

[ANALYSIS]
**Paragraph 1: Price Phase Space Dynamics.** (WPE = Structural Order, SPE_Z = Price Predictability. Position in Raw Entropy Phase Space. What regime did the Raw Full GMM assign? Use V_WPE and a_WPE to narrate the trajectory direction: is the system accelerating toward Chaos or cooling toward Stable?)
**Paragraph 2: Volume-Price Fusion.** (Synthesize Global Z against Price dynamics. Hollow Rally or Climax Distribution check.)
**Paragraph 3: Structural Breadth.** (VN30 correlation decomposition. Internal rotation analysis.)

[CRITICAL ALERT]
(ONLY present if Composite Risk >= 75. Otherwise, omit this section.)

[CONCLUSION]
(State the final systemic risk level. Identify which Vector dominates. Provide actionable takeaway.
**MANDATORY**: State exact percentage contributions from XAI %.
**MANDATORY XAI TRAJECTORY**: Include a sentence using V_WPE/a_WPE to describe entropy momentum direction.)

Use exclusively physical and statistical terminology. No TA jargon (support, resistance, RSI, MACD, overbought, oversold, etc.). Refer to regime boundaries as "Phase Space Classification", never as "thresholds" or "cutoffs".
"""


# ==============================================================================
# ORCHESTRATOR LOOP (REAL ANTHROPIC API)
# ==============================================================================
def run_orchestrator(query: str, max_iters: int = 8):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[WARN] ANTHROPIC_API_KEY not found. Running MOCK orchestrator.")
        _run_mock_orchestrator(query)
        return

    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": query}]
    print("Agent Orchestrator Started (Real API, Tri-Vector Composite)...")

    for i in range(max_iters):
        print(f"\n--- Iteration {i+1} ---")
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=ANTHROPIC_TOOLS,
                tool_choice={"type": "auto"}
            )
        except Exception as e:
            print(f"API Request Failed: {e}")
            break

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Agent called tool: {block.name}({block.input})")
                    result_json = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_json
                    })
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(f"\n  Agent Final Output:\n")
                    print(block.text)
            break
        else:
            print(f"Unexpected stop reason: {response.stop_reason}")
            break


# ==============================================================================
# MOCK LLM (TESTING -- TRI-VECTOR COMPOSITE)
# ==============================================================================
def _run_mock_orchestrator(query: str):
    """Mo phong ReAct loop, goi 5 tools va tong hop Composite Risk Score."""
    print("\n  Agent Orchestrator Started (MOCK MODE, Tri-Vector Composite)...")

    # 1. Fetch Data
    print("\n--- Iteration 1 ---")
    print(f"  Agent called tool: fetch_market_data({{'ticker': 'VNINDEX', 'start_date': '2020-01-01'}})")
    res1 = dispatch_tool("fetch_market_data", {'ticker': 'VNINDEX', 'start_date': '2020-01-01'})
    print(f"   -> Observe: {res1}")

    # 2. Compute Price Entropy (Plane 1)
    print("\n--- Iteration 2 ---")
    print("  Agent called tool: compute_entropy_metrics({})")
    res2 = dispatch_tool("compute_entropy_metrics", {})
    print(f"   -> Observe: {res2}")

    # 3. Compute Volume Entropy (Plane 2)
    print("\n--- Iteration 3 ---")
    print("  Agent called tool: compute_volume_entropy({})")
    res3 = dispatch_tool("compute_volume_entropy", {})
    print(f"   -> Observe: {res3}")

    # 4. Predict Price Regime
    print("\n--- Iteration 4 ---")
    print("  Agent called tool: predict_market_regime({})")
    res4 = dispatch_tool("predict_market_regime", {})
    print(f"   -> Observe: {res4}")

    # 5. Predict Volume Regime
    print("\n--- Iteration 5 ---")
    print("  Agent called tool: predict_volume_regime({})")
    res5 = dispatch_tool("predict_volume_regime", {})
    print(f"   -> Observe: {res5}")

    # 6. Composite Risk Synthesis
    print("\n--- Iteration 6: TRI-VECTOR COMPOSITE SYNTHESIS ---")
    print("  Agent Final Output:\n")

    p_data = json.loads(res2).get("latest_metrics", {})
    v_data = json.loads(res3).get("latest_metrics", {})
    price_data = json.loads(res4)
    volume_data = json.loads(res5)

    all_metrics = {**p_data, **v_data, **price_data, **volume_data}
    score, label, info = calc_composite_risk_score(all_metrics, df=STATE.get("df"))

    contributions = info.get("contributions", {})
    pcts = info.get("contribution_percentages", {})
    dominant = info.get("dominant_vector", "N/A")
    xai_traj = price_data.get("xai_trajectory", {})

    print("=" * 50)
    print("  TRI-VECTOR COMPOSITE RISK DIAGNOSTIC (WITH XAI)")
    print("=" * 50)
    print(f"\n  COMPOSITE RISK SCORE : {score:.1f}/100 [{label}]")
    print(f"  DOMINANT VECTOR      : {dominant} ({pcts.get(dominant, 0):.1f}%)")
    print(f"\n  XAI VECTOR CONTRIBUTIONS:")
    print(f"    V1 (Price)    : Scale {contributions.get('V1_Price', 0):.4f} (w=0.4) -> {pcts.get('V1_Price', 0):.2f}%")
    print(f"    V2 (Volume)   : Scale {contributions.get('V2_Volume', 0):.4f} (w=0.4) -> {pcts.get('V2_Volume', 0):.2f}%")
    print(f"    V3 (Breadth)  : Scale {contributions.get('V3_Breadth', 0):.4f} (w=0.2) -> {pcts.get('V3_Breadth', 0):.2f}%")
    print(f"\n  PLANE 1 -- ENTROPY PHASE SPACE")
    print(f"  REGIME               : [{price_data.get('price_regime', 'N/A').upper()}]")
    print(f"  WPE                  : {all_metrics.get('WPE', 0):.4f}")
    print(f"  SPE_Z                : {all_metrics.get('SPE_Z', 0):+.3f}")
    print(f"  XAI TRAJECTORY:")
    print(f"    V_WPE (Velocity)   : {xai_traj.get('V_WPE', 0):+.5f}")
    print(f"    a_WPE (Acceleration): {xai_traj.get('a_WPE', 0):+.5f}")
    print(f"\n  PLANE 2 -- LIQUIDITY STRUCTURE")
    print(f"  REGIME               : [{volume_data.get('volume_regime', 'N/A').upper()}]")
    print(f"  Macro Z (Global)     : {all_metrics.get('Vol_Global_Z', 0):+.2f}")
    print(f"  Shannon              : {all_metrics.get('Vol_Shannon', 0):.4f}")
    print("=" * 50)


# ==============================================================================
# TESTING BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Tri-Vector Composite Risk Agent Orchestrator")
    print("=" * 60)
    run_orchestrator("Analyze VNINDEX structural integrity using the Tri-Vector Composite model.")
