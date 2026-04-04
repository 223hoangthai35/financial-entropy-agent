"""
Agent Orchestrator -- Financial Entropy Agent (Dual-Plane Engine)
ReAct Loop + Anthropic Tool Use Protocol.
Cross-Plane Reasoning: Price Dynamics x Liquidity Structure.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import anthropic

from skills.data_skill import get_latest_market_data
from skills.quant_skill import calc_rolling_wpe, calc_mfi, calc_rolling_volume_entropy
from skills.ds_skill import fit_predict_regime, fit_predict_volume_regime

warnings.filterwarnings("ignore")


# ==============================================================================
# GLOBAL STATE (Luu tru trang thai cho agent context)
# ==============================================================================
STATE = {
    "df": None,
    "metrics_computed": False,
    "volume_metrics_computed": False,
    "price_classifier": None,
    "volume_classifier": None,
}


# ==============================================================================
# TOOL IMPLEMENTATIONS (Python Callables)
# ==============================================================================
def tool_fetch_market_data(ticker="VNINDEX", start_date="2024-01-01"):
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
    """Plane 1: Tinh WPE, Complexity, MFI tu Price data."""
    print("  [Tool Execution] Computing Price Entropy Metrics (WPE, C, MFI)...")
    df = STATE.get("df")
    if df is None:
        return json.dumps({"error": "No market data. Call fetch_market_data first."})

    log_returns = np.log(df["Close"] / df["Close"].shift(1)).values
    df["Volatility"] = pd.Series(log_returns, index=df.index).rolling(20).std() * np.sqrt(252) * 100

    wpe_arr, c_arr = calc_rolling_wpe(log_returns, m=3, tau=1, window=22)
    mfi_arr = calc_mfi(wpe_arr, c_arr)

    df["WPE"] = wpe_arr
    df["Complexity"] = c_arr
    df["MFI"] = mfi_arr

    # Kinematic Vectors: 3-day momentum derivatives of PE
    df["PE_Velocity"] = df["WPE"].diff(3).fillna(0)
    df["PE_Acceleration"] = df["PE_Velocity"].diff(3).fillna(0)

    STATE["df"] = df
    STATE["metrics_computed"] = True

    latest = df.dropna(subset=["WPE"]).iloc[-1]
    return json.dumps({
        "status": "success",
        "latest_WPE": float(latest["WPE"]),
        "latest_Complexity": float(latest["Complexity"]),
        "latest_MFI": float(latest["MFI"]),
        "latest_Volatility": float(latest["Volatility"]),
        "PE_Velocity": float(latest["PE_Velocity"]),
        "PE_Acceleration": float(latest["PE_Acceleration"]),
    })


def tool_compute_volume_entropy():
    """Plane 2: Tinh Volume Shannon Entropy, Sample Entropy, va Macro-Micro Z-Scores."""
    print("  [Tool Execution] Computing Macro-Micro Fusion Volume Entropy (window=60, z_window=252)...")
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
        "latest_Vol_Shannon": float(latest["Vol_Shannon"]),
        "latest_Vol_SampEn": float(latest["Vol_SampEn"]),
        "latest_Vol_Global_Z": float(latest["Vol_Global_Z"]),
    })


def tool_predict_market_regime():
    """Plane 1: GMM predict Price Regime."""
    print("  [Tool Execution] Predicting Price Regime via GMM (Plane 1)...")
    df = STATE.get("df")
    if df is None or not STATE.get("metrics_computed"):
        return json.dumps({"error": "Price metrics missing. Compute entropy first."})

    valid_df = df.dropna(subset=["WPE", "Complexity", "MFI"]).copy()
    features = valid_df[["WPE", "Complexity", "MFI"]].values

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
        "description": "Compute Plane 1 (Price) metrics: WPE, Statistical Complexity, MFI, Volatility.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "compute_volume_entropy",
        "description": "Compute Plane 2 (Volume) metrics using Macro-Micro Fusion: Global Z-Score (macro scale), Rolling Z-Score (micro structure), Shannon Entropy, and Sample Entropy.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "predict_market_regime",
        "description": "Classify Price regime via GMM (Plane 1). Labels: Stable Growth, Fragile Growth, Chaos/Panic.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "predict_volume_regime",
        "description": "Classify Volume regime via GMM (Plane 2). Labels: Consensus Flow, Dispersed Flow, Erratic/Noisy Flow.",
        "input_schema": {"type": "object", "properties": {}}
    },
]


# ==============================================================================
# CROSS-PLANE SYNTHESIS LOGIC
# ==============================================================================
def _cross_plane_synthesis(price_regime: str, volume_regime: str) -> tuple[str, str]:
    """
    Ma tran Cross-Plane: ket hop Price Plane va Volume Plane
    de suy ra ket luan he thong.
    Returns: (synthesis_label, explanation)
    """
    p = price_regime.upper()
    v = volume_regime.upper()

    p_fragile_chaos = "CHAOS" in p or "PANIC" in p or "FRAGILE" in p
    v_consensus = "CONSENSUS" in v
    v_erratic = "ERRATIC" in v or "NOISY" in v
    p_stable = "STABLE" in p

    if p_fragile_chaos and v_consensus:
        return (
            "STRUCTURAL ACCUMULATION",
            "Price chaos is contained by highly organized liquidity. "
            "Smart Money absorption detected. The physical disorder is a surface "
            "phenomenon; underlying capital flow remains structurally coherent."
        )
    elif p_fragile_chaos and v_erratic:
        return (
            "CRITICAL BREAKDOWN",
            "Physical price chaos is amplified by fragmented liquidity. "
            "Both planes confirm systemic instability. High systemic risk "
            "with no structural support from capital flow."
        )
    elif p_stable and v_erratic:
        return (
            "TREND EXHAUSTION",
            "Price trend appears stable, but liquidity structure is breaking "
            "down beneath the surface. The divergence between stable prices "
            "and erratic volume signals unsustainable momentum."
        )
    else:
        return (
            "SYSTEM COHERENT",
            "Both planes exhibit structural alignment. No cross-plane "
            "divergence detected. Current market dynamics are internally consistent."
        )


# ==============================================================================
# SYSTEM PROMPT (DUAL-PLANE)
# ==============================================================================
SYSTEM_PROMPT = """
You are a Dual-Plane Financial Entropy Expert specializing in Non-linear Dynamics
and Systemic Risk Analysis.

Your task is to analyze the market through TWO independent observation planes:
1. PLANE 1 (Price Dynamics): WPE, Kinematic Vectors (V, a), MFI, Volatility.
2. PLANE 2 (Liquidity Structure -- Macro-Micro Fusion): Global Z-Score (macro scale),
   Shannon Entropy + Sample Entropy computed on Rolling Z-Score (micro structure).

EXECUTION ORDER (mandatory):
1. fetch_market_data -> get OHLCV data.
2. compute_entropy_metrics -> compute Plane 1 (Price) features + Kinematic Vectors.
3. compute_volume_entropy -> compute Plane 2 (Volume) features via Macro-Micro Fusion.
4. predict_market_regime -> classify Price regime (Stable/Fragile/Chaos).
5. predict_volume_regime -> classify Volume regime (Consensus/Dispersed/Erratic).
6. Synthesize BOTH planes into a unified conclusion.

KINEMATIC VECTOR ANALYSIS:
Analyze the Kinematic Vectors (V and a) of Permutation Entropy:
- V (Velocity = dE/dt): Determines the direction. V > 0 means chaos is expanding.
  V < 0 means order is forming.
- a (Acceleration = d2E/dt2): Determines the momentum force. a > 0 means the trend
  (chaos or order) is exploding/accelerating. a < 0 means the momentum is fading/exhausting.

MACRO-MICRO FUSION HEURISTICS (Plane 2):
Evaluate Liquidity using Macro-Micro Fusion:
1. Macro Scale (Global Z): Indicates absolute systemic liquidity.
   Z > 1.5 means massive historical capital inflow. Z < -1 means systemic liquidity drought.
2. Micro Structure (Entropy Regimes): Indicates immediate trading behavior
   (Consensus vs. Erratic) based on Rolling Z-Score entropy.
3. Fusion Logic (Crucial): If Macro Scale is extremely HIGH (e.g., Z > 2.0) BUT the
   Micro Structure is 'Erratic/Noisy Flow' AND Plane 1 shows 'Critical Fragility'
   (a > 0, V > 0), diagnose this as **'Climax Distribution'** -- a major systemic
   top/bubble burst. Massive capital is present, but behavior is highly
   panicked/fragmented.

CROSS-PLANE SYNTHESIS MATRIX:
- IF Price=[Fragile/Chaos] AND Volume=[Consensus Flow] -> STRUCTURAL ACCUMULATION
  "Price chaos contained by organized liquidity. Smart Money absorption."
- IF Price=[Fragile/Chaos] AND Volume=[Erratic/Noisy Flow] -> CRITICAL BREAKDOWN
  "Price chaos amplified by fragmented liquidity. High systemic risk."
- IF Price=[Stable Growth] AND Volume=[Erratic/Noisy Flow] -> TREND EXHAUSTION
  "Stable prices but liquidity breaking down. Unsustainable momentum."

OUTPUT FORMAT INSTRUCTIONS:
You MUST format your final response EXACTLY using the following Markdown structure. Use a table for the metrics and preserve the exact headings for the text sections. Do NOT omit the [CONCLUSION].

| Telemetry Module | Key Metrics | Regime / Status |
| :--- | :--- | :--- |
| **Plane 1: Price Dynamics** | WPE: [Value] • V (dE/dt): [Value] • a (d2E/dt2): [Value] | **[Regime 1]** |
| **Plane 2: Liquidity Structure** | Macro Z: [Value] • Shannon: [Value] • SampEn: [Value] | **[Micro Regime Name]** |
| **Cross-Plane Synthesis** | Systemic Risk: [Risk Level] | **[Synthesis Label]** |

### [ANALYSIS]
(Write your detailed cross-plane reasoning here. Include Kinematic interpretation of V and a. Include Macro-Micro Fusion interpretation. Keep it concise and analytical.)

### [VN30 STRUCTURAL DYNAMICS]
(Write your analysis of internal capital rotation here. Keep it brief.)

### [CONCLUSION]
(Your final, definitive actionable takeaway here. THIS SECTION IS MANDATORY. You MUST explicitly mention the Micro-structural state using its specific Regime name (e.g., 'Dispersed Flow') and contextualize it against the Macro Scale (Global Z). For example: "The Dual-Plane Engine confirms SYSTEM COHERENT. While Macro Scale (Z: +0.55) indicates healthy capital depth, the Micro-structural regime (Dispersed Flow) ensures liquidity is distributed efficiently... Combined with Plane 1's momentum fading (a < 0), systemic risk remains MODERATE.")

Maintain an academic, quantitative tone. No speculation without data.
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
    print("Agent Orchestrator Started (Real API, Dual-Plane)...")

    for i in range(max_iters):
        print(f"\n--- Iteration {i+1} ---")
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
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
# MOCK LLM (TESTING PURPOSES -- DUAL-PLANE)
# ==============================================================================
def _run_mock_orchestrator(query: str):
    """
    Mo phong vong lap ReAct Dual-Plane, goi 5 tools lien tiep
    va tong hop Cross-Plane Synthesis.
    """
    print("\n  Agent Orchestrator Started (MOCK MODE, Dual-Plane)...")

    # 1. Fetch Data
    print("\n--- Iteration 1 ---")
    print("  Agent called tool: fetch_market_data({'ticker': 'VNINDEX', 'start_date': '2024-01-01'})")
    res1 = dispatch_tool("fetch_market_data", {'ticker': 'VNINDEX', 'start_date': '2024-01-01'})
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

    # 6. Cross-Plane Synthesis
    print("\n--- Iteration 6: CROSS-PLANE SYNTHESIS ---")
    print("  Agent Final Output:\n")

    price_data = json.loads(res4)
    volume_data = json.loads(res5)
    price_regime = price_data.get("price_regime", "Unknown")
    volume_regime = volume_data.get("volume_regime", "Unknown")
    vol_global_z = volume_data.get("vol_global_z", float("nan"))

    synthesis_label, synthesis_detail = _cross_plane_synthesis(price_regime, volume_regime)

    risk_level = "CRITICAL" if synthesis_label == "CRITICAL BREAKDOWN" else (
        "ELEVATED" if synthesis_label == "TREND EXHAUSTION" else "MODERATE"
    )

    vol_gz_display = f"{vol_global_z:+.2f}" if not np.isnan(vol_global_z) else "N/A"

    print("=" * 50)
    print("  DUAL-PLANE DIAGNOSTIC REPORT (MACRO-MICRO FUSION)")
    print("=" * 50)
    print(f"\n  PLANE 1 -- PRICE DYNAMICS")
    print(f"  REGIME          : [{price_regime.upper()}]")
    print(f"  MFI             : {price_data.get('mfi', 0):.4f}")
    pe_v = price_data.get('PE_Velocity', 0)
    pe_a = price_data.get('PE_Acceleration', 0)
    print(f"  PE Velocity (V) : {pe_v:+.4f} ({'chaos expanding' if pe_v > 0 else 'order forming'})")
    print(f"  PE Accel (a)    : {pe_a:+.4f} ({'momentum accelerating' if pe_a > 0 else 'momentum fading'})")
    print(f"\n  PLANE 2 -- LIQUIDITY STRUCTURE (MACRO-MICRO FUSION)")
    print(f"  MICRO REGIME    : [{volume_regime.upper()}]")
    print(f"  Macro Z (Global): {vol_gz_display}")
    print(f"  Vol Shannon     : {volume_data.get('vol_shannon', 0):.4f}")
    print(f"  Vol SampEn      : {volume_data.get('vol_sampen', 0):.4f}")
    print(f"\n  CROSS-PLANE SYNTHESIS")
    print(f"  CONCLUSION      : [{synthesis_label}]")
    print(f"  SYSTEMIC RISK   : [{risk_level}]")
    print(f"\n  {synthesis_detail}")
    print("=" * 50)


# ==============================================================================
# TESTING BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Dual-Plane Agent Orchestrator ReAct Loop")
    print("=" * 60)
    run_orchestrator("Analyze VNINDEX with Cross-Plane synthesis. Is the market structurally sound?")
