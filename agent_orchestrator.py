"""
Agent Orchestrator -- InfoStat Dynamics
ReAct Loop + Anthropic Tool Use Protocol.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import anthropic

from skills.data_skill import get_latest_market_data
from skills.quant_skill import calc_rolling_wpe, calc_mfi
from skills.ds_skill import fit_predict_regime

warnings.filterwarnings("ignore")


# ==============================================================================
# GLOBAL STATE (Luu tru trang thai cho agent context)
# ==============================================================================
STATE = {
    "df": None,
    "metrics_computed": False,
    "classifier": None,
}


# ==============================================================================
# TOOL IMPLEMENTATIONS (Python Callables)
# ==============================================================================
def tool_fetch_market_data(ticker="VNINDEX", start_date="2024-01-01"):
    print(f"  [Tool Execution] Fetching {ticker} from {start_date}...")
    df = get_latest_market_data(ticker=ticker, start_date=start_date)
    STATE["df"] = df
    STATE["metrics_computed"] = False
    return json.dumps({
        "status": "success", 
        "rows": len(df),
        "latest_close": float(df["Close"].iloc[-1])
    })


def tool_compute_entropy_metrics():
    print("  [Tool Execution] Computing Entropy Metrics (WPE, C, MFI)...")
    df = STATE.get("df")
    if df is None:
        return json.dumps({"error": "No market data available. Call fetch_market_data first."})
    
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).values
    
    wpe_arr, c_arr = calc_rolling_wpe(log_returns, m=3, tau=1, window=22)
    mfi_arr = calc_mfi(wpe_arr, c_arr)
    
    df["WPE"] = wpe_arr
    df["Complexity"] = c_arr
    df["MFI"] = mfi_arr
    
    STATE["df"] = df
    STATE["metrics_computed"] = True
    
    latest = df.dropna().iloc[-1]
    return json.dumps({
        "status": "success",
        "latest_WPE": float(latest["WPE"]),
        "latest_Complexity": float(latest["Complexity"]),
        "latest_MFI": float(latest["MFI"])
    })


def tool_predict_market_regime():
    print("  [Tool Execution] Predicting Market Regime via GMM...")
    df = STATE.get("df")
    if df is None or not STATE.get("metrics_computed"):
        return json.dumps({"error": "Data/Metrics missing. Compute metrics first."})
    
    valid_df = df.dropna(subset=["WPE", "Complexity", "MFI"]).copy()
    features = valid_df[["WPE", "Complexity", "MFI"]].values
    
    labels, clf = fit_predict_regime(features, n_components=3)
    valid_df["RegimeLabel"] = labels
    valid_df["RegimeName"] = [clf.get_regime_name(lbl) for lbl in labels]
    
    STATE["df"] = valid_df
    STATE["classifier"] = clf
    
    latest = valid_df.iloc[-1]
    return json.dumps({
        "status": "success", 
        "current_regime": str(latest["RegimeName"]),
        "mfi": float(latest["MFI"])
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
    elif tool_name == "predict_market_regime":
        return tool_predict_market_regime()
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ==============================================================================
# ANTHROPIC TOOL SCHEMAS
# ==============================================================================
ANTHROPIC_TOOLS = [
    {
        "name": "fetch_market_data",
        "description": "Fetch real-time daily Open, High, Low, Close, Volume data for a specific stock or index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The ticker symbol (e.g., VNINDEX)."},
                "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format."}
            },
            "required": ["ticker", "start_date"]
        }
    },
    {
        "name": "compute_entropy_metrics",
        "description": "Compute advanced Information-Theoretic metrics (WPE, Statistical Complexity, Market Fragility Index) on the fetched data.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "predict_market_regime",
        "description": "Classify the current market regime using a Gaussian Mixture Model based on the computed entropy metrics. Returns the regime name.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]


# ==============================================================================
# ORCHESTRATOR LOOP (REAL ANTHROPIC API)
# ==============================================================================
def run_orchestrator(query: str, max_iters: int = 5):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[WARN] ANTHROPIC_API_KEY not found. Running MOCK orchestrator sequence for testing.")
        _run_mock_orchestrator(query)
        return

    client = anthropic.Anthropic(api_key=api_key)
    
    system_prompt = """
You are the InfoStat Dynamics System Architect.
Your task is to analyze the market by:
1. Fetching data (fetch_market_data).
2. Computing entropy metrics (compute_entropy_metrics).
3. Predicting the regime (predict_market_regime).
4. Analyzing the results. You MUST output a structured Risk Warning if the regime indicates 'Volatile', 'Panic', 'Chaos', or 'Fragile'.
Always use the tools sequentially before making conclusions.
"""
    
    messages = [{"role": "user", "content": query}]
    print("🤖 Agent Orchestrator Started (Real API)...")
    
    for i in range(max_iters):
        print(f"\n--- Iteration {i+1} ---")
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                system=system_prompt,
                messages=messages,
                tools=ANTHROPIC_TOOLS,
                tool_choice={"type": "auto"}
            )
        except Exception as e:
            print(f"API Request Failed: {e}")
            break
        
        # Save assistant message
        messages.append({"role": "assistant", "content": response.content})
        
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"🛠️  Agent called tool: {block.name}({block.input})")
                    result_json = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_json
                    })
            # Send results back to LLM
            messages.append({"role": "user", "content": tool_results})
            
        elif response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(f"\n📢 Agent Final Output:\n")
                    print(block.text)
            break
        else:
            print(f"Unexpected stop factor: {response.stop_reason}")
            break


# ==============================================================================
# MOCK LLM (TESTING PURPOSES ONLY)
# ==============================================================================
def _run_mock_orchestrator(query: str):
    """
    Mo phong vong lap ReAct, tu dong ra quyet dinh goi dung cac tool lien tiep 
    neu khong co key API nao.
    """
    print("\n🤖 Agent Orchestrator Started (MOCK MODE)...")
    
    # 1. Thuc thi Tool 1
    print("\n--- Iteration 1 ---")
    print("🛠️  Agent called tool: fetch_market_data({'ticker': 'VNINDEX', 'start_date': '2024-01-01'})")
    res1 = dispatch_tool("fetch_market_data", {'ticker': 'VNINDEX', 'start_date': '2024-01-01'})
    print(f"   -> LLM Observe: {res1}")
    
    # 2. Thuc thi Tool 2
    print("\n--- Iteration 2 ---")
    print("🛠️  Agent called tool: compute_entropy_metrics({})")
    res2 = dispatch_tool("compute_entropy_metrics", {})
    print(f"   -> LLM Observe: {res2}")
    
    # 3. Thuc thi Tool 3
    print("\n--- Iteration 3 ---")
    print("🛠️  Agent called tool: predict_market_regime({})")
    res3 = dispatch_tool("predict_market_regime", {})
    print(f"   -> LLM Observe: {res3}")
    
    # 4. Giai doan tra ve Final Response cua LLM
    print("\n--- Iteration 4 ---")
    print("📢 Agent Final Output:")
    
    regime_data = json.loads(res3)
    regime = regime_data.get("current_regime", "Unknown")
    
    print(f"Analysis Complete. The current state of VNINDEX is calculated as: **{regime}**.")
    
    if "Chaos" in regime or "Panic" in regime or "Fragile" in regime:
        print("\n==========================================")
        print("⚠️  RISK WARNING: STRUCTURAL VULNERABILITY")
        print("==========================================")
        print(f"The complex systems physics engine detects an MFI of {regime_data.get('mfi', 0):.4f}.")
        print("Market structure is exhibiting severe fragility and entropy saturation.")
        print("Capital flow is highly fragmented. Precautionary risk management is advised.")
    else:
        print("The market implies structural integrity and stable forward momentum.")


# ==============================================================================
# TESTING BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print("TEST: Agent Orchestrator ReAct Loop and Tool Calling Structure")
    print("="*60)
    run_orchestrator("Please check the VNINDEX dynamics and see if the market is Fragile.")
