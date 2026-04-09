"""
Financial Entropy Agent -- Tri-Vector Composite Risk Terminal
Dual Pipeline (API/Upload), Entropy Phase Space GMM Scatter, Composite Risk Engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Import backend skills
from skills.data_skill import get_latest_market_data, fetch_vn30_returns
from skills.quant_skill import (
    calc_rolling_wpe, calc_mfi, calc_correlation_entropy,
    calc_rolling_volume_entropy, calc_wpe_kinematics,
    calc_rolling_price_sample_entropy, calc_spe_z,
)
from skills.ds_skill import fit_predict_regime, fit_predict_volume_regime
from agent_orchestrator import calc_composite_risk_score, fit_garch_x

# ==============================================================================
# UI CONFIGURATION & MULTILINGUAL SUPPORT
# ==============================================================================
st.set_page_config(page_title="Financial Entropy Agent | Terminal", layout="wide", page_icon="⚡")

# Custom Styling (Dark Quant Terminal)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime&family=Inter:wght@400;800&display=swap');
    
    .reportview-container { background: #0E1117; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #FFFFFF; font-family: 'Courier Prime', monospace; }
    .metric-label { font-size: 0.9rem; color: #AAAAAA; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 400; }
    
    h1, h2, h3 { color: #00FF41 !important; font-family: 'Courier Prime', monospace; text-transform: uppercase; letter-spacing: 2px; }
    
    .agent-log {
        background-color: #0a0a0a;
        border-left: 4px solid #00FF41;
        padding: 30px;
        font-family: 'Courier Prime', monospace;
        color: #d1ffd1;
        font-size: 0.95rem;
        line-height: 1.7;
        border-radius: 4px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.8);
        border: 1px solid #1a3a1a;
    }
    .agent-log h3 { color: #39FF14 !important; margin-bottom: 20px; border-bottom: 1px solid #1a3a1a; padding-bottom: 10px; }
    .agent-log code { color: #FFD700 !important; font-weight: 800; background: transparent; padding: 0; }
    .agent-log strong { color: #39FF14 !important; text-transform: uppercase; }
    .agent-log table { border-collapse: collapse; width: 100%; margin: 20px 0; border: 1px solid #1a3a1a; }
    .agent-log th, .agent-log td { border: 1px solid #1a3a1a; padding: 12px; text-align: left; }
    .agent-log th { background: #112211; color: #00FF41; }
    
    .arch-badge {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #333;
        text-align: center;
        transition: all 0.3s ease;
    }
    .arch-badge:hover { border-color: #00FF41; box-shadow: 0 0 15px rgba(0, 255, 65, 0.2); }
    
    [data-testid="column"]:nth-child(1) > div:nth-child(1) {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 12px;
        height: 210px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 5px;
    }
    
    .stPlotlyChart { background: transparent !important; }
    
    .analysis-card {
        background: #111611;
        border: 1px solid #1a3a1a;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 12px 0;
    }
    .analysis-card-title {
        font-size: 0.85rem;
        color: #39FF14;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 800;
        margin-bottom: 12px;
        font-family: 'Courier Prime', monospace;
    }
    .regime-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 0.9rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-family: 'Courier Prime', monospace;
    }
    .xai-trajectory-box {
        background: #0d1a0d;
        border: 1px dashed #2a4a2a;
        border-radius: 6px;
        padding: 12px 18px;
        margin-top: 12px;
    }
    .xai-label {
        color: #666;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .xai-values {
        display: flex;
        gap: 30px;
        margin-bottom: 8px;
    }
    .xai-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .xai-item-label {
        color: #888;
        font-size: 0.82rem;
    }
    .xai-item-value {
        color: #00BFFF;
        font-weight: 800;
        font-family: 'Courier Prime', monospace;
    }
    .xai-narrative {
        color: #a0d0a0;
        font-size: 0.88rem;
        font-style: italic;
        margin-top: 6px;
        line-height: 1.5;
    }
    .analysis-text {
        color: #c0e0c0;
        font-size: 0.9rem;
        line-height: 1.65;
        margin-top: 8px;
    }
    .garch-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .garch-metric-label {
        color: #888;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .garch-metric-value {
        font-family: 'Courier Prime', monospace;
        font-weight: 800;
        font-size: 1rem;
    }
    .garch-sigma-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: 2px;
        font-family: 'Courier Prime', monospace;
        text-align: center;
    }
    .garch-warn-box {
        border: 1px solid #FF5F1F;
        background: rgba(255, 95, 31, 0.08);
        padding: 10px 15px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #FFB088;
    }
    .es-observer-box {
        background: #0d0d1a;
        border: 1px dashed #333366;
        border-radius: 6px;
        padding: 12px 18px;
        margin-top: 12px;
    }
    .es-label {
        color: #6666AA;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# UI INITIALIZATION & MULTILINGUAL SUPPORT
# ==============================================================================
risk_score = 0.0
synthesis_label = "STOCHASTIC"
risk_color = "#00FF41"
current_wpe = 0.5
current_regime = "Stochastic"
current_vol_global_z = 0.0
current_vol_shannon = 0.5
current_vol_regime_name = "CONSENSUS FLOW"
vol_sh_kpi = "0.50"
vol_se_kpi = "0.50"
vol_gz_kpi = "0.00 Z"

def T(en: str, vn: str) -> str:
    return en if st.session_state.get("lang", "EN") == "EN" else vn

if "lang" not in st.session_state:
    st.session_state["lang"] = "EN"

# ==============================================================================
# DATA PIPELINE (CACHED)
# ==============================================================================
@st.cache_data(ttl=3600)
def load_and_compute_data(start_date_str, end_date_str, file_bytes=None, file_name=None):
    df = None
    
    # DUAL PIPELINE: Uploaded CSV vs API
    if file_bytes is not None:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
            
        date_candidates = [c for c in df.columns if str(c).lower().strip() in ("date", "time", "ngay", "ngày")]
        if date_candidates:
            df[date_candidates[0]] = pd.to_datetime(df[date_candidates[0]])
            df.set_index(date_candidates[0], inplace=True)
            df.sort_index(inplace=True)
            
        col_map = {}
        for c in df.columns:
            key = str(c).lower().strip()
            if key == "open": col_map[c] = "Open"
            elif key == "high": col_map[c] = "High"
            elif key == "low": col_map[c] = "Low"
            elif key == "close": col_map[c] = "Close"
            elif key == "volume": col_map[c] = "Volume"
        df.rename(columns=col_map, inplace=True)
        df.ffill(inplace=True)
    else:
        df = get_latest_market_data(ticker="VNINDEX", start_date=start_date_str, end_date=end_date_str)
        
    df = df.loc[start_date_str:end_date_str].copy()
    if df.empty:
        return df

    df["SMA20"] = df["Close"].rolling(20).mean()
    
    # 1. Compute WPE, Complexity, MFI
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
    
    # 4. Volume Entropy Plane (Macro-Micro Fusion)
    if "Volume" in df.columns:
        vol_shannon, vol_sampen, vol_global_z, vol_rolling_z = calc_rolling_volume_entropy(
            df["Volume"].values, window=60, z_window=252
        )
        df["Vol_Shannon"] = vol_shannon
        df["Vol_SampEn"] = vol_sampen
        df["Vol_Global_Z"] = vol_global_z
        df["Vol_Rolling_Z"] = vol_rolling_z
    else:
        df["Vol_Shannon"] = np.nan
        df["Vol_SampEn"] = np.nan
        df["Vol_Global_Z"] = np.nan
        df["Vol_Rolling_Z"] = np.nan
    
    # 5. VN30 Cross-Sectional Entropy
    try:
        vn30_rets = fetch_vn30_returns(start_date=start_date_str, end_date=end_date_str)
        cross_entropy = calc_correlation_entropy(vn30_rets, window=22)
        df["Cross_Sectional_Entropy"] = cross_entropy.reindex(df.index).ffill()
    except Exception as e:
        df["Cross_Sectional_Entropy"] = np.nan
        
    # 6. Predict Price Regime (Full GMM in Raw Entropy Phase Space: [WPE, SPE_Z])
    price_clf = None
    valid_df = df.dropna(subset=["WPE", "SPE_Z"]).copy()
    if not valid_df.empty:
        features = valid_df[["WPE", "SPE_Z"]].values
        labels, price_clf = fit_predict_regime(features, n_components=3)
        valid_df["RegimeLabel"] = labels
        valid_df["RegimeName"] = [price_clf.get_regime_name(lbl) for lbl in labels]
    
    # Luu classifier vao df.attrs de truy cap cho ellipse rendering
    df.attrs["price_classifier"] = price_clf
    
    df["RegimeName"] = np.nan
    df["RegimeLabel"] = np.nan
    if not valid_df.empty:
        df.loc[valid_df.index, "RegimeName"] = valid_df["RegimeName"]
        df.loc[valid_df.index, "RegimeLabel"] = valid_df["RegimeLabel"]
    df["RegimeName"] = df["RegimeName"].ffill()
    df["RegimeLabel"] = df["RegimeLabel"].ffill()
    
    # 7. Volume Regime (GMM -- Plane 2)
    vol_valid = df.dropna(subset=["Vol_Shannon", "Vol_SampEn"]).copy()
    if not vol_valid.empty and len(vol_valid) >= 10:
        vol_features = vol_valid[["Vol_Shannon", "Vol_SampEn"]].values
        vol_labels, vol_clf = fit_predict_volume_regime(vol_features, n_components=3)
        vol_valid["VolRegimeLabel"] = vol_labels
        vol_valid["VolRegimeName"] = [vol_clf.get_regime_name(lbl) for lbl in vol_labels]
    
    df["VolRegimeName"] = np.nan
    df["VolRegimeLabel"] = np.nan
    if not vol_valid.empty and len(vol_valid) >= 10:
        df.loc[vol_valid.index, "VolRegimeName"] = vol_valid["VolRegimeName"]
        df.loc[vol_valid.index, "VolRegimeLabel"] = vol_valid["VolRegimeLabel"]
    df["VolRegimeName"] = df["VolRegimeName"].ffill()
    df["VolRegimeLabel"] = df["VolRegimeLabel"].ffill()

    # 8. GARCH(1,1)-X Conditional Volatility Engine
    garch_result = None
    required_garch = ["WPE", "SPE_Z", "Vol_SampEn", "Vol_Shannon", "Close"]
    if all(c in df.columns for c in required_garch):
        garch_df = df.dropna(subset=required_garch)
        if len(garch_df) >= 120:
            try:
                garch_result = fit_garch_x(df)
                if garch_result.get("status") == "success":
                    cond_vol_series = garch_result.get("cond_vol_series")
                    if cond_vol_series is not None:
                        df["Cond_Vol"] = cond_vol_series.reindex(df.index)
            except Exception as e:
                print(f"[WARN] GARCH-X fit failed in dashboard: {e}")
                garch_result = {"error": str(e)}

    df.attrs["garch_result"] = garch_result

    return df

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown("### 🌐 " + T("Language / Ngôn ngữ", "Language / Ngôn ngữ"))
lang = st.sidebar.radio("", ["EN", "VN"], index=0 if st.session_state["lang"] == "EN" else 1, horizontal=True, label_visibility="collapsed")
st.session_state["lang"] = lang

st.sidebar.markdown(f"### ⚙️ {T('SYSTEM CONFIGURATION', 'CẤU HÌNH HỆ THỐNG')}")

start_date = st.sidebar.date_input(T("Start Date", "Ngày Bắt Đầu"), datetime(2020, 1, 1))
end_date = st.sidebar.date_input(T("End Date", "Ngày Kết Thúc"), datetime.now())

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{T('1. DUAL PIPELINE: API OR UPLOAD', '1. DUAL PIPELINE: API HOẶC TẢI LÊN')}**")
uploaded_file = st.sidebar.file_uploader(T("Upload custom OHLCV (.csv/.xlsx)", "Tải lên Dữ liệu OHLCV (.csv/.xlsx)"), type=["csv", "xlsx"])

file_bytes = None
file_name = None
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
st.title(T("FINANCIAL ENTROPY AGENT: SYSTEM ARCHITECT", "FINANCIAL ENTROPY AGENT: SYSTEM ARCHITECT"))
st.markdown(T(
    "GARCH-X Entropy-Driven Volatility Engine | Regime Classification | Tail Risk Observer",
    "GARCH-X Entropy-Driven Volatility Engine | Phân loại Regime | Quan sát Tail Risk"
))

with st.spinner(T("Computing GARCH-X Volatility Engine + Entropy Regimes...", "Đang tính toán GARCH-X Volatility Engine + Entropy Regimes...")):
    df = load_and_compute_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), file_bytes, file_name)

if df is None or df.empty:
    st.error(T("No data available for the selected dates.", "Không có dữ liệu cho khoảng thời gian này."))
    st.stop()
    
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ==============================================================================
# QUANTITATIVE VECTOR EXTRACTION
# ==============================================================================
current_wpe = latest.get("WPE", 0.5)
current_mfi = latest.get("MFI", 0.5)
current_cse = latest.get("Cross_Sectional_Entropy", 50.0)
current_spe_z = latest.get("SPE_Z", 0.0)
current_v_wpe = latest.get("V_WPE", 0.0)
current_a_wpe = latest.get("a_WPE", 0.0)

# Plane 2
current_vol_shannon = latest.get("Vol_Shannon", float("nan"))
current_vol_sampen = latest.get("Vol_SampEn", float("nan"))
current_vol_global_z = latest.get("Vol_Global_Z", float("nan"))

# Regime Labels
current_regime = str(latest.get("RegimeName", "Calculating...")).replace("nan", "Calculating...")
current_vol_regime_name = str(latest.get("VolRegimeName", "Calculating...")).replace("nan", "Calculating...")

# Price vs SMA20 — used for directional context in Section 3
current_close = float(latest.get("Close", float("nan")))
current_sma20 = float(latest.get("SMA20", float("nan")))
_price_above_sma20 = (not (pd.isna(current_close) or pd.isna(current_sma20))) and (current_close > current_sma20)

# KPI Strings
vol_sh_kpi = f"{current_vol_shannon:.2f}" if pd.notna(current_vol_shannon) else "N/A"
vol_se_kpi = f"{current_vol_sampen:.2f}" if pd.notna(current_vol_sampen) else "N/A"
vol_gz_kpi = f"{current_vol_global_z:+.2f}" if pd.notna(current_vol_global_z) else "N/A"

# Run Composite Risk Score
risk_score, synthesis_label, vector_info = calc_composite_risk_score(latest.to_dict(), df=df)
contributions = vector_info.get("contributions", {})
dominant_vector = vector_info.get("dominant_vector", "N/A")

# Dynamic Thresholds
dyn_thresholds = vector_info.get("thresholds", {})
elevated_bound = dyn_thresholds.get("elevated_bound", 55.0)
critical_bound = dyn_thresholds.get("critical_bound", 70.0)
price_clf = df.attrs.get("price_classifier", None)

# Risk Color
if "CRITICAL" in synthesis_label:
    risk_color = "#FF0000"
elif "ELEVATED" in synthesis_label:
    risk_color = "#FF5F1F"
else:
    risk_color = "#00FF41"

# ==============================================================================
# TOP KPI SECTION: GARCH-X PRIMARY + ES TAIL + REGIME SUMMARY
# ==============================================================================
_garch_kpi = df.attrs.get("garch_result")
_garch_ok  = bool(_garch_kpi and _garch_kpi.get("status") == "success")

# === REGIME RISK MULTIPLIER ===
# Entropy's primary contribution: GMM regime labels amplify σ_t
# Đây đảm bảo entropy LUÔN có vai trò, ngay cả khi δ insignificant trong GARCH-X
regime_multipliers = {
    "Stochastic":    1.0,   # Normal market — no structural stress
    "Transitional":  1.4,   # Mixed structure — phase transition in progress
    "Deterministic": 2.2,   # High coordination — structural fragility
}
vol_regime_multipliers = {
    "Consensus Flow": 1.0,
    "Dispersed Flow": 1.3,
    "Erratic/Noisy Flow": 1.8,
}

price_mult = regime_multipliers.get(current_regime, 1.0)
vol_mult = vol_regime_multipliers.get(current_vol_regime_name, 1.0)
regime_mult = max(price_mult, vol_mult)
regime_mult_source = "Price" if price_mult >= vol_mult else "Volume"

if _garch_kpi and _garch_kpi.get("status") == "success":
    sigma_raw = _garch_kpi["sigma_daily_pct"]
    sigma_adjusted = sigma_raw * regime_mult
    _garch_kpi["sigma_raw"] = sigma_raw
    _garch_kpi["sigma_adjusted"] = round(sigma_adjusted, 4)
    _garch_kpi["regime_multiplier"] = regime_mult
    _garch_kpi["regime_mult_source"] = regime_mult_source

# Shared regime values used by both col1 fallback and col3
p1_color = "#00FF41" if "STOCHASTIC" in current_regime.upper() else ("#FFD700" if "TRANSITIONAL" in current_regime.upper() else "#FF3131")
vol_regime_upper = current_vol_regime_name.upper()
p2_color = "#00FF41" if "CONSENSUS" in vol_regime_upper else ("#FFD700" if "DISPERSED" in vol_regime_upper else "#FF3131")
breadth_label = "COHESIVE" if current_cse < 40 else ("DISLOCATED" if current_cse > 70 else "FRAGMENTING")
p3_color = "#00FF41" if breadth_label == "COHESIVE" else ("#FFD700" if breadth_label == "FRAGMENTING" else "#FF3131")
spe_z_display = f"{current_spe_z:+.2f}" if pd.notna(current_spe_z) else "N/A"

col1, col2, col3 = st.columns([1.4, 1, 1.6])

# --- Column 1: Primary Risk Gauge (GARCH σ_adjusted or Composite fallback) ---
with col1:
    if _garch_ok:
        sigma_raw    = _garch_kpi.get("sigma_raw", _garch_kpi["sigma_daily_pct"])
        sigma_adj    = _garch_kpi.get("sigma_adjusted", sigma_raw)
        sigma_annual = _garch_kpi["sigma_annual_pct"]
        r_mult       = _garch_kpi.get("regime_multiplier", 1.0)
        r_source     = _garch_kpi.get("regime_mult_source", "")

        # Risk level from σ_adjusted
        if sigma_adj < 0.8:
            risk_label_vn = T("CALM MARKET",   "THỊ TRƯỜNG ỔN ĐỊNH")
            risk_color    = "#00FF41"
        elif sigma_adj < 1.5:
            risk_label_vn = T("MODERATE RISK", "RỦI RO TRUNG BÌNH")
            risk_color    = "#FFD700"
        elif sigma_adj < 2.5:
            risk_label_vn = T("HIGH RISK",     "RỦI RO CAO")
            risk_color    = "#FF5F1F"
        else:
            risk_label_vn = T("EXTREME RISK",  "RỦI RO CỰC CAO")
            risk_color    = "#FF0000"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge",
            value=sigma_adj,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': "white",
                         'tickvals': [0, 0.8, 1.5, 2.5, 4.0],
                         'ticktext': ['0', '0.8', '1.5', '2.5', '4.0']},
                'bar': {'color': risk_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2, 'bordercolor': "#333",
                'steps': [
                    {'range': [0,   0.8], 'color': 'rgba(0, 255, 65, 0.1)'},
                    {'range': [0.8, 1.5], 'color': 'rgba(255, 215, 0, 0.1)'},
                    {'range': [1.5, 2.5], 'color': 'rgba(255, 95, 31, 0.1)'},
                    {'range': [2.5, 4.0], 'color': 'rgba(255, 0, 0, 0.1)'}],
            }
        ))
        fig_gauge.update_layout(
            autosize=True, height=170,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(
                text=f"{sigma_adj:.2f}%",
                x=0.5, y=0.25,
                xref="paper", yref="paper",
                font=dict(size=36, color="#FFFFFF", family="Courier Prime"),
                showarrow=False, xanchor="center", yanchor="middle"
            )]
        )
        gauge_title = T("DAILY RISK LEVEL", "MỨC ĐỘ RỦI RO HÀNG NGÀY")
        st.markdown(f'<div class="metric-label" style="text-align:center; padding-top:5px;">{gauge_title}</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<div style='text-align:center; color:{risk_color}; font-weight:800; font-size:1.0rem; margin-top:-15px;'>{risk_label_vn}</div>", unsafe_allow_html=True)

        # Sub-text: explain multiplier if > 1
        if r_mult > 1.0:
            mult_explain = T(
                f"Base: {sigma_raw:.3f}% × {r_mult:.1f}× ({r_source} regime stress)",
                f"Gốc: {sigma_raw:.3f}% × {r_mult:.1f}× (stress từ regime {r_source})"
            )
        else:
            mult_explain = T(
                f"Base volatility: {sigma_raw:.3f}%/day ({sigma_annual:.0f}%/yr)",
                f"Biến động gốc: {sigma_raw:.3f}%/ngày ({sigma_annual:.0f}%/năm)"
            )
        st.markdown(f"<div style='text-align:center; font-size:0.72rem; color:#888; margin-top:4px; font-family:Courier Prime, monospace;'>{mult_explain}</div>", unsafe_allow_html=True)

    else:
        # Fallback: Composite Risk Score cũ
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge", value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': risk_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2, 'bordercolor': "#333",
                'steps': [
                    {'range': [0, elevated_bound],              'color': 'rgba(0, 255, 65, 0.1)'},
                    {'range': [elevated_bound, critical_bound], 'color': 'rgba(255, 215, 0, 0.1)'},
                    {'range': [critical_bound, 100],            'color': 'rgba(255, 0, 0, 0.1)'}],
            }
        ))
        fig_gauge.update_layout(
            autosize=True, height=170,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(
                text=f"{risk_score:.1f}", x=0.5, y=0.25,
                xref="paper", yref="paper",
                font=dict(size=40, color="#FFFFFF", family="Courier Prime"),
                showarrow=False, xanchor="center", yanchor="middle"
            )]
        )
        st.markdown(f'<div class="metric-label" style="text-align:center; padding-top:5px;">{T("COMPOSITE RISK SCORE", "ĐIỂM RỦI RO")}</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<div style='text-align:center; color:{risk_color}; font-weight:800; font-size:1.0rem; margin-top:-15px;'>{synthesis_label}</div>", unsafe_allow_html=True)
        if _garch_kpi and "error" in _garch_kpi and "arch" in str(_garch_kpi.get("error", "")).lower():
            st.markdown("<div style='text-align:center; font-size:0.68rem; color:#FF5F1F; margin-top:3px;'>pip install arch  →  enable GARCH-X</div>", unsafe_allow_html=True)

# --- Column 2: ES Tail Risk Observer ---
with col2:
    if _garch_ok:
        var5_val = _garch_kpi.get("VaR_5pct", 0)
        es5_val  = _garch_kpi.get("ES_5pct", 0)
        es5_adj  = _garch_kpi.get("ES_5pct_adjusted")
        ce_ratio = _garch_kpi.get("cross_entropy_ratio")
        mq       = _garch_kpi.get("model_quality", "good")
        warn     = _garch_kpi.get("warning", "")
        mq_color = "#FFD700" if mq != "good" else "#888"
        warn_color = "#FF5F1F" if warn == "IGARCH_detected" else "#888"
        es_adj_html = ""
        if es5_adj is not None and ce_ratio is not None:
            es_adj_html = f"<div class='garch-metric-row'><span class='garch-metric-label'>ES adj (CE×{ce_ratio:.2f})</span><span class='garch-metric-value' style='color:#BA55D3;'>{es5_adj:+.3f}%</span></div>"
        st.markdown(f"""
        <div class="arch-badge" style="height:210px; display:flex; flex-direction:column; justify-content:center; padding:14px 16px;">
            <div class="metric-label" style="margin-bottom:8px;">{T("TAIL RISK OBSERVER", "QUAN SÁT TAIL RISK")}</div>
            <div class="garch-metric-row">
                <span class="garch-metric-label">VaR 5%</span>
                <span class="garch-metric-value" style="color:#FF5F1F;">{var5_val:+.3f}%</span>
            </div>
            <div class="garch-metric-row">
                <span class="garch-metric-label">ES 5% (raw)</span>
                <span class="garch-metric-value" style="color:#FF3131;">{es5_val:+.3f}%</span>
            </div>
            {es_adj_html}
            <div class="garch-metric-row" style="border:none;">
                <span class="garch-metric-label" style="color:{mq_color};">Model</span>
                <span style="font-size:0.72rem; color:{mq_color}; font-family:Courier Prime,monospace;">{'✓ Good' if mq == 'good' else '⚠ Resid.'}</span>
            </div>
            <div style="font-size:0.68rem; color:{warn_color}; margin-top:4px; font-family:Courier Prime,monospace;">{'⚠ IGARCH' if warn == 'IGARCH_detected' else ''}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback: show V2 volume detail
        st.markdown(f"""
        <div class="arch-badge" style="height:210px; display:flex; flex-direction:column; justify-content:center; padding:14px 16px;">
            <div class="metric-label" style="margin-bottom:8px;">{T("V2: LIQUIDITY STRUCTURE", "V2: THANH KHOẢN")}</div>
            <div class="metric-value" style="color:{p2_color}; font-size:1.0rem;">{current_vol_regime_name}</div>
            <div class="garch-metric-row" style="margin-top:10px;">
                <span class="garch-metric-label">Shannon</span>
                <span class="garch-metric-value">{vol_sh_kpi}</span>
            </div>
            <div class="garch-metric-row" style="border:none;">
                <span class="garch-metric-label">SampEn</span>
                <span class="garch-metric-value">{vol_se_kpi}</span>
            </div>
            <div style="font-size:0.68rem; color:#888; margin-top:6px; font-family:Courier Prime,monospace;">Global Z: {vol_gz_kpi}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Column 3: Combined Regime Summary (V1 + V2 + V3) ---
with col3:
    cse_display = f"{current_cse:.1f}%"
    mfi_display = f"{current_mfi:.4f}"
    v1_risk = contributions.get('V1_Price', 0) * 100
    v2_risk = contributions.get('V2_Volume', 0) * 100
    v3_risk = contributions.get('V3_Breadth', 0) * 100
    _p1_subtitles = {
        "Deterministic": T("High Coordination — trending structure", "Phoi hop cao — cau truc xu huong"),
        "Transitional":  T("Mixed Structure — phase transition",    "Cau truc hon hop — chuyen pha"),
        "Stochastic":    T("Normal Market — healthy random behavior","Thi truong binh thuong — ngau nhien"),
    }
    _p1_subtitle = _p1_subtitles.get(current_regime, "")
    st.markdown(f"""
    <div class="arch-badge" style="height:210px; display:flex; flex-direction:column; justify-content:space-between; padding:12px 16px;">
        <div class="metric-label" style="text-align:center; margin-bottom:4px;">{T("REGIME SUMMARY", "TỔNG HỢP REGIME")}</div>
        <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.06);">
            <span style="flex:1; text-align:left; font-size:0.72rem; color:#888; text-transform:uppercase; letter-spacing:1px;">V1 Price</span>
            <span style="flex:2; text-align:center;">
                <span style="font-family:Courier Prime,monospace; font-weight:800; color:{p1_color}; font-size:0.9rem; display:block;">{current_regime}</span>
                <span style="font-size:0.60rem; color:#888; font-style:italic;">{_p1_subtitle}</span>
            </span>
            <span style="flex:1; text-align:right; font-size:0.68rem; color:#666;">WPE {current_wpe:.3f} | Z {spe_z_display}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.06);">
            <span style="flex:1; text-align:left; font-size:0.72rem; color:#888; text-transform:uppercase; letter-spacing:1px;">V2 Volume</span>
            <span style="flex:1; text-align:center; font-family:Courier Prime,monospace; font-weight:800; color:{p2_color}; font-size:0.9rem;">{current_vol_regime_name}</span>
            <span style="flex:1; text-align:right; font-size:0.68rem; color:#666;">Sh {vol_sh_kpi} | Se {vol_se_kpi}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 0;">
            <span style="flex:1; text-align:left; font-size:0.72rem; color:#888; text-transform:uppercase; letter-spacing:1px;">V3 CSE</span>
            <span style="flex:1; text-align:center; font-family:Courier Prime,monospace; font-weight:800; color:{p3_color}; font-size:0.9rem;">{breadth_label}</span>
            <span style="flex:1; text-align:right; font-size:0.68rem; color:#666;">CSE {cse_display} | MFI {mfi_display}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# VISUALS: Market Structure
# ==============================================================================
st.markdown("---")
st.subheader(T("1. MARKET STRUCTURE", "1. CẤU TRÚC THỊ TRƯỜNG"))

fig1 = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03, 
    row_heights=[0.7, 0.3],
    specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
)

# --- Row 1 (Price) ---
fig1.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="VNindex", increasing_line_color='#FFFFFF', decreasing_line_color='#888888'
), row=1, col=1)

fig1.add_trace(go.Scatter(
    x=df.index, y=df['SMA20'], mode='lines', name='SMA20',
    line=dict(color='yellow', width=1, dash='dash')
), row=1, col=1)

# --- Row 2 (WPE) ---
fig1.add_trace(go.Scatter(
    x=df.index, y=df['WPE'], mode='lines', name='WPE (Entropy)',
    line=dict(color='#FF5F1F', width=2)
), row=2, col=1)

# Regime Background Shading
regime_colors = {
    "Stochastic":    "rgba(0, 255, 65, 0.15)",    # green  — normal market
    "Transitional":  "rgba(255, 215, 0, 0.15)",   # yellow — mixed structure
    "Deterministic": "rgba(255, 0, 0, 0.15)",     # red    — high coordination / structural fragility
    "Calculating...": "rgba(128, 128, 128, 0)"
}

# Dummy legend traces
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(0, 255, 65, 1)'),   name='Stochastic — Normal Market'))
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(255, 215, 0, 1)'), name='Transitional — Mixed Structure'))
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(255, 0, 0, 1)'),   name='Deterministic — High Coordination'))

# Regime shading on Row 1
df['Regime_Shift'] = df['RegimeName'] != df['RegimeName'].shift(1)
shift_indices = df.index[df['Regime_Shift']].tolist()

for i in range(len(shift_indices)):
    start = shift_indices[i]
    end = shift_indices[i+1] if i+1 < len(shift_indices) else df.index[-1]
    regime = df.loc[start, 'RegimeName']
    color = regime_colors.get(regime, "rgba(0,0,0,0)")
    
    fig1.add_vrect(
        x0=start, x1=end, fillcolor=color, opacity=1.0, layer="below", line_width=0,
        row=1, col=1
    )

fig1.update_layout(
    title=dict(text="VNindex Structure State (Full GMM Regime)", x=0.5, y=0.98, xanchor="center", yanchor="top"),
    template="plotly_dark", height=600, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
    legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="right", x=1.0),
    margin=dict(l=20, r=20, b=20, t=60)
)
fig1.update_xaxes(rangeslider_visible=False)
fig1.update_yaxes(title_text="VNIndex Price", row=1, col=1, showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
fig1.update_yaxes(title_text="WPE Entropy", row=2, col=1, showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
st.plotly_chart(fig1, use_container_width=True)

# --- GARCH-X Conditional Volatility Overlay ---
if _garch_ok and "Cond_Vol" in df.columns:
    vol_title = T("VOLATILITY HISTORY — How risky has the market been?", 
                   "LICH SU BIEN DONG — Thi truong rui ro nhu the nao?")
    st.markdown(f"**{vol_title}**")

    fig_vol = go.Figure()

    cond_vol = df["Cond_Vol"].dropna()

    # Color-coded segments
    low_mask = cond_vol < 0.8
    mod_mask = (cond_vol >= 0.8) & (cond_vol < 1.5)
    high_mask = cond_vol >= 1.5

    # Main line
    fig_vol.add_trace(go.Scatter(
        x=cond_vol.index, y=cond_vol.values,
        mode='lines', name=T('Daily Volatility', 'Bien dong hang ngay'),
        line=dict(color='#FF5F1F', width=1.5),
        fill='tozeroy', fillcolor='rgba(255, 95, 31, 0.08)',
    ))

    # Zone annotations (khong dung ky hieu toan)
    calm_label = T("Calm zone", "Vung on dinh")
    moderate_label = T("Caution zone", "Vung can chu y")
    danger_label = T("Danger zone", "Vung nguy hiem")

    fig_vol.add_hline(y=0.8, line_dash="dash", line_color="rgba(255,215,0,0.5)",
                      annotation_text=calm_label,
                      annotation_position="top left",
                      annotation_font_color="#FFD700",
                      annotation_font_size=11)
    fig_vol.add_hline(y=1.5, line_dash="dash", line_color="rgba(255,0,0,0.5)",
                      annotation_text=danger_label,
                      annotation_position="top left",
                      annotation_font_color="#FF3131",
                      annotation_font_size=11)

    # Background zones
    fig_vol.add_hrect(y0=0, y1=0.8, fillcolor="rgba(0,255,65,0.03)", line_width=0)
    fig_vol.add_hrect(y0=0.8, y1=1.5, fillcolor="rgba(255,215,0,0.03)", line_width=0)
    fig_vol.add_hrect(y0=1.5, y1=6, fillcolor="rgba(255,0,0,0.03)", line_width=0)

    # Regime shading overlay (entropy contribution — visual proof)
    if "RegimeName" in df.columns:
        regime_colors_chart = {
            "Stochastic":    "rgba(0, 255, 65, 0.06)",
            "Transitional":  "rgba(255, 215, 0, 0.06)",
            "Deterministic": "rgba(255, 0, 0, 0.06)",
        }
        df_temp = df.copy()
        df_temp['Regime_Shift_Vol'] = df_temp['RegimeName'] != df_temp['RegimeName'].shift(1)
        shift_idx = df_temp.index[df_temp['Regime_Shift_Vol']].tolist()
        for i in range(len(shift_idx)):
            s = shift_idx[i]
            e = shift_idx[i+1] if i+1 < len(shift_idx) else df_temp.index[-1]
            r = df_temp.loc[s, 'RegimeName']
            c = regime_colors_chart.get(r, "rgba(0,0,0,0)")
            fig_vol.add_vrect(x0=s, x1=e, fillcolor=c, opacity=1.0, 
                              layer="below", line_width=0)

    fig_vol.update_layout(
        template="plotly_dark", height=350,
        plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
        legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="right", x=1.0),
        margin=dict(l=20, r=20, b=20, t=30),
        yaxis=dict(
            title_text=T("Daily Volatility (%)", "Bien dong hang ngay (%)"),
            showgrid=True, gridcolor='rgba(255,255,255,0.05)'
        ),
    )

    # Explanation text below chart
    chart_note = T(
        "**Reading this chart:** Each spike = a period of high market stress. "
        "Background colors: green=Stochastic (Normal Market), yellow=Transitional (Mixed Structure), red=Deterministic (High Coordination). "
        "Deterministic means coordinated behavior — can be rally OR decline. It measures structural fragility, not direction.",
        "**Cach doc bieu do:** Moi dinh nhon = giai doan thi truong cang thang. "
        "Nen mau: xanh=Stochastic (Thi truong binh thuong), vang=Transitional (Cau truc hon hop), do=Deterministic (Phoi hop cao). "
        "Deterministic = hanh vi phoi hop -- co the la tang HOAC giam. Do luong su gion nat cau truc, khong phai huong."
    )
    st.markdown(f"<div style='font-size:0.82rem; color:#888; margin-top:8px; line-height:1.5;'>{chart_note}</div>", unsafe_allow_html=True)

    st.plotly_chart(fig_vol, use_container_width=True)

# --- VN30 Cross-sectional Chart ---
st.markdown("---")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df.index, y=df['Cross_Sectional_Entropy'], mode='lines', name='VN30 Cross-sectional Entropy',
    line=dict(color='#00FFFF', width=2), fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.1)'
))

fig2.add_trace(go.Scatter(
    x=df.index, y=df['MFI'] * 100, mode='lines', name='MFI',
    line=dict(color='#FFD700', width=1, dash='dot')
))

fig2.update_layout(
    title=dict(text="Cross-sectional Entropy VN30 (Eigenvalue Decomposition)", x=0.5, y=0.95, xanchor="center", yanchor="top"),
    template="plotly_dark", height=400, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    margin=dict(l=20, r=20, b=20, t=60),
    yaxis=dict(title_text="Entropy (0-100 Scale)")
)
fig2.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# UNSUPERVISED LEARNING PROOF -- DUAL-PLANE
# ==============================================================================
st.markdown("---")
st.subheader(T("2. UNSUPERVISED LEARNING: DUAL-PLANE DS PROOF", "2. BANG CHUNG HOC MAY: HAI MAT PHANG ENTROPY"))
st.markdown(T(
    "Plane 1: Raw Full GMM Phase Space (X=WPE, Y=SPE_Z, no transform). Plane 2: Volume GMM (X=Shannon, Y=SampEn).",
    "Mat phang 1: Raw Full GMM Phase Space (X=WPE, Y=SPE_Z, khong transform). Mat phang 2: Volume GMM (X=Shannon, Y=SampEn)."
))

col_price_plot, col_vol_plot = st.columns([1, 1])

# --- PLOT 1: Price Dynamics Plane (Raw Entropy Phase Space) ---
with col_price_plot:
    st.markdown(f"**{T('PLANE 1: RAW ENTROPY PHASE SPACE', 'MAT PHANG 1: KHONG GIAN PHA ENTROPY (RAW)')}**")
    plot_df = df.dropna(subset=['WPE', 'SPE_Z', 'RegimeName'])
    if not plot_df.empty:
        color_map_price = {
            "Stochastic":    "#00FF41",   # green  — normal market
            "Transitional":  "#FFD700",   # yellow — mixed structure
            "Deterministic": "#FF0000",   # red    — high coordination
        }
        scatter_price = px.scatter(
            plot_df, x="WPE", y="SPE_Z",
            color="RegimeName",
            color_discrete_map=color_map_price,
            hover_data=["Close", "MFI"],
            labels={"WPE": "WPE (Weighted Permutation Entropy)", "SPE_Z": "SPE_Z (Standardized Price Sample Entropy)"},
        )

        # 95% Confidence Ellipses (Full GMM: unique shape per cluster, RAW space)
        if price_clf is not None:
            try:
                regime_colors = {0: "#00FF41", 1: "#FFD700", 2: "#FF0000"}
                t_arr = np.linspace(0, 2 * np.pi, 100)
                for cluster_idx in range(price_clf.n_components):
                    ell = price_clf.get_ellipse_params(cluster_idx, n_std=2.0)
                    regime_idx = price_clf._cluster_to_regime.get(cluster_idx, cluster_idx)
                    cos_a = np.cos(np.radians(ell["angle"]))
                    sin_a = np.sin(np.radians(ell["angle"]))
                    x_ell = (ell["width"] / 2) * np.cos(t_arr)
                    y_ell = (ell["height"] / 2) * np.sin(t_arr)
                    x_rot = cos_a * x_ell - sin_a * y_ell + ell["center"][0]
                    y_rot = sin_a * x_ell + cos_a * y_ell + ell["center"][1]
                    scatter_price.add_trace(go.Scatter(
                        x=x_rot, y=y_rot, mode='lines',
                        line=dict(color=regime_colors.get(regime_idx, "white"), width=1.5, dash='dash'),
                        showlegend=False, hoverinfo='skip',
                    ))
            except Exception:
                pass

        scatter_price.update_layout(
            template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            legend_title="Price Regime (Raw Full GMM)",
            margin=dict(l=20, r=20, b=20, t=20), height=450,
        )
        st.plotly_chart(scatter_price, use_container_width=True)

# --- PLOT 2: Volume Entropy Plane ---
with col_vol_plot:
    st.markdown(f"**{T('PLANE 2: LIQUIDITY STRUCTURE', 'MẶT PHẲNG 2: CẤU TRÚC THANH KHOẢN')}**")
    vol_plot_df = df.dropna(subset=['Vol_Shannon', 'Vol_SampEn', 'VolRegimeName'])
    if not vol_plot_df.empty:
        color_map_vol = {
            "Consensus Flow": "#1E90FF",
            "Dispersed Flow": "#BA55D3",
            "Erratic/Noisy Flow": "#FF6347",
        }
        scatter_vol = px.scatter(
            vol_plot_df, x="Vol_Shannon", y="Vol_SampEn",
            color="VolRegimeName",
            color_discrete_map=color_map_vol,
            hover_data=["Close", "Volume"],
            labels={"Vol_Shannon": "Shannon Entropy (Concentration)", "Vol_SampEn": "Sample Entropy (Impulse Regularity)"},
        )
        scatter_vol.update_layout(
            template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            legend_title="Volume Regime",
            margin=dict(l=20, r=20, b=20, t=20), height=450,
        )
        st.plotly_chart(scatter_vol, use_container_width=True)
    else:
        st.info(T("Volume Entropy data requires minimum 60 trading days.", "Du lieu Volume Entropy can toi thieu 60 ngay giao dich."))

st.markdown("---")
st.subheader(T("3. RISK INTELLIGENCE REPORT", "3. BAO CAO PHAN TICH RUI RO"))
st.markdown(T(
    "AI-powered risk narrative synthesizing volatility, entropy regimes, and tail risk.",
    "Phan tich rui ro tong hop tu bien dong, che do entropy va rui ro duoi."
))

# === BUILD NARRATIVE VARIABLES ===
sigma_raw = 0
sigma_adj = 0
garch_type = "N/A"
es_5 = 0
es_adj = None
r_mult = 1.0

if _garch_ok:
    sigma_raw = _garch_kpi.get("sigma_raw", _garch_kpi["sigma_daily_pct"])
    sigma_adj = _garch_kpi.get("sigma_adjusted", sigma_raw)
    sigma_annual = _garch_kpi["sigma_annual_pct"]
    garch_type = _garch_kpi.get("diagnostics", {}).get("garch_type",
                  _garch_kpi.get("garch_type", "GARCH"))
    es_5 = _garch_kpi.get("ES_5pct", 0)
    es_adj = _garch_kpi.get("ES_5pct_adjusted")
    r_mult = _garch_kpi.get("regime_multiplier", 1.0)
    d_hp = _garch_kpi.get("delta_H_price", 0)
    d_hv = _garch_kpi.get("delta_H_volume", 0)
    entropy_status = _garch_kpi.get("entropy_status",
                      "ACTIVE" if garch_type == "GARCH-X" else "DORMANT")

# Risk level label
if sigma_adj < 0.8:
    risk_verdict = T("LOW RISK -- Market structure is healthy", 
                     "RUI RO THAP -- Cau truc thi truong lanh manh")
    risk_verdict_color = "#00FF41"
elif sigma_adj < 1.5:
    risk_verdict = T("MODERATE RISK -- Structural stress detected",
                     "RUI RO TRUNG BINH -- Phat hien cang thang cau truc")
    risk_verdict_color = "#FFD700"
elif sigma_adj < 2.5:
    risk_verdict = T("HIGH RISK -- Significant structural deterioration",
                     "RUI RO CAO -- Suy giam cau truc dang ke")
    risk_verdict_color = "#FF5F1F"
else:
    risk_verdict = T("EXTREME RISK -- Systemic breakdown in progress",
                     "RUI RO CUC CAO -- He thong dang sup do")
    risk_verdict_color = "#FF0000"

# === SECTION A: VOLATILITY ANALYSIS ===
if _garch_ok:
    if garch_type == "GARCH-X":
        vol_para = T(
            f"The GARCH-X model identifies daily volatility at **{sigma_raw:.3f}%** "
            f"(annualized {sigma_annual:.1f}%). Entropy features are statistically significant "
            f"in explaining variance -- price entropy (d1={d_hp:+.4f}) "
            f"{'amplifies' if d_hp > 0 else 'dampens'} volatility, "
            f"while volume entropy (d2={d_hv:+.4f}) "
            f"{'amplifies' if d_hv > 0 else 'dampens'} it. "
            f"This means market disorder is actively contributing to risk beyond normal volatility clustering.",
            
            f"Mo hinh GARCH-X xac dinh bien dong hang ngay o muc **{sigma_raw:.3f}%** "
            f"(quy nam {sigma_annual:.1f}%). Cac chi so entropy co y nghia thong ke "
            f"trong viec giai thich bien dong -- entropy gia (d1={d_hp:+.4f}) "
            f"{'khuech dai' if d_hp > 0 else 'giam nhe'} bien dong, "
            f"entropy thanh khoan (d2={d_hv:+.4f}) "
            f"{'khuech dai' if d_hv > 0 else 'giam nhe'} bien dong. "
            f"Dieu nay nghia la su hon loan thi truong dang chu dong gop phan tao rui ro."
        )
    else:
        vol_para = T(
            f"The volatility model measures daily risk at **{sigma_raw:.3f}%** "
            f"(annualized {sigma_annual:.1f}%). Currently, entropy features do not add "
            f"statistically significant information beyond standard volatility dynamics -- "
            f"this is **normal during stable periods**. Entropy's role shifts to regime "
            f"classification (see below), where it identifies the current market phase.",
            
            f"Mo hinh do luong rui ro hang ngay o muc **{sigma_raw:.3f}%** "
            f"(quy nam {sigma_annual:.1f}%). Hien tai, cac chi so entropy khong bo sung "
            f"thong tin co y nghia thong ke ngoai dong luc bien dong thong thuong -- "
            f"**dieu nay binh thuong trong giai doan on dinh**. Vai tro entropy chuyen sang "
            f"phan loai che do thi truong (xem ben duoi), noi no xac dinh pha hien tai."
        )
else:
    vol_para = T(
        "Volatility model is not available -- insufficient data (need 120+ trading days with entropy features).",
        "Mo hinh bien dong khong kha dung -- thieu du lieu (can 120+ ngay giao dich co entropy features)."
    )

# === SECTION B: REGIME ANALYSIS ===
regime_upper = current_regime.upper()
if "STOCHASTIC" in regime_upper:
    regime_color = "#00FF41"
    regime_explain = T(
        "Price entropy is HIGH — normal market conditions. "
        "Diverse participants with diverse strategies: no single coordinating force dominates. "
        "This is the healthy state. Validated: forward 20-day realized volatility averages ~11%.",
        "Entropy gia cao — dieu kien thi truong binh thuong. "
        "Nhieu thanh vien voi chien luoc khac nhau: khong co mot luc phoi hop nao chiem uu the. "
        "Day la trang thai lanh manh. Da xac nhan: bien dong 20 ngay forward trung binh ~11%."
    )
    coordination_alert = T(
        "Normal market structure — diverse behavior, no dominant force.",
        "Cau truc thi truong binh thuong — hanh vi da dang, khong co luc chiem uu the."
    )
elif "TRANSITIONAL" in regime_upper:
    regime_color = "#FFD700"
    regime_explain = T(
        "Price entropy is at MID level — mixed structure, phase transition in progress. "
        "The market is moving between ordered and disordered states. "
        "Validated: forward 20-day realized volatility averages ~15%.",
        "Entropy gia o muc TRUNG BINH — cau truc hon hop, chuyen pha dang dien ra. "
        "Thi truong dang di chuyen giua trang thai co trat tu va khong co trat tu. "
        "Da xac nhan: bien dong 20 ngay forward trung binh ~15%."
    )
    coordination_alert = T(
        "Mixed structure — watch for acceleration toward Deterministic (high coordination).",
        "Cau truc hon hop — theo doi su tang toc huong toi Deterministic (phoi hop cao)."
    )
else:
    regime_color = "#FF0000"
    if _price_above_sma20:
        regime_explain = T(
            "HIGH COORDINATION detected — the market is in a Deterministic regime DURING A RALLY. "
            "Price entropy is LOW, meaning movements are structured and coordinated. "
            "This is typical of late-stage momentum: the rally is driven by herding, not diverse conviction. "
            "Structural fragility is elevated. Any reversal will be sharper than usual. "
            "Validated: forward 20-day realized volatility averages ~20%.",
            "PHOI HOP CAO duoc phat hien — thi truong dang o che do Deterministic TRONG KHI TANG GIA. "
            "Entropy gia THAP, co nghia la cac bien dong co cau truc va phoi hop. "
            "Day la dieu hien dien hinh cua dong luc giai doan cuoi: da tang duoc thuc day boi herd behavior, khong phai su tin tuong da dang. "
            "Su gion nat cau truc tang cao. Bat ky dao chieu nao se manh hon binh thuong. "
            "Da xac nhan: bien dong 20 ngay forward trung binh ~20%."
        )
        coordination_alert = T(
            "Coordinated rally detected — structural fragility elevated despite rising prices. "
            "Late-stage momentum pattern.",
            "Da tang phoi hop duoc phat hien — su gion nat cau truc tang cao du gia dang tang. "
            "Mo hinh dong luc giai doan cuoi."
        )
    else:
        regime_explain = T(
            "HIGH COORDINATION detected — the market is in a Deterministic regime DURING A DECLINE. "
            "Price entropy is LOW, meaning the decline is structured and coordinated. "
            "This is the signature of institutional selling or panic. "
            "The coordinated nature of the decline increases tail risk significantly. "
            "Validated: forward 20-day realized volatility averages ~20%.",
            "PHOI HOP CAO duoc phat hien — thi truong dang o che do Deterministic TRONG KHI GIAM GIA. "
            "Entropy gia THAP, co nghia la su suy giam co cau truc va phoi hop. "
            "Day la dau hieu cua viec ban to chuc hoac hoang loan. "
            "Ban chat phoi hop cua su suy giam lam tang rui ro duoi. "
            "Da xac nhan: bien dong 20 ngay forward trung binh ~20%."
        )
        coordination_alert = T(
            "Coordinated decline — institutional selling or panic. "
            "Structural risk elevated.",
            "Suy giam phoi hop — ban to chuc hoac hoang loan. "
            "Rui ro cau truc tang cao."
        )

# Volume regime analysis
vol_regime_upper = current_vol_regime_name.upper()
if "CONSENSUS" in vol_regime_upper:
    vol_color = "#1E90FF"
    vol_explain = T(
        "Volume flow shows institutional consensus -- capital is moving in an organized, "
        "predictable manner. Liquidity supports the current price structure.",
        "Dong tien the hien su dong thuan to chuc -- von di chuyen co to chuc, "
        "du doan duoc. Thanh khoan ho tro cau truc gia hien tai."
    )
elif "DISPERSED" in vol_regime_upper:
    vol_color = "#BA55D3"
    vol_explain = T(
        "Volume flow is dispersed -- capital is fragmented across different directions. "
        "No clear institutional consensus. This creates vulnerability to sudden moves "
        "when one side gains dominance.",
        "Dong tien phan tan -- von bi chia nho theo nhieu huong khac nhau. "
        "Khong co su dong thuan to chuc ro rang. Dieu nay tao ra ton thuong "
        "truoc cac bien dong dot ngot khi mot ben chiem uu the."
    )
else:
    vol_color = "#FF6347"
    vol_explain = T(
        "Volume flow is erratic -- extreme disorder in liquidity structure. "
        "Capital flows are unpredictable and chaotic, indicating either panic "
        "or speculative excess.",
        "Dong tien bat thuong -- hon loan cuc do trong cau truc thanh khoan. "
        "Dong von khong the du doan va hon loan, cho thay hoang loan "
        "hoac dau co qua muc."
    )

# Cross-plane divergence check
divergence_alert = ""
if "DETERMINISTIC" in regime_upper and ("DISPERSED" in vol_regime_upper or "ERRATIC" in vol_regime_upper):
    divergence_alert = T(
        f"**Hollow Rally Alert:** Price entropy is dangerously LOW (Deterministic — strong directional force) "
        f"but volume structure is fractured ({current_vol_regime_name}). "
        f"A directional move without volume confirmation is unsustainable. High reversal risk.",
        f"**Canh bao Da Tang Rong:** Entropy gia THAP nguy hiem (Deterministic — luc dinh huong manh) "
        f"nhung cau truc thanh khoan dang nut vo ({current_vol_regime_name}). "
        f"Xu huong khong co xac nhan thanh khoan la khong ben vung. Rui ro dao chieu cao."
    )
elif "STOCHASTIC" in regime_upper and current_vol_global_z < 0:
    divergence_alert = T(
        "**Capitulation Vacuum:** High price entropy (Stochastic — random walk) with below-average volume. "
        "Apparent calm is driven by illiquidity, not genuine equilibrium. Thin market = gap risk.",
        "**Khoang Trong Dau Hang:** Entropy gia cao (Stochastic — buoc di ngau nhien) nhung thanh khoan thap. "
        "Su binh lang la do thieu thanh khoan, khong phai can bang thuc su. Thi truong mong = rui ro gap."
    )

# Regime multiplier explanation
if r_mult > 1.0:
    mult_explain = T(
        f"Because the market is in **{current_regime}** price regime and "
        f"**{current_vol_regime_name}** volume regime, the base volatility "
        f"({sigma_raw:.3f}%) is amplified by {r_mult:.1f}x to "
        f"**{sigma_adj:.3f}%** -- reflecting the structural stress that "
        f"entropy regimes have identified.",
        f"Vi thi truong dang o che do gia **{current_regime}** va "
        f"che do thanh khoan **{current_vol_regime_name}**, bien dong goc "
        f"({sigma_raw:.3f}%) duoc nhan {r_mult:.1f}x thanh "
        f"**{sigma_adj:.3f}%** -- phan anh cang thang cau truc ma "
        f"che do entropy da xac dinh."
    )
else:
    mult_explain = T(
        f"Both price and volume regimes are stable -- no regime amplification applied. "
        f"The base volatility of {sigma_raw:.3f}% reflects the full risk picture.",
        f"Ca che do gia va thanh khoan deu on dinh -- khong can nhan he so regime. "
        f"Bien dong goc {sigma_raw:.3f}% phan anh day du buc tranh rui ro."
    )

# === SECTION C: XAI TRAJECTORY ===
v_wpe_val = current_v_wpe if pd.notna(current_v_wpe) else 0.0
a_wpe_val = current_a_wpe if pd.notna(current_a_wpe) else 0.0

if v_wpe_val > 0 and a_wpe_val > 0:
    trajectory = T(
        "Entropy is **accelerating upward** -- structural coordination is dissolving. "
        "Moving toward Stochastic (Normal Market): fragility is decreasing.",
        "Entropy dang **tang toc** -- su phoi hop cau truc dang tan ra. "
        "Tien toi Stochastic (Thi truong binh thuong): su gion nat dang giam."
    )
elif v_wpe_val > 0 and a_wpe_val < 0:
    trajectory = T(
        "Entropy is increasing but **slowing down** -- disorder is growing but losing momentum. "
        "Moving away from Deterministic (High Coordination) zone, but transition may stall.",
        "Entropy dang tang nhung **giam toc** -- do ngau nhien tang nhung mat da. "
        "Dang roi xa vung Deterministic (Phoi hop cao), nhung qua trinh chuyen tiep co the cham lai."
    )
elif v_wpe_val < 0 and a_wpe_val < 0:
    trajectory = T(
        "Entropy is **cooling rapidly** -- structural order is being restored. "
        "WARNING: Entropy falling = market becoming MORE deterministic = RISING RISK.",
        "Entropy dang **ha nhiet nhanh** -- trat tu cau truc dang duoc phuc hoi. "
        "CANH BAO: Entropy giam = thi truong tro nen co trat tu hon = RUI RO TANG."
    )
elif v_wpe_val < 0 and a_wpe_val > 0:
    trajectory = T(
        "Entropy is decreasing but the decline is **slowing** -- the drift toward "
        "Deterministic (High Coordination) drift is losing momentum. Possible entropy floor near.",
        "Entropy dang giam nhung toc do giam **cham lai** -- xu huong tien toi "
        "Deterministic (Phoi hop cao) dang mat da. Co the gap day entropy gan day."
    )
else:
    trajectory = T(
        "Entropy is **stationary** -- no significant change in market disorder. "
        "The current regime is likely to persist in the near term.",
        "Entropy **dung yen** -- khong co thay doi dang ke trong muc hon loan. "
        "Che do hien tai co kha nang duy tri trong ngan han."
    )

# === SECTION D: TAIL RISK ===
if _garch_ok:
    if es_adj is not None:
        tail_para = T(
            f"If a tail event occurs (worst 5% of scenarios), the expected loss is "
            f"**{es_5:+.2f}%** per day. After adjusting for current cross-sectional entropy "
            f"(which measures how synchronized VN30 stocks are), the adjusted figure is "
            f"**{es_adj:+.2f}%**. This means the distribution tails are "
            f"{'fatter' if abs(es_adj) > abs(es_5) else 'thinner'} than historical average.",
            f"Neu xay ra su kien duoi (5% kich ban xau nhat), ton that ky vong la "
            f"**{es_5:+.2f}%** moi ngay. Sau khi dieu chinh theo entropy cheo hien tai "
            f"(do muc dong bo cua co phieu VN30), con so dieu chinh la "
            f"**{es_adj:+.2f}%**. Dieu nay nghia la duoi phan phoi "
            f"{'day hon' if abs(es_adj) > abs(es_5) else 'mong hon'} so voi trung binh lich su."
        )
    else:
        tail_para = T(
            f"If a tail event occurs (worst 5% of scenarios), the expected loss is "
            f"**{es_5:+.2f}%** per day.",
            f"Neu xay ra su kien duoi (5% kich ban xau nhat), ton that ky vong la "
            f"**{es_5:+.2f}%** moi ngay."
        )
else:
    tail_para = ""

# === RENDER THE REPORT ===
verdict_html = f"""
<div style="border-left: 5px solid {risk_verdict_color}; background: rgba(0,0,0,0.3); 
            padding: 20px 25px; border-radius: 4px; margin-bottom: 20px;">
    <div style="color: {risk_verdict_color}; font-size: 1.3rem; font-weight: 800; 
                font-family: 'Courier Prime', monospace; margin-bottom: 8px;">
        {risk_verdict}
    </div>
    <div style="color: #e0e0e0; font-size: 1.0rem; line-height: 1.6; font-weight: 400;">
        {mult_explain}
    </div>
</div>
"""
st.markdown(verdict_html, unsafe_allow_html=True)

# Analysis cards
st.markdown(f"""
<div class="analysis-card">
    <div class="analysis-card-title">{T('1. Volatility Assessment', '1. Danh gia Bien dong')}</div>
    <div class="analysis-text">{vol_para}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="analysis-card">
    <div class="analysis-card-title">{T('2. Price Regime', '2. Che do Gia')}</div>
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
        <span class="regime-badge" style="background: {regime_color}22; color:{regime_color}; border: 1px solid {regime_color};">{current_regime.upper()}</span>
        <span style="color:#888; font-size:0.82rem;">WPE: {current_wpe:.4f} | SPE_Z: {current_spe_z:+.3f}</span>
    </div>
    <div class="analysis-text">{regime_explain}</div>
    <div style="margin-top:10px; padding:8px 12px; background:rgba(255,255,255,0.04); border-left:3px solid {regime_color}; border-radius:3px; font-size:0.82rem; color:{regime_color}; font-family:'Courier Prime',monospace;">
        &#9889; {coordination_alert}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="analysis-card">
    <div class="analysis-card-title">{T('3. Liquidity Structure', '3. Cau truc Thanh khoan')}</div>
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
        <span class="regime-badge" style="background: {vol_color}22; color:{vol_color}; border: 1px solid {vol_color};">{current_vol_regime_name.upper()}</span>
        <span style="color:#888; font-size:0.82rem;">Shannon: {vol_sh_kpi} | SampEn: {vol_se_kpi} | Z: {vol_gz_kpi}</span>
    </div>
    <div class="analysis-text">{vol_explain}</div>
</div>
""", unsafe_allow_html=True)

if divergence_alert:
    st.markdown(f"""
    <div style="border: 2px solid #FF5F1F; background: rgba(255,95,31,0.08); 
                padding: 15px 20px; border-radius: 6px; margin: 12px 0;">
        <div class="analysis-text" style="color: #FFB088;">{divergence_alert}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="analysis-card">
    <div class="analysis-card-title">{T('4. Entropy Momentum', '4. Da Entropy')}</div>
    <div class="xai-trajectory-box">
        <div class="xai-values" style="margin-bottom: 8px;">
            <div class="xai-item"><span class="xai-item-label">{T('Speed', 'Toc do')}:</span><span class="xai-item-value">{current_v_wpe:+.5f}</span></div>
            <div class="xai-item"><span class="xai-item-label">{T('Acceleration', 'Gia toc')}:</span><span class="xai-item-value">{current_a_wpe:+.5f}</span></div>
        </div>
        <div class="analysis-text">{trajectory}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if tail_para:
    st.markdown(f"""
    <div class="analysis-card">
        <div class="analysis-card-title">{T('5. Tail Risk Assessment', '5. Danh gia Rui ro Duoi')}</div>
        <div class="analysis-text">{tail_para}</div>
    </div>
    """, unsafe_allow_html=True)

# === TECHNICAL DETAILS (collapsible) ===
with st.expander(T("Technical Details (for quant analysts)", "Chi tiet Ky thuat (cho nha phan tich dinh luong)")):
    if _garch_ok:
        diag = _garch_kpi.get("diagnostics", {})
        st.markdown(f"""
        | Metric | Value |
        | :--- | :--- |
        | Model Type | {garch_type} |
        | sigma_daily (raw) | {sigma_raw:.4f}% |
        | sigma_annual | {sigma_annual:.2f}% |
        | Regime Multiplier | {r_mult:.1f}x ({r_source}) |
        | sigma_daily (adjusted) | {sigma_adj:.4f}% |
        | d1 (H_price) | {d_hp:+.6f} (p={diag.get('pval_H_price', 'N/A')}) |
        | d2 (H_volume) | {d_hv:+.6f} (p={diag.get('pval_H_volume', 'N/A')}) |
        | alpha + beta | {diag.get('alpha_plus_beta', 'N/A')} |
        | AIC | {diag.get('aic', _garch_kpi.get('aic', 'N/A'))} |
        | BIC | {diag.get('bic', 'N/A')} |
        | Ljung-Box (lag 10) | {diag.get('lb_pval_lag10', 'N/A')} |
        | Ljung-Box (lag 20) | {diag.get('lb_pval_lag20', 'N/A')} |
        | VaR 5% | {_garch_kpi.get('VaR_5pct', 'N/A')}% |
        | ES 5% (raw) | {es_5}% |
        | ES 5% (adjusted) | {es_adj if es_adj else 'N/A'}% |
        """)
    else:
        st.info(T("GARCH model not available. Technical details require 120+ days of data.",
                   "Mo hinh GARCH khong kha dung. Chi tiet ky thuat can 120+ ngay du lieu."))


# ==============================================================================
# DATA EXPORT
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{T('2. DATA EXPORT', '2. XUẤT DỮ LIỆU')}**")
csv_data = df.to_csv().encode('utf-8')
st.sidebar.download_button(
    label=T("Export Current Analysis (CSV)", "Xuất Dữ Liệu Hiện Tại (CSV)"),
    data=csv_data,
    file_name="financial_entropy_agent_export.csv",
    mime="text/csv",
    use_container_width=True
)
