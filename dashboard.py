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
from agent_orchestrator import calc_composite_risk_score

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
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# UI INITIALIZATION & MULTILINGUAL SUPPORT
# ==============================================================================
risk_score = 0.0
synthesis_label = "STABLE"
risk_color = "#00FF41"
current_wpe = 0.5
current_regime = "STABLE"
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
    "Tri-Vector Composite Risk Surveillance: Price Entropy, Liquidity Structure, VN30 Breadth.",
    "Giam sat Rui ro Hop thanh Tri-Vector: Entropy Gia, Cau truc Thanh khoan, Do rong VN30."
))

with st.spinner(T("Computing Tri-Vector Composite Engine...", "Đang tính toán Tri-Vector Composite Engine...")):
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
# TOP KPI SECTION: TRI-VECTOR ALIGNED
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)

# --- Column 1: Systemic Risk Gauge + Dominant Force ---
with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': risk_color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, elevated_bound], 'color': 'rgba(0, 255, 65, 0.1)'},
                {'range': [elevated_bound, critical_bound], 'color': 'rgba(255, 215, 0, 0.1)'},
                {'range': [critical_bound, 100], 'color': 'rgba(255, 0, 0, 0.1)'}],
        }
    ))
    
    fig_gauge.update_layout(
        autosize=True,
        height=170, 
        margin=dict(l=20, r=20, t=10, b=10), 
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=f"{risk_score:.1f}",
            x=0.5, y=0.25,
            xref="paper", yref="paper",
            font=dict(size=40, color="#FFFFFF", family="Courier Prime"),
            showarrow=False,
            xanchor="center",
            yanchor="middle"
        )]
    )
    
    st.markdown(f'<div class="metric-label" style="text-align:center; padding-top:5px;">{T("COMPOSITE RISK SCORE", "DIEM RUI RO HOP THANH")}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    st.markdown(f"<div style='text-align:center; color:{risk_color}; font-weight:800; font-size:1.0rem; margin-top:-15px;'>{synthesis_label}</div>", unsafe_allow_html=True)
    # Dominant Force sub-text
    dominant_display = dominant_vector.replace("V1_Price", "V1: PRICE").replace("V2_Volume", "V2: VOLUME").replace("V3_Breadth", "V3: BREADTH")
    dominant_pct = vector_info.get('contribution_percentages', {}).get(dominant_vector, 0)
    st.markdown(f"<div style='text-align:center; font-size:0.75rem; color:#888; margin-top:4px; font-family:Courier Prime, monospace;'>Dominant: <span style=\"color:#FFD700;\">{dominant_display}</span> ({dominant_pct:.0f}% XAI)</div>", unsafe_allow_html=True)

# --- Column 2: V1 Price Phase Space ---
with col2:
    p1_color = "#00FF41" if "STABLE" in current_regime.upper() else ("#FFD700" if "FRAGILE" in current_regime.upper() else "#FF3131")
    v1_risk = contributions.get('V1_Price', 0) * 100
    spe_z_display = f"{current_spe_z:+.2f}" if pd.notna(current_spe_z) else "N/A"
    st.markdown(f"""
    <div class="arch-badge" style="height: 210px; display: flex; flex-direction: column; justify-content: center;">
        <div class="metric-label">{T("V1: PRICE PHASE SPACE", "V1: KHONG GIAN PHA GIA")}</div>
        <div class="metric-value" style="color: {p1_color}; font-size: 1.4rem;">{current_regime}</div>
        <div style="font-size: 0.8rem; color: #888; margin-top: 10px;">WPE: {current_wpe:.4f} | SPE_Z: {spe_z_display}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Column 3: V2 Liquidity Structure ---
with col3:
    vol_regime_upper = current_vol_regime_name.upper()
    p2_color = "#00FF41" if "CONSENSUS" in vol_regime_upper else ("#FFD700" if "DISPERSED" in vol_regime_upper else "#FF3131")
    v2_risk = contributions.get('V2_Volume', 0) * 100
    st.markdown(f"""
    <div class="arch-badge" style="height: 210px; display: flex; flex-direction: column; justify-content: center;">
        <div class="metric-label">{T("V2: LIQUIDITY STRUCTURE", "V2: CAU TRUC THANH KHOAN")}</div>
        <div class="metric-value" style="color: {p2_color}; font-size: 1.1rem;">{current_vol_regime_name}</div>
        <div style="font-size: 0.8rem; color: #888; margin-top: 10px;">Shannon: {vol_sh_kpi} | SampEn: {vol_se_kpi}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Column 4: V3 VN30 Breadth ---
with col4:
    breadth_label = "COHESIVE" if current_cse < 40 else ("DISLOCATED" if current_cse > 70 else "FRAGMENTING")
    p3_color = "#00FF41" if breadth_label == "COHESIVE" else ("#FFD700" if breadth_label == "FRAGMENTING" else "#FF3131")
    v3_risk = contributions.get('V3_Breadth', 0) * 100
    cse_display = f"{current_cse:.1f}%"
    mfi_display = f"{current_mfi:.4f}"
    st.markdown(f"""
    <div class="arch-badge" style="height: 210px; display: flex; flex-direction: column; justify-content: center;">
        <div class="metric-label">{T("V3: VN30 BREADTH", "V3: DO RONG VN30")}</div>
        <div class="metric-value" style="color: {p3_color}; font-size: 1.4rem;">{breadth_label}</div>
        <div style="font-size: 0.8rem; color: #888; margin-top: 10px;">Corr Entropy: {cse_display} | MFI: {mfi_display}</div>
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
    "Stable": "rgba(0, 255, 65, 0.15)",
    "Fragile": "rgba(255, 215, 0, 0.15)",
    "Chaos": "rgba(255, 0, 0, 0.15)",
    "Calculating...": "rgba(128, 128, 128, 0)"
}

# Dummy legend traces
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(0, 255, 65, 1)'), name='Stable'))
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(255, 215, 0, 1)'), name='Fragile'))
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(255, 0, 0, 1)'), name='Chaos'))

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
            "Stable": "#00FF41",
            "Fragile": "#FFD700",
            "Chaos": "#FF0000",
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

# ==============================================================================
# AGENT DIAGNOSTIC -- TRI-VECTOR COMPOSITE SYNTHESIS
# ==============================================================================
st.markdown("---")
st.subheader(T("3. TRI-VECTOR COMPOSITE RISK DIAGNOSTIC", "3. CHẨN ĐOÁN RỦI RO HỢP THÀNH TRI-VECTOR"))
st.markdown(T(
    "Integrated composite diagnostic: 0.4*V1(Price) + 0.4*V2(Volume) + 0.2*V3(VN30 Breadth) scaled via Z-score + Sigmoid.",
    "Chẩn đoán hợp thành: 0.4*V1(Giá) + 0.4*V2(Thanh khoản) + 0.2*V3(Độ rộng VN30) chuẩn hóa Z-score + Sigmoid."
))

# Extract values for display
current_cse_norm = current_cse / 100.0 if pd.notna(current_cse) else 0.5
agent_regime = str(latest.get("RegimeName", "Calculating...")).upper()
current_vol_regime_upper = str(latest.get("VolRegimeName", "Calculating...")).upper()

# Structural Breadth analysis
if current_cse > 60:
    vn30_analysis = "High VN30 Entropy: Blue chips exhibit extreme fragmentation with no market consensus. Capital dispersion across sectors signals systemic decorrelation."
elif current_cse < 40:
    vn30_analysis = "Low VN30 Entropy: Blue chips maintain deterministic consensus. Centralized capital flow indicates institutional structural integrity."
else:
    vn30_analysis = "Neutral VN30 Entropy: Moderate internal rotation among capital pillars. Sector rebalancing underway without systemic stress."

vn30_breadth_status = "Unified" if current_cse < 40 else ("Divergent" if current_cse > 70 else "Fragmenting")

# Formatted code values
wpe_formatted = f"<code>{current_wpe:.4f}</code>"
spe_z_formatted = f"<code>{current_spe_z:+.3f}</code>" if pd.notna(current_spe_z) else "<code>N/A</code>"
v_wpe_formatted = f"<code>{current_v_wpe:+.5f}</code>" if pd.notna(current_v_wpe) else "<code>N/A</code>"
a_wpe_formatted = f"<code>{current_a_wpe:+.5f}</code>" if pd.notna(current_a_wpe) else "<code>N/A</code>"
gz_formatted = f"<code>{current_vol_global_z:+.2f} Z</code>" if pd.notna(current_vol_global_z) else "<code>N/A</code>"
sh_formatted = f"<code>{vol_sh_kpi}</code>"
se_formatted = f"<code>{vol_se_kpi}</code>"
cse_formatted = f"<code>{current_cse_norm:.2f}</code>"
mfi_formatted = f"<code>{current_mfi:.4f}</code>"
risk_formatted = f"<code>{risk_score:.1f}</code>"

status_strong = f"<strong>{synthesis_label}</strong>"
breadth_strong = f"<strong>{vn30_breadth_status}</strong>"
regime_strong = f"<strong>{current_regime}</strong>"
vol_regime_strong = f"<strong>{current_vol_regime_upper}</strong>"
dominant_strong = f"<strong>{dominant_vector.replace('V1_Price','V1: PRICE').replace('V2_Volume','V2: VOLUME').replace('V3_Breadth','V3: BREADTH')}</strong>"

# Critical alert block
critical_block = f"""
<div class="critical-alert" style="border: 2px solid #FF3131; background: rgba(255, 49, 49, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
    <strong>[CRITICAL ALERT]</strong><br>
    Composite Risk Score ({risk_score:.1f}) exceeds P90 dynamic threshold ({critical_bound:.1f}). Phase transition imminent. 
    Dominant risk vector: {dominant_strong}. Immediate risk hedging recommended.
</div>
""" if "CRITICAL" in synthesis_label else ""

# Protocol logs
protocol_logs = f"""
> INITIATING TRI-VECTOR COMPOSITE ENGINE...<br>
> FITTING RAW ENTROPY PHASE SPACE (Full GMM, NO transform)...<br>
> EXTRACTING VECTORS [V1: PRICE, V2: VOLUME, V3: BREADTH]...<br>
> NORMALIZING VIA PowerTransformer(yeo-johnson) + MinMaxScaler PIPELINE (RISK ENGINE ONLY)...<br>
> DYNAMIC RISK THRESHOLDS: ELEVATED = P75({elevated_bound:.1f}), CRITICAL = P90({critical_bound:.1f})<br>
> WEIGHTED SYNTHESIS COMPLETE (40/40/20 MODEL).<br>
"""

# XAI Trajectory analysis
v_wpe_val = current_v_wpe if pd.notna(current_v_wpe) else 0.0
a_wpe_val = current_a_wpe if pd.notna(current_a_wpe) else 0.0
if v_wpe_val > 0 and a_wpe_val > 0:
    trajectory_analysis = "WPE is accelerating upward -- the system is rapidly approaching higher entropy (toward Chaos)."
elif v_wpe_val > 0 and a_wpe_val < 0:
    trajectory_analysis = "WPE is increasing but decelerating -- entropy growth is slowing, possible stabilization ahead."
elif v_wpe_val < 0 and a_wpe_val < 0:
    trajectory_analysis = "WPE is accelerating downward -- the system is rapidly cooling, structural order is being restored."
elif v_wpe_val < 0 and a_wpe_val > 0:
    trajectory_analysis = "WPE is decreasing but deceleration in the decline -- entropy may bottom out soon."
else:
    trajectory_analysis = "Entropy trajectory is near stationary. No significant regime transition in progress."

# Liquidity synthesis
gz_val_safe = current_vol_global_z if pd.notna(current_vol_global_z) else 0.0
vol_is_consensus = "CONSENSUS" in current_vol_regime_upper
if (gz_val_safe > 0 and latest['Close'] > df['SMA20'].iloc[-1]) or (gz_val_safe < 0 and latest['Close'] < df['SMA20'].iloc[-1]):
    liquidity_direction = "supporting"
else:
    liquidity_direction = "diverging from"

agent_log = f"""
<div class="agent-log">
{protocol_logs}
<br>
<h3>[ {T("TRI-VECTOR COMPOSITE RISK DIAGNOSTIC", "CHAN DOAN RUI RO HOP THANH TRI-VECTOR")} ]</h3>

| Vector Module | Key Metrics | Scaled Value | Weight | XAI % |
| :--- | :--- | :--- | :--- | :--- |
| **V1: Price Phase Space** | WPE: {wpe_formatted} -- SPE_Z: {spe_z_formatted} | <code>{contributions.get('V1_Price',0):.4f}</code> | **40%** | **{vector_info.get('contribution_percentages', {}).get('V1_Price', 0):.1f}%** |
| **V2: Volume Entropy** | SampEn: {se_formatted} -- Macro Z: {gz_formatted} -- Shannon: {sh_formatted} | <code>{contributions.get('V2_Volume',0):.4f}</code> | **40%** | **{vector_info.get('contribution_percentages', {}).get('V2_Volume', 0):.1f}%** |
| **V3: VN30 Breadth** | Corr Entropy: {cse_formatted} -- MFI: {mfi_formatted} | <code>{contributions.get('V3_Breadth',0):.4f}</code> | **20%** | **{vector_info.get('contribution_percentages', {}).get('V3_Breadth', 0):.1f}%** |
| **Composite Result** | Score: {risk_formatted}/100 -- Dominant: {dominant_strong} | -- | {status_strong} | -- |

{critical_block}

<h3>[ {T("ANALYSIS", "PHAN TICH CHUYEN SAU")} ]</h3>

<div class="analysis-card">
    <div class="analysis-card-title">1. Price Phase Space Dynamics</div>
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
        <span style="color: #888; font-size: 0.82rem;">GMM REGIME:</span>
        <span class="regime-badge" style="background: {'rgba(0,255,65,0.15); color:#00FF41; border: 1px solid #00FF41' if 'STABLE' in current_regime.upper() else 'rgba(255,215,0,0.15); color:#FFD700; border: 1px solid #FFD700' if 'FRAGILE' in current_regime.upper() else 'rgba(255,49,49,0.15); color:#FF3131; border: 1px solid #FF3131'};">{current_regime.upper()}</span>
    </div>
    <div class="xai-trajectory-box">
        <div class="xai-label">XAI Trajectory (Kinematic Overlay -- not ML features)</div>
        <div class="xai-values">
            <div class="xai-item">
                <span class="xai-item-label">V_WPE:</span>
                <span class="xai-item-value">{current_v_wpe:+.5f}</span>
            </div>
            <div class="xai-item">
                <span class="xai-item-label">a_WPE:</span>
                <span class="xai-item-value">{current_a_wpe:+.5f}</span>
            </div>
        </div>
        <div class="xai-narrative">{trajectory_analysis}</div>
    </div>
</div>

<div class="analysis-card">
    <div class="analysis-card-title">2. Volume-Price Fusion</div>
    <div class="analysis-text">
        Macro liquidity is <strong>{liquidity_direction}</strong> the current Price Regime ({regime_strong}).
        Volume structure: {vol_regime_strong}.
        {'Liquidity is structurally aligned with price dynamics.' if vol_is_consensus else 'Structural fragility detected: volume behavior diverges from price entropy.'}
    </div>
</div>

<div class="analysis-card">
    <div class="analysis-card-title">3. Structural Breadth</div>
    <div class="analysis-text">
        {vn30_analysis}
    </div>
</div>

<h3>[ {T("CONCLUSION", "KET LUAN")} ]</h3>
<strong>XAI ATTRIBUTION:</strong> Composite Risk {risk_formatted}/100 ({status_strong}). Dominant: {dominant_strong} ({vector_info.get('contribution_percentages', {}).get(dominant_vector, 0):.1f}%). Market coherence: {'intact' if risk_score < 40 else 'under stress' if risk_score < 75 else 'critically degraded'}. 
<br>
<strong>XAI TRAJECTORY:</strong> {trajectory_analysis}

</div>
"""
st.markdown(agent_log, unsafe_allow_html=True)


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
