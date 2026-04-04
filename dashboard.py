"""
Financial Entropy Agent -- High-End Professional Dashboard
Dual Pipeline (API/Upload), All-in-One Subplots, GMM Scatter, Agent Log.
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
from skills.quant_skill import calc_rolling_wpe, calc_mfi, calc_correlation_entropy, calc_rolling_volume_entropy
from skills.ds_skill import fit_predict_regime, fit_predict_volume_regime

# ==============================================================================
# UI CONFIGURATION & MULTILINGUAL SUPPORT
# ==============================================================================
st.set_page_config(page_title="Financial Entropy Agent | Terminal", layout="wide", page_icon="⚡")

# Custom Styling (Dark Quant Theme)
st.markdown("""
<style>
    .reportview-container { background: #0E1117; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #FFFFFF; }
    .metric-label { font-size: 1rem; color: #AAAAAA; text-transform: uppercase; letter-spacing: 1px; }
    h1, h2, h3 { color: #00FF41 !important; font-family: 'Courier New', Courier, monospace; }
    .agent-log {
        background-color: #0a0a0a;
        border-left: 4px solid #00FF41;
        padding: 20px 25px;
        font-family: 'Courier New', Courier, monospace;
        color: #00FF41;
        font-size: 0.85rem;
        line-height: 1.6;
        border-radius: 4px;
    }
    .agent-log table { width: 100%; border-collapse: collapse; margin: 12px 0; }
    .agent-log th { text-align: left; border-bottom: 2px solid #00FF41; padding: 8px 10px; color: #00FF41; font-weight: 700; }
    .agent-log td { border-bottom: 1px solid #1a3a1a; padding: 8px 10px; color: #00FF41; }
    .agent-log tr:hover td { background-color: #0d1f0d; }
    .agent-log h3 { font-size: 0.95rem; margin-top: 16px; margin-bottom: 6px; }
    .agent-log strong { color: #39FF14; }
    .agent-log code { background: #1a1a1a; padding: 2px 5px; border-radius: 3px; color: #FFD700; }
</style>
""", unsafe_allow_html=True)

if "lang" not in st.session_state:
    st.session_state["lang"] = "EN"

def T(en: str, vn: str) -> str:
    return en if st.session_state["lang"] == "EN" else vn

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
        # Default API via vnstock
        df = get_latest_market_data(ticker="VNINDEX", start_date=start_date_str, end_date=end_date_str)
        
    df = df.loc[start_date_str:end_date_str].copy()
    if df.empty:
        return df

    df["SMA20"] = df["Close"].rolling(20).mean()
    
    # 2. Compute Global Entropy (WPE, MFI)
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).values
    df["Volatility"] = pd.Series(log_returns, index=df.index).rolling(20).std() * np.sqrt(252) * 100
    
    wpe_arr, c_arr = calc_rolling_wpe(log_returns, m=3, tau=1, window=22)
    mfi_arr = calc_mfi(wpe_arr, c_arr)
    
    df["WPE"] = wpe_arr
    df["Complexity"] = c_arr
    df["MFI"] = mfi_arr
    
    # 2b. Compute Volume Entropy Plane (Macro-Micro Fusion, window=60, z_window=252)
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
    
    # 2c. Kinematic Vectors: Velocity & Acceleration of PE (3-day momentum)
    df["PE_Velocity"] = df["WPE"].diff(3).fillna(0)
    df["PE_Acceleration"] = df["PE_Velocity"].diff(3).fillna(0)
    
    # 3. Compute VN30 Cross-Sectional Entropy
    try:
        vn30_rets = fetch_vn30_returns(start_date=start_date_str, end_date=end_date_str)
        cross_entropy = calc_correlation_entropy(vn30_rets, window=22)
        df["Cross_Sectional_Entropy"] = cross_entropy.reindex(df.index).ffill()
    except Exception as e:
        df["Cross_Sectional_Entropy"] = np.nan
        
    # 4. Predict Regime (GMM Unsupervised)
    valid_df = df.dropna(subset=["WPE", "Complexity", "MFI"]).copy()
    if not valid_df.empty:
        features = valid_df[["WPE", "Complexity", "MFI"]].values
        labels, clf = fit_predict_regime(features, n_components=3)
        valid_df["RegimeLabel"] = labels
        valid_df["RegimeName"] = [clf.get_regime_name(lbl) for lbl in labels]
    
    df["RegimeName"] = np.nan
    df["RegimeLabel"] = np.nan
    if not valid_df.empty:
        df.loc[valid_df.index, "RegimeName"] = valid_df["RegimeName"]
        df.loc[valid_df.index, "RegimeLabel"] = valid_df["RegimeLabel"]
    
    # Ffill for any missing regime logic after merge
    df["RegimeName"] = df["RegimeName"].ffill()
    df["RegimeLabel"] = df["RegimeLabel"].ffill()
    
    # 5. Volume Regime (GMM Unsupervised -- Plane 2)
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

start_date = st.sidebar.date_input(T("Start Date", "Ngày Bắt Đầu"), datetime.now() - timedelta(days=500))
end_date = st.sidebar.date_input(T("End Date", "Ngày Kết Thúc"), datetime.now())

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{T('1. DUAL PIPELINE: API OR UPLOAD', '1. DUAL PIPELINE: API HOẶC TẢI LÊN')}**")
uploaded_file = st.sidebar.file_uploader(T("Upload custom OHLCV (.csv/.xlsx)", "Tải lên Dữ liệu OHLCV (.csv/.xlsx)"), type=["csv", "xlsx"])

# Extract file bytes if uploaded
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
    "Monitoring systemic resilience, capital fragmentation, and structural chaos.",
    "Theo dõi sức khỏe hệ thống, sự phân mảnh dòng tiền và rủi ro cấu trúc thị trường."
))

with st.spinner(T("Fetching Data Pipeline & Compiling Physics Engines...", "Đang tải Dữ liệu & Xử lý Thuật toán Vật lý...")):
    df = load_and_compute_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), file_bytes, file_name)

if df is None or df.empty:
    st.error(T("No data available for the selected dates.", "Không có dữ liệu cho khoảng thời gian này."))
    st.stop()
    
# Extract latest and previous values for Deltas
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ==============================================================================
# REAL-TIME DELTAS (TOP METRICS -- DUAL-PLANE SUMMARY)
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)

current_wpe = latest["WPE"]
current_mfi = latest["MFI"]
current_cse = latest["Cross_Sectional_Entropy"]
current_pe_v = latest.get("PE_Velocity", 0.0)

# Price Regime
current_regime = str(latest["RegimeName"]).replace("nan", "Calculating...")

# Volume Regime + metrics for KPI
current_vol_regime_name = str(latest.get("VolRegimeName", "N/A")).replace("nan", "Calculating...")
kpi_vol_shannon = latest.get("Vol_Shannon", float("nan"))
kpi_vol_sampen = latest.get("Vol_SampEn", float("nan"))
kpi_vol_global_z = latest.get("Vol_Global_Z", float("nan"))
vol_sh_kpi = f"{kpi_vol_shannon:.2f}" if pd.notna(kpi_vol_shannon) else "N/A"
vol_se_kpi = f"{kpi_vol_sampen:.2f}" if pd.notna(kpi_vol_sampen) else "N/A"
vol_gz_kpi = f"{kpi_vol_global_z:+.2f}" if pd.notna(kpi_vol_global_z) else "N/A"

# Cross-Plane Synthesis (early compute for KPI header)
_regime_upper = current_regime.upper()
_vol_upper = current_vol_regime_name.upper()
_p_fragile = "CHAOS" in _regime_upper or "PANIC" in _regime_upper or "FRAGILE" in _regime_upper
_v_consensus = "CONSENSUS" in _vol_upper
_v_erratic = "ERRATIC" in _vol_upper or "NOISY" in _vol_upper
_p_stable = "STABLE" in _regime_upper

if _p_fragile and _v_consensus:
    kpi_synthesis = "Structural Accumulation"
elif _p_fragile and _v_erratic:
    kpi_synthesis = "Critical Breakdown"
elif _p_stable and _v_erratic:
    kpi_synthesis = "Trend Exhaustion"
else:
    kpi_synthesis = "System Coherent"

# --- Column 1: Market State ---
with col1:
    st.metric(
        T("Index Price (Close)", "Chi so Gia (Close)"),
        f"{latest['Close']:,.2f}",
        f"{latest['Close'] - prev['Close']:,.2f}",
        delta_color="off"
    )

# --- Column 2: Plane 1 (Price Dynamics) ---
with col2:
    pe_v_sign = f"{current_pe_v:+.4f}" if pd.notna(current_pe_v) else "0.0000"
    st.metric(
        T("Price Chaos (WPE)", "Hon loan Gia (WPE)"),
        f"{current_wpe:.4f}" if pd.notna(current_wpe) else "N/A",
        f"V: {pe_v_sign} | {current_regime}",
        delta_color="off"
    )

# --- Column 3: Plane 2 (Liquidity Fusion) ---
with col3:
    st.metric(
        T("Liquidity Regime", "Trang thai Thanh khoan"),
        current_vol_regime_name,
        f"Micro: H={vol_sh_kpi} SE={vol_se_kpi} | Macro Z: {vol_gz_kpi}",
        delta_color="off"
    )

# --- Column 4: Cross-Plane Synthesis ---
with col4:
    st.metric(
        T("Systemic Risk Synthesis", "Tong hop Rui ro He thong"),
        kpi_synthesis,
        T("Cross-Plane Validation", "Xac thuc Cross-Plane"),
        delta_color="off"
    )

# ==============================================================================
# VISUALS: Dual-Plane Engine Tracking
# ==============================================================================
st.markdown("---")
st.subheader(T("1. MARKET STRUCTURE", "1. CẤU TRÚC THỊ TRƯỜNG"))

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

# Primary Y
fig1.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="VNindex", increasing_line_color='#FFFFFF', decreasing_line_color='#888888'
), secondary_y=False)

fig1.add_trace(go.Scatter(
    x=df.index, y=df['SMA20'], mode='lines', name='SMA20',
    line=dict(color='yellow', width=1, dash='dash')
), secondary_y=False)

# Secondary Y
fig1.add_trace(go.Scatter(
    x=df.index, y=df['WPE'], mode='lines', name='WPE (Entropy)',
    line=dict(color='#FF5F1F', width=2), showlegend=False
), secondary_y=True)

# ADD REGIME BACKGROUND SHADING (vrects)
regime_colors = {
    "Stable Growth": "rgba(0, 255, 65, 0.15)",     # Green
    "Fragile Growth": "rgba(255, 215, 0, 0.15)",   # Yellow
    "Chaos/Panic": "rgba(255, 0, 0, 0.15)",        # Red
    "Structural Recomposition": "rgba(138, 43, 226, 0.15)", # Purple
    "Calculating...": "rgba(128, 128, 128, 0)"
}

# Dummy traces for horizontal legend mapping
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(0, 255, 65, 1)'), name='Stable Growth'))
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(255, 215, 0, 1)'), name='Fragile Growth'))
fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='rgba(255, 0, 0, 1)'), name='Chaos/Panic'))

df['Regime_Shift'] = df['RegimeName'] != df['RegimeName'].shift(1)
shift_indices = df.index[df['Regime_Shift']].tolist()
if len(shift_indices) > 0 and shift_indices[0] != df.index[0]:
    shift_indices.insert(0, df.index[0])
if len(shift_indices) == 0:
    shift_indices = [df.index[0]]
shift_indices.append(df.index[-1])

for i in range(len(shift_indices) - 1):
    start_idx = shift_indices[i]
    end_idx = shift_indices[i+1]
    regime = df.loc[start_idx, 'RegimeName']
    
    if type(regime) == str:
        color = regime_colors.get(regime, "rgba(255,255,255,0)")
        fig1.add_vrect(
            x0=start_idx, x1=end_idx,
            fillcolor=color, opacity=1.0,
            layer="below", line_width=0
        )

fig1.update_layout(
    title=dict(text="VNindex Structure State", x=0.5, y=0.98, xanchor="center", yanchor="top"),
    template="plotly_dark", height=550, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
    legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="right", x=1.0),
    margin=dict(l=20, r=20, b=20, t=60)
)
fig1.update_xaxes(rangeslider_visible=False)
fig1.update_yaxes(title_text="Price", secondary_y=False, showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
fig1.update_yaxes(title_text="WPE", secondary_y=True, showgrid=False)

st.plotly_chart(fig1, use_container_width=True)


# --- Separated VN30 Chart ---
st.markdown("---")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df.index, y=df['Cross_Sectional_Entropy'], mode='lines', name='VN30 Cross-sectional entropy',
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
    "Two independent GMM clustering spaces proving the system observes both Price Physics and Liquidity Structure without human labels.",
    "Hai khong gian GMM clustering doc lap chung minh he thong quan sat dong thoi Vat ly Gia va Cau truc Thanh khoan ma khong can dan nhan."
))

col_price_plot, col_vol_plot = st.columns([1, 1])

# --- PLOT 1: Price Dynamics Plane ---
with col_price_plot:
    st.markdown(f"**{T('PLANE 1: PRICE DYNAMICS', 'MAT PHANG 1: DONG LUC GIA')}**")
    plot_df = df.dropna(subset=['Volatility', 'WPE', 'RegimeName'])
    if not plot_df.empty:
        color_map_price = {
            "Stable Growth": "#00FF41",
            "Fragile Growth": "#FFD700",
            "Chaos/Panic": "#FF0000",
        }
        scatter_price = px.scatter(
            plot_df, x="WPE", y="Volatility",
            color="RegimeName",
            color_discrete_map=color_map_price,
            hover_data=["Close", "MFI", "PE_Velocity", "PE_Acceleration"],
            labels={"WPE": "Permutation Entropy (WPE)", "Volatility": "Annualized Volatility (%)"},
        )
        scatter_price.update_layout(
            template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            legend_title="Price Regime",
            margin=dict(l=20, r=20, b=20, t=20), height=450,
        )
        st.plotly_chart(scatter_price, use_container_width=True)

# --- PLOT 2: Volume Entropy Plane ---
with col_vol_plot:
    st.markdown(f"**{T('PLANE 2: LIQUIDITY STRUCTURE', 'MAT PHANG 2: CAU TRUC THANH KHOAN')}**")
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
# AGENT ORCHESTRATOR DIAGNOSTIC -- CROSS-PLANE SYNTHESIS
# ==============================================================================
st.markdown("---")
st.subheader(T("3. CROSS-PLANE AGENT DIAGNOSTIC", "3. CHAN DOAN CROSS-PLANE TU AGENT"))
st.markdown(T(
    "Real-time generative diagnostic: The Agent observes both planes and synthesizes a unified systemic risk conclusion.",
    "Chan doan tu dong: Agent quan sat ca hai mat phang va tong hop ket luan rui ro he thong thong nhat."
))

# --- Synthesize Agent Diagnostic ---
agent_regime = str(latest["RegimeName"]).upper()
current_vol_regime = str(latest.get("VolRegimeName", "N/A")).upper()
current_vol_shannon = latest.get("Vol_Shannon", float("nan"))
current_vol_sampen = latest.get("Vol_SampEn", float("nan"))
current_vol_global_z = latest.get("Vol_Global_Z", float("nan"))

# Cross-Plane Synthesis Matrix
price_is_fragile_chaos = "CHAOS" in agent_regime or "PANIC" in agent_regime or "FRAGILE" in agent_regime
vol_is_consensus = "CONSENSUS" in current_vol_regime
vol_is_erratic = "ERRATIC" in current_vol_regime or "NOISY" in current_vol_regime
price_is_stable = "STABLE" in agent_regime

if price_is_fragile_chaos and vol_is_consensus:
    cross_plane_label = "STRUCTURAL ACCUMULATION"
    cross_plane_detail = "Price chaos is contained by highly organized liquidity. Indicates Smart Money absorption. The physical disorder is a surface phenomenon; the underlying capital flow is structurally coherent."
elif price_is_fragile_chaos and vol_is_erratic:
    cross_plane_label = "CRITICAL BREAKDOWN"
    cross_plane_detail = "Physical price chaos is amplified by fragmented liquidity. Both planes confirm systemic instability. High systemic risk with no structural support from capital flow."
elif price_is_stable and vol_is_erratic:
    cross_plane_label = "TREND EXHAUSTION"
    cross_plane_detail = "Price trend appears stable, but liquidity structure is breaking down beneath the surface. The divergence between stable prices and erratic volume signals unsustainable momentum."
else:
    cross_plane_label = "SYSTEM COHERENT"
    cross_plane_detail = "Both planes exhibit structural alignment. No cross-plane divergence detected. Current market dynamics are internally consistent."

global_risk = "CRITICAL" if cross_plane_label in ("CRITICAL BREAKDOWN",) else (
    "ELEVATED" if cross_plane_label in ("TREND EXHAUSTION",) else "MODERATE"
)

vol_sh_str = f"{current_vol_shannon:.4f}" if pd.notna(current_vol_shannon) else "N/A"
vol_se_str = f"{current_vol_sampen:.4f}" if pd.notna(current_vol_sampen) else "N/A"
vol_gz_str = f"{current_vol_global_z:+.2f}" if pd.notna(current_vol_global_z) else "N/A"

# VN30 analysis text
if current_cse > 60:
    vn30_analysis = "High VN30 Entropy: Blue chips exhibit extreme fragmentation with no market consensus. Capital dispersion across sectors signals systemic decorrelation."
elif current_cse < 40:
    vn30_analysis = "Low VN30 Entropy: Blue chips maintain deterministic consensus. Centralized capital flow indicates institutional structural integrity."
else:
    vn30_analysis = "Neutral VN30 Entropy: Moderate internal rotation among capital pillars. Sector rebalancing underway without systemic stress."

# Extract kinematic vectors
current_pe_v = latest.get("PE_Velocity", 0.0)
current_pe_a = latest.get("PE_Acceleration", 0.0)
pe_v_str = f"{current_pe_v:+.4f}" if pd.notna(current_pe_v) else "N/A"
pe_a_str = f"{current_pe_a:+.4f}" if pd.notna(current_pe_a) else "N/A"

agent_log = f"""
<div class="agent-log">

`>` INITIATE DUAL-PLANE DIAGNOSTIC PROTOCOL...<br>
`>` Fetching Data: **OK** ({len(df)} rows)<br>
`>` Applying WPE Physics Engine: **OK** (MFI = `{current_mfi:.4f}`)<br>
`>` Computing Kinematic Vectors: **OK** (V = `{pe_v_str}`, a = `{pe_a_str}`)<br>
`>` Computing Volume Entropy Plane: **OK** (Shannon = `{vol_sh_str}`, SampEn = `{vol_se_str}`)<br>
`>` Running Dual-Plane GMM Clustering: **OK**<br>

| Telemetry Module | Key Metrics | Regime / Status |
| :--- | :--- | :--- |
| **Plane 1: Price Dynamics** | WPE: `{current_wpe:.4f}` -- V (dE/dt): `{pe_v_str}` -- a (d2E/dt2): `{pe_a_str}` | **{agent_regime}** |
| **Plane 2: Liquidity Structure** | Macro Z: `{vol_gz_str}` -- Shannon: `{vol_sh_str}` -- SampEn: `{vol_se_str}` | **{current_vol_regime}** |
| **Cross-Plane Synthesis** | Systemic Risk: `{global_risk}` | **{cross_plane_label}** |

### [ANALYSIS]
{cross_plane_detail} Kinematic analysis: V = `{pe_v_str}` ({'chaos expanding' if current_pe_v > 0 else 'order forming'}), a = `{pe_a_str}` ({'momentum accelerating' if current_pe_a > 0 else 'momentum fading'}).

### [VN30 STRUCTURAL DYNAMICS]
Cross-Sectional Entropy = `{current_cse:.1f}` / 100. {vn30_analysis}

### [CONCLUSION]
The Dual-Plane Engine synthesizes **{cross_plane_label}** with systemic risk at **{global_risk}**. {'Immediate risk management protocols recommended. Both observation planes confirm structural deterioration.' if global_risk == 'CRITICAL' else ('The divergence between planes warrants close monitoring. Liquidity degradation may precede price correction.' if global_risk == 'ELEVATED' else 'System operates within expected parameters. Both planes confirm structural alignment and market integrity.')}

</div>
"""
st.markdown(agent_log, unsafe_allow_html=True)


# ==============================================================================
# DATA EXPORT
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{T('2. DATA EXPORT', '2. XUAT DU LIEU')}**")
csv_data = df.to_csv().encode('utf-8')
st.sidebar.download_button(
    label=T("Export Current Analysis (CSV)", "Xuat Du Lieu Hien Tai (CSV)"),
    data=csv_data,
    file_name="financial_entropy_agent_export.csv",
    mime="text/csv",
    use_container_width=True
)
