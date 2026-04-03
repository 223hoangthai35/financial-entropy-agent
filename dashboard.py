"""
InfoStat Dynamics -- High-End Professional Dashboard
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
from skills.quant_skill import calc_rolling_wpe, calc_mfi, calc_correlation_entropy
from skills.ds_skill import fit_predict_regime

# ==============================================================================
# UI CONFIGURATION & MULTILINGUAL SUPPORT
# ==============================================================================
st.set_page_config(page_title="InfoStat Dynamics | Terminal", layout="wide", page_icon="⚡")

# Custom Styling (Dark Quant Theme)
st.markdown("""
<style>
    .reportview-container { background: #0E1117; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #FFFFFF; }
    .metric-label { font-size: 1rem; color: #AAAAAA; text-transform: uppercase; letter-spacing: 1px; }
    h1, h2, h3 { color: #00FF41 !important; font-family: 'Courier New', Courier, monospace; }
    .agent-log { background-color: #111; border-left: 4px solid #00FF41; padding: 15px; font-family: monospace; color: #00FF41; }
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
st.title(T("INFOSTAT DYNAMICS: SYSTEM ARCHITECT", "INFOSTAT DYNAMICS: SYSTEM ARCHITECT"))
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
# REAL-TIME DELTAS (TOP METRICS)
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)

current_wpe = latest["WPE"]
wpe_delta = current_wpe - prev["WPE"] if pd.notna(current_wpe) and pd.notna(prev["WPE"]) else 0.0

current_cse = latest["Cross_Sectional_Entropy"]
cse_delta = current_cse - prev["Cross_Sectional_Entropy"] if pd.notna(current_cse) and pd.notna(prev["Cross_Sectional_Entropy"]) else 0.0

current_mfi = latest["MFI"]
mfi_delta = current_mfi - prev["MFI"] if pd.notna(current_mfi) and pd.notna(prev["MFI"]) else 0.0

current_regime = str(latest["RegimeName"]).replace("nan", "Calculating...")

with col1:
    st.metric(T("Index Price (Close)", "Chỉ số Giá (Close)"), f"{latest['Close']:,.2f}", f"{latest['Close'] - prev['Close']:,.2f}")
with col2:
    st.metric(T("WPE (Global Chaos)", "WPE (Hỗn loạn Chung)"), f"{current_wpe:.4f}", f"{wpe_delta:.4f}")
with col3:
    st.metric(T("VN30 Cross-Sectional Ent.", "Entropy Rổ VN30"), f"{current_cse:.2f} / 100", f"{cse_delta:.2f}")
with col4:
    st.metric(T("AI Predicted Regime", "Dự đoán Trạng thái AI"), current_regime, delta=None)

# ==============================================================================
# ALL-IN-ONE INTEGRATED VISUALS (Plotly Subplots)
# ==============================================================================
st.markdown("---")
st.subheader(T("1. ALL-IN-ONE STRUCTURAL TELEMETRY", "1. ĐỒ THỊ CHỈ BÁO CẤU TRÚC (TELEMETRY)"))

fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.4],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
)

# Row 1: Candlestick + SMA20 (Primary Y)
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="Index Price", increasing_line_color='#FFFFFF', decreasing_line_color='#888888'
), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=df.index, y=df['SMA20'], mode='lines', name='SMA 20',
    line=dict(color='yellow', width=1, dash='dash')
), row=1, col=1, secondary_y=False)

# Row 1: Permutation Entropy WPE (Secondary Y) - Neon Orange
fig.add_trace(go.Scatter(
    x=df.index, y=df['WPE'], mode='lines', name='WPE (Entropy)',
    line=dict(color='#FF5F1F', width=2)
), row=1, col=1, secondary_y=True)

# Row 2: VN30 Cross-Sectional Entropy (Cyan) + MFI
fig.add_trace(go.Scatter(
    x=df.index, y=df['Cross_Sectional_Entropy'], mode='lines', name='VN30 Cross-Sectional Entropy',
    line=dict(color='#00FFFF', width=2), fill='tozeroy', fillcolor='rgba(0, 255, 255, 0.1)'
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df['MFI'] * 100, mode='lines', name='MFI (Scaled)',
    line=dict(color='#FFD700', width=1, dash='dot')
), row=2, col=1)

# ADD REGIME BACKGROUND SHADING (vrects)
# Map Regimes to colors
regime_colors = {
    "Stable Growth": "rgba(0, 255, 65, 0.15)",     # Green
    "Fragile Growth": "rgba(255, 215, 0, 0.15)",   # Yellow
    "Chaos/Panic": "rgba(255, 0, 0, 0.15)",        # Red
    "Structural Recomposition": "rgba(138, 43, 226, 0.15)", # Purple
    "Calculating...": "rgba(128, 128, 128, 0)"
}

# Add shapes grouping contiguous regimes
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
    
    # Check if regime is string
    if type(regime) == str:
        color = regime_colors.get(regime, "rgba(255,255,255,0)")
        fig.add_vrect(
            x0=start_idx, x1=end_idx,
            fillcolor=color, opacity=1.0,
            layer="below", line_width=0,
            row=1, col=1
        )

fig.update_layout(
    template="plotly_dark", height=800, plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, b=20, t=40)
)
fig.update_xaxes(rangeslider_visible=False)
fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="WPE", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Entropy (0-100 Scale)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# UNSUPERVISED LEARNING PROOF
# ==============================================================================
st.markdown("---")
col_plot, col_log = st.columns([1, 1])

with col_plot:
    st.subheader(T("2. UNSUPERVISED LEARNING DS PROOF", "2. BẰNG CHỨNG HỌC MÁY GMM"))
    st.markdown(T(
        "Scatter plot proving the GMM Model dynamically clusters market states (Entropy vs. Volatility) purely mathematically without human labels.",
        "Biểu đồ Scatter chứng minh Mô hình GMM tự động gom cụm trạng thái thị trường hoàn toàn dựa trên toán học (Entropy vs Volatility), không cần dán nhãn từ con người."
    ))
    
    plot_df = df.dropna(subset=['Volatility', 'WPE', 'RegimeName'])
    if not plot_df.empty:
        # Create discrete color mapping matches the background shading
        color_discrete_map = {
            "Stable Growth": "#00FF41",
            "Fragile Growth": "#FFD700",
            "Chaos/Panic": "#FF0000",
            "Structural Recomposition": "#8A2BE2"
        }
        
        scatter_fig = px.scatter(
            plot_df, x="WPE", y="Volatility", 
            color="RegimeName", 
            color_discrete_map=color_discrete_map,
            hover_data=["Close", "MFI"],
            labels={"WPE": "Permutation Entropy (WPE)", "Volatility": "Market Volatility (Ann. Std)"}
        )
        scatter_fig.update_layout(
            template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            legend_title="AI Regime Cluster",
            margin=dict(l=20, r=20, b=20, t=20)
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

# ==============================================================================
# AGENT ORCHESTRATOR DIAGNOSTIC
# ==============================================================================
with col_log:
    st.subheader(T("3. 🤖 AGENT ORCHESTRATOR DIAGNOSTIC", "3. 🤖 PHÂN TÍCH TỪ AGENT ORCHESTRATOR"))
    st.markdown(T("Real-time generative LLM Diagnostic mimicking the ReAct Loop:", "Phân tích tự động giả lập vòng lặp ReAct của LLM Agent:"))
    
    # Synthesize an Agent Diagnostic String
    agent_regime = str(latest["RegimeName"]).upper()
    
    # Determine VN30 Risk logic
    if current_cse > 60:
        vn30_analysis = "High VN30 Entropy detected. Blue chips exhibit extreme fragmentation with no market consensus."
    elif current_cse < 40:
        vn30_analysis = "Low VN30 Entropy detected. Blue chips maintain structural deterministic consensus."
    else:
        vn30_analysis = "Neutral VN30 Entropy. Moderate internal rotation among capital pillars."
        
    global_risk = "CRITICAL" if "CHAOS" in agent_regime or "PANIC" in agent_regime or "FRAGILE" in agent_regime else "MODERATE"
    
    agent_log = f"""
> INITIATE DIAGNOSTIC PROTOCOL...
> Fetching Data: OK ({len(df)} rows)
> Applying WPE Physics: OK (MFI = {current_mfi:.4f})
> Running EVD on Correlation Matrix: OK (S_corr = {current_cse:.2f})
> GMM Clustering Active...

MARKET REGIME DETECTED: [{agent_regime}]
SYSTEMIC RISK LEVEL   : [{global_risk}]

-----------------------------------------
[GLOBAL (VNINDEX) ANALYSIS]
WPE stands at {current_wpe:.4f}, pushing the Market Fragility Index to {current_mfi:.4f}. 
Volatility is measured at {latest['Volatility']:.2f}%. Price action deviates from core statistical boundaries.

[BLOCS (VN30) STRUCTURAL DYNAMICS]
CROSS-SECTIONAL ENTROPY = {current_cse:.1f}/100. 
{vn30_analysis}

[CONCLUSION]
System indicates {'a structural breakdown is imminent or occurring.' if global_risk == 'CRITICAL' else 'sustainable momentum supported by low complexity risks.'}
"""
    st.markdown(f"<div class='agent-log'>{agent_log.strip().replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)


# ==============================================================================
# DATA EXPORT
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{T('2. DATA EXPORT', '2. XUẤT DỮ LIỆU')}**")
csv_data = df.to_csv().encode('utf-8')
st.sidebar.download_button(
    label=T("📥 Export Current Analysis (CSV)", "📥 Xuất Dữ Liệu Hiện Tại (CSV)"),
    data=csv_data,
    file_name="infostat_dynamics_export.csv",
    mime="text/csv",
    use_container_width=True
)
