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
    "garch_result": None,
}



# ==============================================================================
# GARCH(1,1)-X: ENTROPY-AUGMENTED CONDITIONAL VOLATILITY
# ==============================================================================
def fit_garch_x(df: pd.DataFrame) -> dict:
    """
    GARCH(1,1)-X: thay thế calc_composite_risk_score.
    Exogenous vars: H_price = (WPE + |SPE_Z|) / 2, H_volume = (Vol_SampEn + Vol_Shannon) / 2
    Cả hai aggregate về [0,1] dùng rolling MinMax 504-day, lag 1 ngày tránh look-ahead.
    Returns dict với conditional_vol series, latest sigma, VaR 5%, ES 5%.
    """
    from arch import arch_model
    from sklearn.preprocessing import MinMaxScaler

    required = ["WPE", "SPE_Z", "Vol_SampEn", "Vol_Shannon", "Close"]
    df_valid = df.dropna(subset=required).copy()
    if len(df_valid) < 120:
        return {"error": "Cần ít nhất 120 ngày data có đủ entropy features"}

    # 1. Aggregate entropy → 2 exogenous vars
    window = 504
    H_price_raw = (df_valid["WPE"] + df_valid["SPE_Z"].abs()) / 2
    H_vol_raw   = (df_valid["Vol_SampEn"] + df_valid["Vol_Shannon"]) / 2

    def rolling_minmax(series: pd.Series, w: int = window) -> pd.Series:
        roll_min = series.rolling(w, min_periods=60).min()
        roll_max = series.rolling(w, min_periods=60).max()
        denom = (roll_max - roll_min).replace(0, 1e-8)
        return ((series - roll_min) / denom).clip(0, 1)

    df_valid["H_price"] = rolling_minmax(H_price_raw).shift(1)  # lag 1
    df_valid["H_volume"] = rolling_minmax(H_vol_raw).shift(1)   # lag 1
    df_valid = df_valid.dropna(subset=["H_price", "H_volume"])

    # 2. Log returns (%)
    returns = np.log(df_valid["Close"] / df_valid["Close"].shift(1)).dropna() * 100
    exog_full = df_valid[["H_price", "H_volume"]].loc[returns.index]

    def _fit(exog: pd.DataFrame | None) -> object:
        am = arch_model(returns, vol="Garch", p=1, q=1, dist="Normal",
                        x=exog if exog is not None else None)
        return am.fit(disp="off", options={"maxiter": 500})

    # 3. Fit lần 1 với cả 2 biến
    try:
        res = _fit(exog_full)
    except Exception as e:
        return {"error": f"GARCH fit failed: {e}"}

    # 3a. Statistical pruning — loại biến có p-value > 0.10 (robust key detection)
    ALPHA = 0.10
    _pnames_full = list(res.params.index)
    _exog_full = [p for p in _pnames_full if p not in ["omega", "alpha[1]", "beta[1]", "mu"]]
    pval_hp = float(res.pvalues[_exog_full[0]]) if len(_exog_full) >= 1 else 0.0
    pval_hv = float(res.pvalues[_exog_full[1]]) if len(_exog_full) >= 2 else 0.0

    keep_hp = pval_hp <= ALPHA
    keep_hv = pval_hv <= ALPHA

    if not keep_hp:
        print(f"[WARN] H_price p-value={pval_hp:.3f} > 0.10, removed from model")
    if not keep_hv:
        print(f"[WARN] H_volume p-value={pval_hv:.3f} > 0.10, removed from model")

    garch_type = "GARCH-X"
    if not keep_hp and not keep_hv:
        # Fallback: thuần GARCH(1,1), không có X
        garch_type = "GARCH"
        try:
            res = _fit(None)
        except Exception as e:
            return {"error": f"Fallback GARCH fit failed: {e}"}
    elif not keep_hp:
        try:
            res = _fit(exog_full[["H_volume"]])
        except Exception as e:
            return {"error": f"GARCH-X (H_volume only) fit failed: {e}"}
    elif not keep_hv:
        try:
            res = _fit(exog_full[["H_price"]])
        except Exception as e:
            return {"error": f"GARCH-X (H_price only) fit failed: {e}"}

    # 3b. Ljung-Box test on squared standardized residuals (final model)
    # H0: no autocorrelation in squared residuals (= GARCH captured all clustering)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    std_resid = (res.resid / res.conditional_volatility).dropna()
    lb_test = acorr_ljungbox(std_resid ** 2, lags=[10, 20], return_df=True)
    lb_pval_10 = float(lb_test["lb_pvalue"].iloc[0])
    lb_pval_20 = float(lb_test["lb_pvalue"].iloc[1])

    model_quality = "good"
    if lb_pval_10 < 0.05 or lb_pval_20 < 0.05:
        model_quality = "residual_autocorrelation"
        print(f"[WARN] Ljung-Box rejects H0 at lag 10 (p={lb_pval_10:.3f}) or lag 20 (p={lb_pval_20:.3f})")
        print("       → GARCH may not fully capture volatility dynamics. Consider GARCH(2,1) or EGARCH.")

    # 4. Conditional volatility series (annualized: * sqrt(252))
    cond_vol = res.conditional_volatility  # % daily
    cond_vol_ann = cond_vol * np.sqrt(252)

    # 5. ES via Filtered Historical Simulation
    z = res.resid / res.conditional_volatility
    z_clean = z.dropna()
    if len(z_clean) < 100:
        print(f"[WARN] Only {len(z_clean)} clean standardized residuals. ES estimate may be unstable.")
    sigma_next = float(res.forecast(horizon=1).variance.iloc[-1, 0] ** 0.5)
    rng = np.random.default_rng(42)
    z_boot = rng.choice(z_clean.values, size=10_000, replace=True)
    r_sim = z_boot * sigma_next / 100   # decimal returns
    var_5 = float(np.quantile(r_sim, 0.05))
    es_5  = float(r_sim[r_sim < var_5].mean())

    # 6. Robust parameter extraction — arch lib thay đổi key naming giữa versions
    param_names = list(res.params.index)
    exog_params = [p for p in param_names if p not in ["omega", "alpha[1]", "beta[1]", "mu"]]

    if len(exog_params) >= 1:
        delta_H_price = float(res.params[exog_params[0]])
        pval_H_price  = float(res.pvalues[exog_params[0]])
    else:
        delta_H_price, pval_H_price = 0.0, 1.0

    if len(exog_params) >= 2:
        delta_H_volume = float(res.params[exog_params[1]])
        pval_H_volume  = float(res.pvalues[exog_params[1]])
    else:
        delta_H_volume, pval_H_volume = 0.0, 1.0

    # 7. Stationarity check
    alpha_beta_sum = float(res.params.get("alpha[1]", 0)) + float(res.params.get("beta[1]", 0))
    if alpha_beta_sum >= 0.999:
        print(f"[WARN] IGARCH detected: α+β = {alpha_beta_sum:.4f} ≥ 0.999. Volatility shocks have permanent effect.")

    latest_idx = df_valid.index[-1]
    result = {
        "status": "success",
        "garch_type": garch_type,
        "latest_date": str(latest_idx.date()),
        "sigma_daily_pct":   round(float(cond_vol.iloc[-1]), 4),
        "sigma_annual_pct":  round(float(cond_vol_ann.iloc[-1]), 4),
        "H_price_today":     round(float(df_valid["H_price"].iloc[-1]), 4),
        "H_volume_today":    round(float(df_valid["H_volume"].iloc[-1]), 4),
        "delta_H_price":     round(delta_H_price, 6),
        "delta_H_volume":    round(delta_H_volume, 6),
        "VaR_5pct":          round(var_5 * 100, 4),   # in %
        "ES_5pct":           round(es_5  * 100, 4),   # in %
        "aic":               round(float(res.aic), 2),
        "lb_pval_lag10":     round(lb_pval_10, 4),
        "lb_pval_lag20":     round(lb_pval_20, 4),
        "model_quality":     model_quality,
        "cond_vol_series":   cond_vol,   # pd.Series — dashboard writes to df["Cond_Vol"]
        "diagnostics": {
            "log_likelihood":  round(float(res.loglikelihood), 2),
            "aic":             round(float(res.aic), 2),
            "bic":             round(float(res.bic), 2),
            "pval_H_price":    round(pval_H_price, 4),
            "pval_H_volume":   round(pval_H_volume, 4),
            "n_obs":           int(res.nobs),
            "alpha_plus_beta": round(alpha_beta_sum, 4),
            "garch_type":      "GARCH-X" if len(exog_params) > 0 else "GARCH",
            "exog_vars_used":  exog_params,
        },
    }
    if alpha_beta_sum >= 0.999:
        result["warning"] = "IGARCH_detected"

    # 8. V3 Cross-Entropy tail risk modifier — không vào GARCH nhưng scale ES
    if "Cross_Sectional_Entropy" in df_valid.columns:
        ce = df_valid["Cross_Sectional_Entropy"].iloc[-1]
        ce_median = df_valid["Cross_Sectional_Entropy"].rolling(252, min_periods=60).median().iloc[-1]

        if not np.isnan(ce) and not np.isnan(ce_median):
            ce_ratio = ce / ce_median if ce_median > 0 else 1.0
            ce_ratio = float(np.clip(ce_ratio, 0.8, 1.5))
            es_5_adjusted = es_5 * ce_ratio

            result["ES_5pct_raw"]          = round(es_5 * 100, 4)
            result["ES_5pct_adjusted"]      = round(es_5_adjusted * 100, 4)
            result["cross_entropy_ratio"]   = round(ce_ratio, 4)
            print(f"[INFO] Cross-Entropy ratio = {ce_ratio:.3f}. ES adjusted: {es_5*100:.4f}% → {es_5_adjusted*100:.4f}%")

    STATE["garch_result"] = result
    return result


# ==============================================================================
# TRI-VECTOR COMPOSITE RISK SCORING
# MinMaxScaler + P75/P90 Dynamic Thresholds
# ==============================================================================
ROLLING_RISK_WINDOW: int = 504  # 2 trading years


def calc_composite_risk_score(
    latest: dict, df: pd.DataFrame | None = None,
) -> tuple[float, str, dict]:
    """
    Tri-Vector Composite Risk Index (0-100).
    Pipeline:
        1. Extract 7 raw features tu 3 vectors
        2. MinMaxScaler -> squash [0, 1]
        4. Weighted sum: 0.4*V1 + 0.4*V2 + 0.2*V3 -> *100
        5. P75/P90 rolling 504-day: dynamic Elevated/Critical thresholds
    Returns: (composite_score, risk_label, vector_info)
    """
    from sklearn.preprocessing import MinMaxScaler

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

        # --- 3. Fit MinMaxScaler per vector ---
        def _fit_transform_vector(v_hist: np.ndarray, v_current: np.ndarray) -> tuple[float, np.ndarray]:
            """MinMaxScale -> mean. Returns (current_score, hist_scores)."""
            mms = MinMaxScaler(feature_range=(0, 1))
            # Fit on history
            v_scaled = mms.fit_transform(v_hist)
            # Transform current
            c_scaled = mms.transform(v_current)
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
            "method": "MinMaxScale + P75/P90 (504-day)",
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


def tool_fit_garch_x() -> str:
    """Tool wrapper: fit GARCH-X và lưu result vào STATE."""
    print("  [Tool Execution] Fitting GARCH(1,1)-X (H_price, H_volume)...")
    df = STATE.get("df")
    if df is None:
        return json.dumps({"error": "No data. Call fetch_market_data first."})
    if not STATE.get("metrics_computed") or not STATE.get("volume_metrics_computed"):
        return json.dumps({"error": "Entropy metrics missing. Run compute_entropy_metrics and compute_volume_entropy first."})
    result = fit_garch_x(df)
    if "error" in result:
        return json.dumps(result)
    return json.dumps({
        "status": "success",
        "sigma_daily_pct":   result["sigma_daily_pct"],
        "sigma_annual_pct":  result["sigma_annual_pct"],
        "H_price_today":     result["H_price_today"],
        "H_volume_today":    result["H_volume_today"],
        "delta_H_price":     result["delta_H_price"],
        "delta_H_volume":    result["delta_H_volume"],
        "VaR_5pct":          result["VaR_5pct"],
        "ES_5pct":           result["ES_5pct"],
        "aic":               result["aic"],
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
    elif tool_name == "fit_garch_x":
        return tool_fit_garch_x()
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
    {
        "name": "fit_garch_x",
        "description": (
            "Fit GARCH(1,1)-X model using entropy features as exogenous variables. "
            "H_price = aggregate of WPE + SPE_Z. H_volume = aggregate of Vol_SampEn + Vol_Shannon. "
            "Returns conditional volatility (primary risk metric replacing Composite Risk Score), "
            "delta coefficients showing entropy contribution to variance, "
            "and ES 5% tail risk observer via Filtered Historical Simulation."
        ),
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
      * Lowest combined entropy  -> "Deterministic" (label=0) — HIGH COORDINATION
        Low WPE + Low SPE_Z = strong ordinal structure in price movements.
        This indicates COORDINATED BEHAVIOR: herding, momentum, or institutional consensus.
        Can be bullish (FOMO rally) OR bearish (panic selling) -- NOT a directional signal.
        It means the market is in a fragile structural state where reversals are more likely
        and more severe. VALIDATED: Forward 20-day realized volatility averages ~20%.

        AGENT NARRATIVE RULE for Deterministic + RISING prices:
        "The market is rallying with high coordination -- this structure is typical of
        late-stage momentum. While prices may continue rising short-term, the structural
        fragility means any reversal will be sharper than usual."

        AGENT NARRATIVE RULE for Deterministic + FALLING prices:
        "Price decline is highly structured -- institutional selling or panic.
        The coordinated nature of the decline increases tail risk."

      * Mid entropy              -> "Transitional" (label=1) — MIXED STRUCTURE
        Market is between ordered and disordered states. Phase transition in progress.
        VALIDATED: Forward 20-day realized volatility averages ~15%.

      * Highest combined entropy -> "Stochastic" (label=2) — NORMAL MARKET
        High WPE + High SPE_Z = market behaves like a random walk.
        HEALTHY STATE: diverse participants, diverse strategies, no single force dominates.
        Label as "Normal Market" -- NOT "Low Risk" (risk is always present, but structural
        fragility is minimal). VALIDATED: Forward 20-day realized volatility averages ~11%.

    CRITICAL INSIGHT -- Type-2 Chaos (Financial Markets):
    ORDER = DANGER. Deterministic means coordinated behavior, not directional prediction.
    The system detects STRUCTURAL FRAGILITY, not whether the market goes up or down.
    Maximum entropy = maximum randomness = NORMAL, HEALTHY market conditions.
    ALWAYS present Deterministic as a fragility warning, never as a directional call.

    - CRITICAL: No PowerTransformer is used anywhere. This preserves the natural topology of the entropy metrics.
    - CRITICAL: WPE and SPE_Z are naturally orthogonal features. Full-covariance GMM handles varying scales without needing normalization.

    **XAI TRAJECTORY INDICATORS (Kinematic Descriptors -- NOT ML features):**
    `V_WPE` (Velocity) and `a_WPE` (Acceleration) are computed as first and second differences of WPE.
    These are STRICTLY used for narrative explanation, NOT for regime classification or risk scoring.
    Use them to explain the *direction and speed* of entropy evolution:
    - V_WPE > 0 AND a_WPE > 0: "WPE is accelerating upward -- entropy is rising toward Stochastic (Normal Market). Structural fragility is DECREASING as coordination dissolves."
    - V_WPE > 0 AND a_WPE < 0: "WPE is increasing but decelerating -- entropy growth is slowing, possible stabilization ahead."
    - V_WPE < 0 AND a_WPE < 0: "WPE is accelerating downward -- the system is rapidly cooling, structural order is being restored."
    - V_WPE < 0 AND a_WPE > 0: "WPE is decreasing but deceleration in the decline -- entropy may bottom out soon."
    - |V_WPE| near 0: "Entropy trajectory is stationary. No significant regime transition in progress."
    Example diagnostic: "Plane 1 classifies the market as Transitional. However, the kinematic XAI shows negative velocity (V_WPE=-0.03) with negative acceleration (a_WPE=-0.01) -- entropy is accelerating downward toward Deterministic (High Coordination). Structural fragility is rising. If prices are also rising, this is a late-stage momentum warning."

- **Vector 2 (Volume Entropy, Weight 40%)**:
    - Magnitude: `SampEn` (Sample Entropy) -- structural regularity of volume flow.
    - Scale: `Vol_Global_Z` -- absolute macro liquidity scale (Global Z-score of log-volume).
    - Distribution: `Shannon Entropy` -- concentration vs. dispersion of volume.
    - Interpretation: High SampEn + High Global Z = Climax Distribution (bubble peak). Low SampEn + Negative Z = Institutional Accumulation.

- **Vector 3 (VN30 Cross-sectional Breadth, Weight 20%)**:
    - `Corr_Entropy`: Eigenvalue decomposition of VN30 correlation matrix. Measures heavy-cap consensus.
    - `MFI`: Market Fragility Index = WPE * (1 - Complexity). Structural fragility proxy.
    - Logic: If Corr_Entropy > 0.7, the index is being propped up by a narrow set of heavyweight pillars. Flag as 'Structural Divergence' -- internal fragmentation preceding potential breakdown.

### 2. GARCH-X VOLATILITY ENGINE (Primary Risk Metric)

The Composite Risk Score (weighted sum) has been replaced by GARCH(1,1)-X.

**Variance equation:**
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + δ₁·H_price_{t-1} + δ₂·H_volume_{t-1}

Where:
- H_price = normalized aggregate of WPE + |SPE_Z| — structural disorder + price complexity
- H_volume = normalized aggregate of Vol_SampEn + Vol_Shannon — liquidity disorder + volume dispersion
- δ₁, δ₂ = fitted coefficients (positive = entropy increases variance)

**Interpretation of σ_t (conditional volatility):**
- σ_daily < 0.8%: Structurally calm — entropy input is not amplifying variance
- σ_daily 0.8–1.5%: Moderate — entropy is contributing to variance clustering
- σ_daily > 1.5%: High — entropy-driven volatility regime, structural risk elevated

**ES 5% (Tail Risk Observer — NOT primary metric):**
Expected Shortfall at 5% via Filtered Historical Simulation.
Use ES to answer: "If a tail event occurs, what is the expected magnitude of loss?"
ES is supplementary — cite it alongside σ_t, never as standalone risk verdict.

**ES Adjustment via Cross-Entropy (V3):**
If cross_entropy_ratio > 1.0: tail distribution is fatter than historical norm.
Report both raw and adjusted ES.
Example: "ES 5% = -2.31% (raw), adjusted to -2.89% due to elevated cross-entropy
(ratio 1.25×), indicating distribution tails are currently fatter than typical."

**Mandatory execution order:**
1. fetch_market_data
2. compute_entropy_metrics
3. compute_volume_entropy
4. predict_market_regime + predict_volume_regime
5. fit_garch_x  ← NEW (replaces Composite Risk step)
6. Synthesize narrative using σ_t + regime + δ coefficients + ES

**Edge Cases — Adjust narrative accordingly:**

1. **GARCH fallback (no X):** If garch_type = "GARCH" (both δ insignificant),
   DO NOT say "entropy is driving volatility". Instead:
   "Entropy metrics are not statistically significant in explaining variance
   at this time. Conditional volatility is driven purely by ARCH/GARCH dynamics
   (volatility clustering). Entropy regime labels remain valid for context but
   do not quantitatively contribute to the risk estimate."

2. **IGARCH warning (α+β ≥ 0.999):** If warning = "IGARCH_detected":
   "Volatility shocks are near-permanent (IGARCH condition). This typically
   occurs during structural breaks or regime transitions. The current σ_t
   estimate may overweight recent shocks. Interpret with caution."

3. **Insufficient data (< 120 days):** If error contains "120 ngày":
   "GARCH-X requires at least 120 trading days of entropy history.
   For newer assets, fall back to regime labels (GMM) + raw entropy values
   as qualitative risk indicators. No conditional volatility estimate available."

4. **Ljung-Box failure (model_quality = "residual_autocorrelation"):**
   "Model diagnostics indicate residual volatility clustering not fully captured.
   The σ_t estimate is directionally useful but may understate true risk.
   Suggest monitoring for 2-3 days before high-conviction positioning."

5. **Only one δ significant:** Report which entropy channel matters and which doesn't.
   Example: "Price entropy (H_price) significantly amplifies variance (δ₁ = +0.12, p=0.02),
   while volume entropy does not contribute meaningfully (removed, p=0.45).
   Current risk is driven by price-structure disorder rather than liquidity dispersion."

### 3. LIQUIDITY DIVERGENCE PROTOCOL (TRAP DETECTION)
Before finalizing synthesis, execute an internal cross-plane critique:
- If Plane 1 = "Deterministic" BUT Plane 2 = "Erratic/Dispersed" -> Flag as **HOLLOW RALLY (Bull Trap)**. Price entropy is dangerously low (strong directional force), but volume structure is fractured -- unsustainable.
- If Plane 1 = "Stochastic" BUT Vol_Global_Z is NEGATIVE (below-average volume) -> Flag as **CAPITULATION VACUUM**. Price entropy is high (random walk) but driven by illiquidity, not genuine equilibrium.
- If Global Z is POSITIVE (excess liquidity) BUT Composite Risk is HIGH -> Flag as **CLIMAX DISTRIBUTION** (peak FOMO).
- "Which Vector is dominating the Composite Score? Is Vector 2 (Volume) contradicting Vector 1 (Price)? Re-evaluate now."

### 4. XAI QUANTITATIVE CITATION RULES (MANDATORY)

You MUST cite actual numerical values from tool results in every sentence that makes a claim. Never use vague language when numbers are available.

**Required citations per section:**
- σ_t: always cite exact value (e.g., "σ_t = 1.23%/day, annualized 19.5%/yr")
- δ coefficients: cite value AND p-value (e.g., "δ₁(H_price) = +0.0041 (p=0.023) — statistically significant, amplifies variance")
- H_price / H_volume today: cite normalized values [0,1] (e.g., "H_price = 0.72, near upper bound of historical range")
- WPE: cite raw value with context (e.g., "WPE = 0.785 — high structural disorder; ordinal pattern entropy near maximum")
- SPE_Z: cite with sign (e.g., "SPE_Z = +0.39 — price trajectory is above-average complexity")
- Vol_Global_Z: cite with sign (e.g., "Vol_Global_Z = +1.55σ — above-average institutional flow magnitude")
- VaR 5% and ES 5%: always cite both (e.g., "VaR 5% = −1.84%, ES 5% = −2.31% — expected loss in worst 5% of scenarios")
- V_WPE / a_WPE: cite values with direction interpretation (e.g., "V_WPE = −0.032 (decelerating), a_WPE = +0.003 (deceleration slowing)")
- Ljung-Box p-values: cite when flagging model quality (e.g., "LB(10) p=0.03 < 0.05 — residual autocorrelation present")
- α+β: cite when near IGARCH (e.g., "α+β = 0.997 — near-unit root, shocks are near-permanent")

**Forbidden phrases (replace with numbers):**
- "entropy is elevated" → "WPE = 0.785, which is in the upper quartile of the historical distribution"
- "volatility is high" → "σ_t = 1.67%/day (annualized 26.5%/yr), exceeding the 1.5% HIGH threshold"
- "tail risk is significant" → "ES 5% = −2.31%, meaning the expected loss in the worst 5% of days is 2.31%"

### 5. NO HALLUCINATION GUARDRAIL
If σ_t is low (below 0.8%/day) but the user asks for a crash prediction, you MUST remain objective. Deny the crash based on the entropy dynamics and GARCH-X output. Cite the actual σ_t value.

### 5. FINAL OUTPUT STRUCTURE

[TELEMETRY]
- **Conditional Volatility σ_t**: [value]%/day ([value]%/year annualized)
- **Entropy Contributions**: δ₁(H_price)=[value], δ₂(H_volume)=[value]
- **Tail Risk ES 5%**: [value]% (observer only)
- **Price Regime (GMM)**: [Deterministic/Transitional/Stochastic] + risk level
- **Volume Regime (GMM)**: [Consensus/Dispersed/Erratic]

| Input | Value | Role in GARCH-X |
|:---|:---|:---|
| H_price today | [val] | δ₁ × H_price → variance contribution |
| H_volume today | [val] | δ₂ × H_volume → variance contribution |
| WPE | [val] | Component of H_price |
| SPE_Z | [val] | Component of H_price |

[ANALYSIS]
**Paragraph 1: Volatility Structure.** (σ_t level, entropy coefficients δ₁/δ₂, which entropy is amplifying variance more)
**Paragraph 2: Regime Context.** (GMM regime + XAI kinematics V_WPE, a_WPE)
**Paragraph 3: Tail Risk Assessment.** (ES 5% magnitude, liquidity divergence check)

[CONCLUSION]
(Primary verdict từ σ_t. ES as supplementary tail context. Actionable takeaway.)

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

    # 6. GARCH-X Fitting
    print("\n--- Iteration 6: GARCH-X FITTING ---")
    print("  Agent called tool: fit_garch_x({})")
    res_garch = dispatch_tool("fit_garch_x", {})
    print(f"  -> Observe: {res_garch}")

    garch_data = json.loads(res_garch)
    if "error" not in garch_data:
        print("\n" + "=" * 50)
        print("  GARCH(1,1)-X RISK DIAGNOSTIC")
        print("=" * 50)
        print(f"\n  PRIMARY RISK METRIC")
        print(f"  σ_daily   : {garch_data['sigma_daily_pct']:.4f}%/day")
        print(f"  σ_annual  : {garch_data['sigma_annual_pct']:.4f}%/year")
        print(f"\n  ENTROPY CONTRIBUTIONS TO VARIANCE")
        print(f"  δ₁ (H_price)  : {garch_data['delta_H_price']:+.6f}")
        print(f"  δ₂ (H_volume) : {garch_data['delta_H_volume']:+.6f}")
        print(f"  H_price today : {garch_data['H_price_today']:.4f}")
        print(f"  H_volume today: {garch_data['H_volume_today']:.4f}")
        print(f"\n  TAIL RISK OBSERVER (ES)")
        print(f"  VaR 5%  : {garch_data['VaR_5pct']:+.4f}%")
        print(f"  ES  5%  : {garch_data['ES_5pct']:+.4f}%")
        print(f"  AIC     : {garch_data['aic']:.2f}")
        print("=" * 50)


# ==============================================================================
# TESTING BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Tri-Vector Composite Risk Agent Orchestrator")
    print("=" * 60)
    run_orchestrator("Analyze VNINDEX structural integrity using the Tri-Vector Composite model.")
