"""
Quantitative Physics Engine -- Financial Entropy Agent
WPE, Statistical Complexity, MFI, Cross-Sectional Entropy,
Volume Entropy (Shannon, SampEn), Price Sample Entropy (SPE_Z),
WPE Kinematics (V_WPE, a_WPE) for XAI trajectory indicators.
Toi uu bang @numba.njit va numpy vectorized.
"""

import numpy as np
import numba
import pandas as pd


# ==============================================================================
# NUMBA CORE: LEHMER CODE
# ==============================================================================
@numba.njit(cache=True)
def _perm_to_index(perm: np.ndarray, m: int) -> int:
    """Chuyen permutation thanh index duy nhat qua Lehmer code. Range [0, m!)."""
    index = 0
    for i in range(m):
        count = 0
        for j in range(i + 1, m):
            if perm[j] < perm[i]:
                count += 1
        fact = 1
        for k in range(1, m - i):
            fact *= k
        index += count * fact
    return index


# ==============================================================================
# NUMBA CORE: WPE + COMPLEXITY (SINGLE WINDOW)
# ==============================================================================
@numba.njit(cache=True)
def _calc_wpe_complexity_jit(x: np.ndarray, m: int, tau: int) -> tuple:
    """
    Tinh WPE (H) va Jensen-Shannon Complexity (C) cho 1 chuoi.
    Cong thuc: H(WPE) = -1/ln(m!) * SUM(p_w * ln(p_w))
               C = Q0 * JSD(P, U) * H
    """
    N = len(x)
    n_patterns = N - (m - 1) * tau
    if n_patterns <= 0:
        return np.nan, np.nan

    n_states = 1
    for i in range(2, m + 1):
        n_states *= i

    # Tich luy weighted frequency
    w_accum = np.zeros(n_states)

    for i in range(n_patterns):
        vec = np.empty(m)
        for j in range(m):
            vec[j] = x[i + j * tau]

        perm = np.argsort(vec)

        # Weight = variance cua amplitude vector
        mean_v = 0.0
        for j in range(m):
            mean_v += vec[j]
        mean_v /= m
        var_v = 0.0
        for j in range(m):
            var_v += (vec[j] - mean_v) ** 2
        var_v /= m

        idx = _perm_to_index(perm, m)
        w_accum[idx] += var_v

    total_w = 0.0
    for i in range(n_states):
        total_w += w_accum[i]
    if total_w <= 0.0:
        return np.nan, np.nan

    # Normalized distribution
    p_dist = np.empty(n_states)
    for i in range(n_states):
        p_dist[i] = w_accum[i] / total_w

    # Shannon Entropy cua P
    S_P = 0.0
    for i in range(n_states):
        if p_dist[i] > 0.0:
            S_P -= p_dist[i] * np.log(p_dist[i])
    S_U = np.log(n_states)
    H_P = S_P / S_U

    # Jensen-Shannon Divergence
    U = 1.0 / n_states
    S_mid = 0.0
    for i in range(n_states):
        p_mid = (p_dist[i] + U) / 2.0
        if p_mid > 0.0:
            S_mid -= p_mid * np.log(p_mid)
    JSD = S_mid - 0.5 * S_P - 0.5 * S_U

    # Q0 normalization constant
    P_star_0 = (1.0 + U) / 2.0
    P_star_rest = U / 2.0
    S_star = -(P_star_0 * np.log(P_star_0)) - \
             (n_states - 1) * (P_star_rest * np.log(P_star_rest))
    D_max = S_star - 0.5 * S_U
    Q0 = 1.0 / D_max if D_max > 0.0 else 0.0

    C_JS = Q0 * JSD * H_P
    return H_P, C_JS


# ==============================================================================
# NUMBA CORE: ROLLING WPE
# ==============================================================================
@numba.njit(cache=True)
def calc_rolling_wpe(
    log_returns: np.ndarray,
    m: int,
    tau: int,
    window: int,
) -> tuple:
    """
    Ap dung WPE + Complexity tren sliding window.
    Output: (wpe_array, complexity_array) -- cung shape voi input.
    """
    n = len(log_returns)
    wpe_out = np.full(n, np.nan)
    c_out = np.full(n, np.nan)

    for i in range(window, n):
        raw = log_returns[i - window: i]
        valid = np.empty(window)
        count = 0
        for j in range(window):
            if np.isfinite(raw[j]):
                valid[count] = raw[j]
                count += 1

        if count >= m:
            h, c = _calc_wpe_complexity_jit(valid[:count], m, tau)
            wpe_out[i] = h
            c_out[i] = c

    return wpe_out, c_out


# ==============================================================================
# PUBLIC API
# ==============================================================================
def calc_wpe_complexity(
    x: np.ndarray, m: int = 3, tau: int = 1,
) -> tuple[float, float]:
    """Public wrapper: tinh WPE va Complexity cho 1 mang. Returns (H, C)."""
    return _calc_wpe_complexity_jit(np.asarray(x, dtype=np.float64), m, tau)


def calc_mfi(wpe: np.ndarray, complexity: np.ndarray) -> np.ndarray:
    """Market Fragility Index: MFI = WPE * (1 - C). Vectorized."""
    return wpe * (1.0 - complexity)


# ==============================================================================
# CROSS-SECTIONAL CORRELATION ENTROPY (VN30 EVD)
# ==============================================================================
def calc_correlation_entropy(
    df_returns: pd.DataFrame, window: int = 22,
) -> pd.Series:
    """
    S_corr = -(SUM(p_i * ln(p_i)) / ln(M)) * 100
    voi p_i = lambda_i / SUM(lambda_j) tu EVD cua Pearson Correlation Matrix.
    Output: Series 0-100. Thap (<40) = consensus, Cao (>70) = fragmented.
    """
    n_days = len(df_returns)
    corr_entropy = pd.Series(index=df_returns.index, dtype="float64")

    for i in range(window, n_days):
        window_rets = df_returns.iloc[i - window: i]
        corr_matrix = window_rets.corr().values
        corr_matrix = np.nan_to_num(corr_matrix)

        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        total_var = eigenvalues.sum()
        if total_var == 0:
            continue

        p_i = eigenvalues / total_var
        p_i = p_i[p_i > 0]

        max_entropy = np.log(len(eigenvalues))
        if max_entropy > 0:
            corr_entropy.iloc[i] = (
                -np.sum(p_i * np.log(p_i)) / max_entropy
            ) * 100.0

    return corr_entropy



# ==============================================================================
# NUMBA CORE: SAMPLE ENTROPY (SampEn)
# ==============================================================================
@numba.njit(cache=True)
def _calc_sample_entropy_jit(x: np.ndarray, m: int, r: float) -> float:
    """
    SampEn(m, r) = -ln(A / B).
    B = so cap template match o chieu m. A = so cap match o chieu m+1.
    Complexity O(N^2) -- bat buoc JIT.
    """
    N = len(x)
    if N < m + 2:
        return np.nan

    B = 0  # matches tai embedding dim m
    A = 0  # matches tai embedding dim m+1

    for i in range(N - m):
        for j in range(i + 1, N - m):
            # Kiem tra m-length match
            match_m = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r:
                    match_m = False
                    break
            if match_m:
                B += 1
                # Kiem tra (m+1)-length match (chi khi m-match da thanh cong)
                if i + m < N and j + m < N:
                    if abs(x[i + m] - x[j + m]) <= r:
                        A += 1

    if B == 0:
        return np.nan
    if A == 0:
        # Khong co match nao o m+1 -> entropy cuc dai (maximal irregularity)
        return np.nan

    return -np.log(np.float64(A) / np.float64(B))


def calc_sample_entropy(
    x: np.ndarray, m: int = 2, r: float | None = None,
) -> float:
    """
    Public wrapper: tinh Sample Entropy cho 1 mang.
    Neu r=None, tu dong tinh r = 0.2 * std(x).
    Returns float (NaN neu input khong hop le hoac zero-variance).
    """
    arr = np.asarray(x, dtype=np.float64)
    if len(arr) < m + 2:
        return np.nan
    if r is None:
        std = np.std(arr)
        if std == 0.0:
            return np.nan
        r = 0.2 * std
    return _calc_sample_entropy_jit(arr, m, r)


# ==============================================================================
# VOLUME SHANNON ENTROPY (HISTOGRAM-BASED)
# ==============================================================================
def calc_shannon_entropy_hist(
    x: np.ndarray, bins: str | int = "auto",
) -> float:
    """
    Shannon Entropy cua phan phoi histogram: H = -SUM(p_i * ln(p_i)) / ln(n_bins).
    Normalized ve [0, 1]. bins='auto' ap dung Freedman-Diaconis / Sturges rule
    de xu ly phan phoi heavy-tailed cua volume data (tranh empty bins).
    Returns float [0, 1]. 0 = tap trung hoan toan, 1 = phan tan deu.
    """
    arr = np.asarray(x, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if len(valid) < 2:
        return np.nan

    counts, _ = np.histogram(valid, bins=bins)
    # Loai bo bins trong
    counts_nz = counts[counts > 0]
    n_bins = len(counts)

    if n_bins <= 1:
        return 0.0

    p = counts_nz / counts_nz.sum()
    H = -np.sum(p * np.log(p)) / np.log(n_bins)
    return float(np.clip(H, 0.0, 1.0))


# ==============================================================================
# MACRO-MICRO FUSION: VOLUME ENTROPY PIPELINE
# ==============================================================================
def calc_rolling_volume_entropy(
    volume: np.ndarray, window: int = 60, z_window: int = 252,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Macro-Micro Fusion Volume Entropy Pipeline.

    Pipeline:
      1. Base: log_vol = log1p(volume)
      2. Path A (Macro): Vol_Global_Z = global z-score cua log_vol.
         -> Do luong quy mo thanh khoan tuyet doi (macro inflow/drought).
      3. Path B (Micro): Vol_Rolling_Z = rolling z-score (z_window ngay) cua log_vol.
         -> Feed vao Entropy de phat hien hanh vi giao dich cuc bo,
            khang structural break.
      4. Shannon + SampEn tinh TREN Vol_Rolling_Z (KHONG phai Global Z).
         SampEn tolerance r = 0.2 (co dinh, vi rolling z-score co unit variance).

    Returns: (shannon_arr, sampen_arr, vol_global_z, vol_rolling_z).
    """
    vol = np.asarray(volume, dtype=np.float64)
    n = len(vol)
    shannon_out = np.full(n, np.nan)
    sampen_out = np.full(n, np.nan)

    # --- Base Transform ---
    log_vol = np.log1p(np.maximum(vol, 0.0))

    # --- Path A: Macro Scale (Global Z-Score) ---
    log_vol_series = pd.Series(log_vol)
    global_mean = log_vol_series.mean()
    global_std = log_vol_series.std()
    if global_std > 0:
        vol_global_z = ((log_vol_series - global_mean) / global_std).values
    else:
        vol_global_z = np.zeros(n)

    # --- Path B: Micro Structure (Rolling Z-Score, z_window ngay) ---
    rolling_mean = log_vol_series.rolling(z_window, min_periods=1).mean()
    rolling_std = log_vol_series.rolling(z_window, min_periods=1).std()
    # Tranh chia cho 0: thay std=0 bang NaN de skip
    rolling_std = rolling_std.replace(0, np.nan)
    vol_rolling_z = ((log_vol_series - rolling_mean) / rolling_std).values

    # --- Entropy tren Micro Path (Vol_Rolling_Z) ---
    for i in range(window, n):
        segment = vol_rolling_z[i - window: i]

        # Loc NaN/Inf
        valid = segment[np.isfinite(segment)]
        if len(valid) < 10:
            continue

        # Shannon Entropy (bins='auto')
        shannon_out[i] = calc_shannon_entropy_hist(valid, bins="auto")

        # Sample Entropy (m=2, r=0.2 co dinh -- unit variance input)
        sampen_out[i] = _calc_sample_entropy_jit(valid, 2, 0.2)

    return shannon_out, sampen_out, vol_global_z, vol_rolling_z


# ==============================================================================
# WPE KINEMATICS (DECOUPLED -- XAI TRAJECTORY INDICATORS ONLY)
# ==============================================================================
def calc_wpe_kinematics(
    wpe: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kinematic descriptors of WPE trajectory (XAI only, NOT used in ML).
    V_WPE = Delta(WPE) (Velocity of entropy change).
    a_WPE = Delta(V) (Acceleration of entropy change).
    Returns: (velocity, acceleration)
    """
    wpe = np.asarray(wpe, dtype=np.float64)
    n = len(wpe)

    velocity = np.full(n, np.nan)
    acceleration = np.full(n, np.nan)

    # V = WPE(t) - WPE(t-1)
    velocity[1:] = np.diff(wpe)

    # a = V(t) - V(t-1)
    valid_v = velocity.copy()
    acceleration[2:] = np.diff(valid_v[1:])

    return velocity, acceleration


# ==============================================================================
# PRICE SAMPLE ENTROPY (SPE_Z) -- PLANE 1 Y-AXIS
# ==============================================================================
def calc_rolling_price_sample_entropy(
    close_prices: np.ndarray,
    window: int = 60,
    m: int = 2,
    r_factor: float = 0.2,
) -> np.ndarray:
    """
    Rolling Sample Entropy tren close prices (KHONG phai log-returns).
    r = r_factor * std(window). Dung de do do phuc tap cua quy dao gia.
    Returns: sampen_array (cung shape voi input, NaN cho cac diem dau).
    """
    prices = np.asarray(close_prices, dtype=np.float64)
    n = len(prices)
    sampen_out = np.full(n, np.nan)

    for i in range(window, n):
        segment = prices[i - window: i]
        valid = segment[np.isfinite(segment)]
        if len(valid) < m + 2:
            continue
        std_val = np.std(valid)
        if std_val == 0.0:
            continue
        r = r_factor * std_val
        sampen_out[i] = _calc_sample_entropy_jit(valid, m, r)

    return sampen_out


def calc_spe_z(sampen_arr: np.ndarray) -> np.ndarray:
    """
    Global Z-Score normalization cua Price Sample Entropy.
    SPE_Z = (SampEn - mean) / std. Chuan hoa toan cuc de tao
    orthogonal axis voi WPE trong Phase Space.
    Returns: spe_z_array (NaN-safe).
    """
    arr = np.asarray(sampen_arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if len(valid) < 2:
        return np.full_like(arr, np.nan)
    mu = np.mean(valid)
    sigma = np.std(valid)
    if sigma == 0.0:
        return np.full_like(arr, 0.0)
    spe_z = np.where(np.isfinite(arr), (arr - mu) / sigma, np.nan)
    return spe_z


# ==============================================================================
# TESTING BLOCK
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("TEST 1: WPE + Complexity (single array, random)")
    print("=" * 60)
    dummy = np.random.randn(100)
    h, c = calc_wpe_complexity(dummy, m=3, tau=1)
    mfi = h * (1.0 - c)
    print(f"  H_wpe  = {h:.6f}")
    print(f"  C_js   = {c:.6f}")
    print(f"  MFI    = {mfi:.6f}")

    print("\n" + "=" * 60)
    print("TEST 2: WPE Kinematics (XAI Trajectory Indicators)")
    print("=" * 60)
    test_wpe = np.array([0.50, 0.52, 0.55, 0.60, 0.58, 0.55, 0.50, 0.48, 0.52, 0.56])
    vel, acc = calc_wpe_kinematics(test_wpe)
    print(f"  WPE Input      : {test_wpe}")
    print(f"  V_WPE          : {np.round(vel, 5)}")
    print(f"  a_WPE          : {np.round(acc, 5)}")

    print("\n" + "=" * 60)
    print("TEST 3: Price Sample Entropy (SPE_Z)")
    print("=" * 60)
    fake_prices = np.cumsum(np.random.randn(200) * 2.0) + 1200
    sampen_price = calc_rolling_price_sample_entropy(fake_prices, window=60)
    spe_z = calc_spe_z(sampen_price)
    valid_spe = spe_z[np.isfinite(spe_z)]
    print(f"  Price points       : {len(fake_prices)}")
    print(f"  Valid SPE_Z points : {len(valid_spe)}")
    if len(valid_spe) > 0:
        print(f"  SPE_Z range        : [{np.min(valid_spe):.3f}, {np.max(valid_spe):.3f}]")
        print(f"  SPE_Z mean         : {np.mean(valid_spe):.3f}")
        print(f"  SPE_Z std          : {np.std(valid_spe):.3f}")

    print("\n" + "=" * 60)
    print("TEST 4: Rolling WPE -> Kinematics Pipeline")
    print("=" * 60)
    log_rets = np.random.randn(200) * 0.01
    wpe_arr, c_arr = calc_rolling_wpe(log_rets, m=3, tau=1, window=22)
    valid_wpe = wpe_arr[~np.isnan(wpe_arr)]
    if len(valid_wpe) > 5:
        v, a = calc_wpe_kinematics(valid_wpe)
        print(f"  Valid WPE points : {len(valid_wpe)}")
        print(f"  V_WPE range      : [{np.nanmin(v):.5f}, {np.nanmax(v):.5f}]")
        print(f"  a_WPE range      : [{np.nanmin(a):.5f}, {np.nanmax(a):.5f}]")
    else:
        print("  Khong du du lieu WPE.")

