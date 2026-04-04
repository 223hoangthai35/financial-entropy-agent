"""
Quantitative Physics Engine -- Financial Entropy Agent
WPE, Statistical Complexity, MFI, Cross-Sectional Entropy,
Volume Entropy (Shannon, SampEn).
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
    print(f"  H_wpe  = {h:.6f}  (random -> ky vong ~0.9+)")
    print(f"  C_js   = {c:.6f}  (random -> ky vong thap)")
    print(f"  MFI    = {mfi:.6f}")
    print(f"  Types  : H={type(h).__name__}, C={type(c).__name__}")

    print()
    print("=" * 60)
    print("TEST 2: Rolling WPE (shape verification)")
    print("=" * 60)
    series = np.random.randn(200)
    wpe_arr, c_arr = calc_rolling_wpe(series, m=3, tau=1, window=22)
    mfi_arr = calc_mfi(wpe_arr, c_arr)
    print(f"  Input shape : {series.shape}")
    print(f"  WPE shape   : {wpe_arr.shape}")
    print(f"  MFI shape   : {mfi_arr.shape}")
    print(f"  NaN count   : WPE={np.isnan(wpe_arr).sum()}, C={np.isnan(c_arr).sum()}")
    print(f"  Last 5 WPE  : {wpe_arr[-5:]}")
    print(f"  Last 5 C    : {c_arr[-5:]}")

    print()
    print("=" * 60)
    print("TEST 3: Cross-Sectional Correlation Entropy")
    print("=" * 60)
    fake_rets = pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f"S{i}" for i in range(10)],
    )
    corr_ent = calc_correlation_entropy(fake_rets, window=22)
    valid = corr_ent.dropna()
    print(f"  Input shape  : {fake_rets.shape}")
    print(f"  Valid values : {len(valid)}")
    print(f"  Mean entropy : {valid.mean():.2f} / 100")
    print(f"  Last 3       : {valid.tail(3).values}")

    print()
    print("=" * 60)
    print("TEST 4: Sample Entropy (SampEn)")
    print("=" * 60)
    # Random -> SampEn cao (bat quy luat)
    rand_sig = np.random.randn(200)
    se_rand = calc_sample_entropy(rand_sig, m=2)
    # Periodic -> SampEn thap (co quy luat)
    periodic_sig = np.sin(np.linspace(0, 20 * np.pi, 200))
    se_periodic = calc_sample_entropy(periodic_sig, m=2)
    print(f"  SampEn (random)   = {se_rand:.4f}  (ky vong cao, ~2.0+)")
    print(f"  SampEn (periodic) = {se_periodic:.4f}  (ky vong thap, <1.0)")
    print(f"  Validation        : random > periodic = {se_rand > se_periodic}")

    print()
    print("=" * 60)
    print("TEST 6: Shannon Entropy (Histogram, bins='auto')")
    print("=" * 60)
    uniform_data = np.random.uniform(0, 100, 200)
    concentrated = np.concatenate([np.full(180, 50.0), np.random.randn(20)])
    h_uniform = calc_shannon_entropy_hist(uniform_data, bins="auto")
    h_concentrated = calc_shannon_entropy_hist(concentrated, bins="auto")
    print(f"  H (uniform)      = {h_uniform:.4f}  (ky vong ~1.0)")
    print(f"  H (concentrated) = {h_concentrated:.4f}  (ky vong thap, <0.5)")
    print(f"  Validation       : uniform > concentrated = {h_uniform > h_concentrated}")

    print()
    print("=" * 60)
    print("TEST 7: Macro-Micro Fusion Volume Entropy (window=60)")
    print("=" * 60)
    fake_volume = np.abs(np.random.randn(300)) * 1e6 + 5e5
    sh_arr, se_arr, gz_arr, rz_arr = calc_rolling_volume_entropy(fake_volume, window=60)
    print(f"  Input shape       : {fake_volume.shape}")
    print(f"  Shannon shape     : {sh_arr.shape}")
    print(f"  SampEn shape      : {se_arr.shape}")
    print(f"  Global Z shape    : {gz_arr.shape}")
    print(f"  Rolling Z shape   : {rz_arr.shape}")
    print(f"  NaN count         : Shannon={np.isnan(sh_arr).sum()}, SampEn={np.isnan(se_arr).sum()}")
    valid_sh = sh_arr[np.isfinite(sh_arr)]
    valid_se = se_arr[np.isfinite(se_arr)]
    valid_gz = gz_arr[np.isfinite(gz_arr)]
    valid_rz = rz_arr[np.isfinite(rz_arr)]
    print(f"  Shannon range     : [{valid_sh.min():.4f}, {valid_sh.max():.4f}]")
    print(f"  SampEn range      : [{valid_se.min():.4f}, {valid_se.max():.4f}]")
    print(f"  Global Z range    : [{valid_gz.min():.4f}, {valid_gz.max():.4f}]")
    print(f"  Rolling Z range   : [{valid_rz.min():.4f}, {valid_rz.max():.4f}]")
    print(f"  Last 5 Global Z   : {gz_arr[-5:]}")
    print(f"  Last 5 Rolling Z  : {rz_arr[-5:]}")
