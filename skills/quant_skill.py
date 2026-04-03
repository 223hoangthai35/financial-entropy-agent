"""
Quantitative Physics Engine -- Financial Entropy Agent
WPE, Statistical Complexity, MFI, Cross-Sectional Entropy, CECP Boundary.
Toi uu bang @numba.njit va numpy vectorized.
"""

import numpy as np
import numba
import pandas as pd
from math import factorial


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
# CECP BOUNDARY CURVES (LOPEZ-RUIZ)
# ==============================================================================
def generate_cecp_boundary(
    m: int = 3,
) -> tuple[list, list, list, list]:
    """
    Tao Upper/Lower bound cho Complexity-Entropy Causality Plane.
    Returns: (H_upper, C_upper, H_lower, C_lower).
    """
    N = factorial(m)
    U = 1.0 / N
    S_U = np.log(N)

    P_s0 = (1.0 + U) / 2.0
    P_sr = U / 2.0
    S_star = -(P_s0 * np.log(P_s0)) - (N - 1) * (P_sr * np.log(P_sr))
    D_max = S_star - 0.5 * S_U
    Q0 = 1.0 / D_max if D_max > 0 else 0.0

    def _h_c(p_dist: np.ndarray) -> tuple[float, float]:
        P = p_dist[p_dist > 0]
        if len(P) == 0:
            return 0.0, 0.0
        S_P = -np.sum(P * np.log(P))
        H_P = S_P / S_U
        P_mid = (p_dist + U) / 2.0
        S_mid = -np.sum(P_mid * np.log(P_mid))
        JSD = S_mid - 0.5 * S_P - 0.5 * S_U
        return H_P, Q0 * JSD * H_P

    # Upper bound
    H_upper, C_upper = [], []
    for p_max in np.linspace(1.0 / N, 1.0, 200):
        dist = np.full(N, (1.0 - p_max) / (N - 1))
        dist[0] = p_max
        h, c = _h_c(dist)
        H_upper.append(h)
        C_upper.append(c)

    # Lower bound
    H_lower, C_lower = [], []
    for k in range(1, N):
        for p in np.linspace(1.0 / k, 1.0 / (k + 1), 50):
            dist = np.zeros(N)
            dist[:k] = p
            dist[k] = 1.0 - k * p
            h, c = _h_c(dist)
            H_lower.append(h)
            C_lower.append(c)

    h, c = _h_c(np.full(N, 1.0 / N))
    H_lower.append(h)
    C_lower.append(c)

    return H_upper, C_upper, H_lower, C_lower


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
    print("TEST 4: CECP Boundary (m=3)")
    print("=" * 60)
    Hu, Cu, Hl, Cl = generate_cecp_boundary(m=3)
    print(f"  Upper points : {len(Hu)}")
    print(f"  Lower points : {len(Hl)}")
    print(f"  H range      : [{min(Hu):.4f}, {max(Hu):.4f}]")
    print(f"  C range      : [{min(Cu):.4f}, {max(Cu):.4f}]")
