"""
lam_engine.py
=============
Reference (NumPy) LAM engine + closed-form theory, valid for any L >= 2.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf as sp_erf
from scipy.special import gammaln, logsumexp


# ===========================================================================
# Numerical & information-theoretic primitives
# ===========================================================================
def _as_float_or_array(x):
    arr = np.asarray(x, dtype=np.float64)
    return float(arr.item()) if arr.ndim == 0 else arr


def _logcosh(x):
    """Overflow-safe log(cosh(x)) without small-x cancellation."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    small = ax < 10.0
    ax_s = np.where(small, ax, 0.0)
    res = np.where(
        small,
        0.5 * np.log1p(np.sinh(ax_s) ** 2),
        ax + np.log1p(np.exp(-2.0 * ax)) - np.log(2.0),
    )
    return float(res.item()) if np.ndim(x) == 0 else res


def _log_abs_sinh(x):
    """Overflow-safe log(|sinh(x)|) returning -inf at x = 0."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    small = ax < 10.0
    ax_s = np.where(small, ax, 1.0)
    with np.errstate(divide="ignore"):
        log_small = np.log(np.sinh(ax_s))
    log_large = ax - np.log(2.0) + np.log1p(-np.exp(-2.0 * np.maximum(ax, 10.0)))
    res = np.where(ax == 0.0, -np.inf, np.where(small, log_small, log_large))
    return float(res.item()) if np.ndim(x) == 0 else res


def _log_binom_pmf(n: int) -> np.ndarray:
    k = np.arange(n + 1, dtype=np.float64)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) - n * np.log(2.0)


def _iter_index_chunks(num_values: int, dim: int, max_chunk: int = 200_000):
    """Yield `(chunk_len, dim)` index arrays partitioning `num_values**dim` tuples."""
    total = num_values**dim
    shape = (num_values,) * dim
    for start in range(0, total, max_chunk):
        end = min(start + max_chunk, total)
        lin = np.arange(start, end, dtype=np.int64)
        yield np.stack(np.unravel_index(lin, shape), axis=1)


def I_R(m):
    """Symmetric Bernoulli rate function 0.5*[(1+m)ln(1+m) + (1-m)ln(1-m)]."""
    arr = np.asarray(m, dtype=np.float64)
    abs_m = np.abs(arr)
    if np.any(abs_m > 1.0):
        raise ValueError("I_R requires |m| <= 1.")
    safe = np.clip(abs_m, 0.0, 1.0 - 1e-15)
    val = 0.5 * ((1.0 + safe) * np.log1p(safe) + (1.0 - safe) * np.log1p(-safe))
    res = np.where(abs_m == 1.0, np.log(2.0), val)
    return float(res.item()) if np.ndim(m) == 0 else res


# ===========================================================================
# Asymptotic stability theory & Proposition 1 (bounds and exact finite-N)
# ===========================================================================
def saddle_point(L: int) -> dict:
    """Annealed saddle point for L layers. Requires L >= 2."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    a = 2.0 * (L - 1)
    x_star = float(brentq(lambda x: np.tanh(x) - x / a, 1e-6, 4.0 * (L - 1) + 40.0))
    m_star = float(np.tanh(x_star))
    phi_L = -(x_star**2) / (4.0 * (L - 1)) + float(_logcosh(x_star))
    rho_L = L * ((L - 1) - phi_L)
    r_c = float(np.sqrt(max(1.0 - rho_L / (L * (L - 1)), 0.0)))
    return dict(x_star=x_star, m_star=m_star, phi_L=phi_L, rho_L=rho_L, r_c=r_c)


def mu1_signal(L: int) -> float:
    """Asymptotic one-step signal strength e^{-(L-1)} sinh(L-1)."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    return float(0.5 * (1.0 - np.exp(-2.0 * (L - 1))))


def mu1_signal_finiteN(L: int, N: int) -> float:
    """Exact finite-N one-step signal strength s_N (Proposition 1)."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    if N < 1:
        raise ValueError("N must be >= 1.")
    q = (N - 1) / N
    log_env = -(L - 1) / N - (L - 1) * (L - 2) / (2.0 * N)
    return float(0.5 * np.exp(log_env) * (1.0 - np.exp(-2.0 * (L - 1) * q)))


s_N = mu1_signal_finiteN


def K_L_prefactor(L: int) -> float:
    """Asymptotic noise prefactor K_L = (2 / sqrt(det H_L*)) sinh^2 cosh^{L-1}."""
    sp = saddle_point(L)
    x_star = sp["x_star"]
    m_star = sp["m_star"]
    lcosh2 = 2.0 * float(_logcosh(x_star))
    inv_c2 = np.exp(-lcosh2)
    log_det = (lcosh2 + np.log1p(-2.0 * (L - 1) * inv_c2)) + (L - 1) * (
        lcosh2 + np.log1p(2.0 * inv_c2)
    )
    log_sinh = _log_abs_sinh((L - 1) * m_star)
    log_KL = (
        np.log(2.0)
        - 0.5 * log_det
        + 2.0 * log_sinh
        + (L - 1) * _logcosh(2.0 * (L - 1) * m_star)
    )
    return float(np.exp(log_KL))


def noise_var_per_pattern_annealed(N: int, L: int) -> float:
    sp = saddle_point(L)
    return float(K_L_prefactor(L) * np.exp(-N * sp["rho_L"]))


def log_v_N(L: int, N: int) -> float:
    """Log exact finite-N per-pattern noise variance v_N (Proposition 1)."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    if N < 1:
        raise ValueError("N must be >= 1.")
    if N == 1:
        return -np.inf
    Nm1 = N - 1
    lp = _log_binom_pmf(Nm1)
    mh = (2.0 * np.arange(N, dtype=np.float64) - Nm1) / N
    b = L * (L - 1) / 2.0
    chunk_logs = []
    for K in _iter_index_chunks(N, L):
        m = mh[K]
        lpk = lp[K].sum(axis=1)
        E = 0.5 * N * (m.sum(axis=1) ** 2 - (m * m).sum(axis=1)) - N * b
        F = m.sum(axis=1, keepdims=True) - m
        t = (
            2.0 * E
            + 2.0 * _log_abs_sinh(F[:, 0])
            + _logcosh(2.0 * F[:, 1:]).sum(axis=1)
        )
        valid = np.isfinite(t)
        if np.any(valid):
            chunk_logs.append(float(logsumexp(lpk[valid] + t[valid])))
    return float(logsumexp(chunk_logs)) if chunk_logs else -np.inf


log_noise_var_ann_exact = log_v_N


def gamma_N_exact(P, N: int, L: int):
    """Exact finite-N reduced noise-to-signal ratio gamma_N(P) = (P-1) v_N / s_N^2."""
    log_s = np.log(mu1_signal_finiteN(L, N))
    log_ratio = log_v_N(L, N) - 2.0 * log_s
    P_arr = np.maximum(np.asarray(P, dtype=np.float64) - 1.0, 0.0)
    res = np.where(P_arr > 0.0, P_arr * np.exp(log_ratio), 0.0)
    return _as_float_or_array(res)


def cantelli_flip_bound(gamma):
    """Proposition 1(i) Cantelli upper bound P(X_i^a <= 0) <= gamma / (1 + gamma)."""
    g = np.maximum(np.asarray(gamma, dtype=np.float64), 0.0)
    return _as_float_or_array(g / (1.0 + g))


def cantelli_overlap_bound(gamma):
    """Proposition 1(i) Cantelli lower bound E[m^(1)] >= (1 - gamma) / (1 + gamma)."""
    g = np.maximum(np.asarray(gamma, dtype=np.float64), 0.0)
    return _as_float_or_array((1.0 - g) / (1.0 + g))


def counting_flip_lower_bound(P, N: int, L: int):
    """Proposition 1(iii) non-asymptotic counting lower bound on single-bit error prob."""
    P_arr = np.asarray(P, dtype=np.float64)
    Pm1 = np.maximum(P_arr - 1.0, 0.0)
    if L * N < 50:
        q = 2.0 ** (-L * N)
        exponent = Pm1 * np.log1p(-2.0 * q)
    else:
        log_lam = np.where(
            Pm1 > 0.0, np.log(np.maximum(Pm1, 1.0)) - L * N * np.log(2.0), -np.inf
        )
        exponent = np.where(Pm1 > 0.0, -2.0 * np.exp(np.minimum(log_lam, 700.0)), 0.0)
    res = np.where(Pm1 > 0.0, -0.25 * np.expm1(exponent), 0.0)
    return _as_float_or_array(res)


# ===========================================================================
# Reduced-load (gamma) parameterisation & log-domain typical noise
# ===========================================================================
def gamma_eff(P, N: int, L: int):
    """Dimensionless noise-to-signal ratio governing the asymptotic one-step transition."""
    sp = saddle_point(L)
    K_L = K_L_prefactor(L)
    mu1_sq = mu1_signal(L) ** 2
    P_arr = np.atleast_1d(np.asarray(P, dtype=np.float64))
    g = np.maximum(P_arr - 1.0, 0.0) * K_L * np.exp(-N * sp["rho_L"]) / mu1_sq
    return float(g.item()) if np.ndim(P) == 0 else g


def P_for_gamma(
    gamma: float, N: int, L: int, P_min: int = 2, P_max: float | None = None
) -> int:
    """Invert gamma_eff: the (rounded) P such that gamma_eff(P, N, L) == gamma."""
    sp = saddle_point(L)
    K_L = K_L_prefactor(L)
    mu1_sq = mu1_signal(L) ** 2
    denom = K_L * np.exp(-N * sp["rho_L"])
    P = 1.0 + gamma * mu1_sq / max(denom, 1e-300)
    if P_max is not None:
        P = min(P, P_max)
    return max(P_min, int(round(P)))


def m1_theory_gamma(gamma):
    """Asymptotic one-step overlap as a pure function of the reduced load."""
    g = np.maximum(np.asarray(gamma, dtype=np.float64), 1e-300)
    return _as_float_or_array(sp_erf(1.0 / np.sqrt(2.0 * g)))


def N_for_log_capacity(L: int, log_P_target: float, N_min: int = 6) -> int:
    """Smallest N (floored at N_min) such that ln P_c(N,L) = N rho_L is approx log_P_target."""
    sp = saddle_point(L)
    N = int(round(log_P_target / sp["rho_L"]))
    return max(N_min, N)


def log_noise_var_typ(L: int, N: int, n_calib: int = 20000, seed: int = 42) -> float:
    """Log-domain Monte Carlo estimate of typical per-pattern noise variance."""
    rng = np.random.default_rng(seed)
    xi1 = rng.choice([-1, 1], size=(L, N)).astype(np.int8)
    sigma = xi1.copy()
    binomL2 = L * (L - 1) // 2
    xi_mu = rng.choice([-1, 1], size=(n_calib, L, N)).astype(np.int8)
    a = 0
    mhat = (xi_mu[:, :, 1:] * sigma[None, :, 1:]).sum(axis=2) / N
    sum_m = mhat.sum(axis=1)
    sum_m2 = (mhat**2).sum(axis=1)
    Ehat = 0.5 * N * (sum_m**2 - sum_m2) - N * binomL2
    F = mhat.sum(axis=1, keepdims=True) - mhat
    S = (xi_mu[:, :, 0] * sigma[None, :, 0] * F).sum(axis=1)
    logPhi = S - xi_mu[:, a, 0] * sigma[a, 0] * F[:, a]
    lx2 = 2.0 * (Ehat + logPhi + _log_abs_sinh(F[:, a]))
    valid = np.isfinite(lx2)
    return float(logsumexp(lx2[valid]) - np.log(n_calib)) if np.any(valid) else -np.inf


def noise_var_per_pattern_typical(
    L: int, N: int, n_calib: int = 20000, seed: int = 42
) -> float:
    """Monte Carlo estimate of typical per-pattern noise variance (may underflow for large N)."""
    return float(np.exp(log_noise_var_typ(L, N, n_calib=n_calib, seed=seed)))


# ===========================================================================
# Corrupted-basin saddle, rates, and thresholds
# ===========================================================================
def corrupted_saddle(L: int, r: float) -> float:
    """Tilted saddle overlap m_s(r) solving m = tanh((L-1)m + arctanh(r))."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    if r >= 1.0 - 1e-12:
        return 1.0
    a = float(np.arctanh(np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12)))
    m = float(max(r, 0.5))
    for _ in range(1000):
        mn = float(np.tanh((L - 1) * m + a))
        if abs(mn - m) < 1e-14:
            return mn
        m = mn
    return m


def Sigma_L(L: int, r: float) -> float:
    """Annealed corrupted-signal large-deviation rate Sigma_L(r)."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    if abs(r) >= 1.0 - 1e-12:
        return 0.0
    a = float(np.arctanh(r))
    ms = corrupted_saddle(L, r)
    return float(
        0.5 * L * (L - 1) * (1.0 + ms * ms)
        - L * (_logcosh((L - 1) * ms + a) - _logcosh(a))
    )


def C_L_prefactor(L: int, r: float) -> float:
    """Corrupted-signal Laplace prefactor C_L(r) (Appendix C)."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    if r >= 1.0 - 1e-12:
        return mu1_signal(L)
    if r <= 0.0 and L == 2:
        return 0.0
    a = float(np.arctanh(r))
    ms = corrupted_saddle(L, r)
    F = (L - 1) * ms
    if F == 0.0:
        return 0.0
    lcosh2 = 2.0 * float(_logcosh(F + a))
    inv_c2 = np.exp(-lcosh2)
    log_det = (lcosh2 + np.log1p(-(L - 1) * inv_c2)) + (L - 1) * (
        lcosh2 + np.log1p(inv_c2)
    )
    log_C = (
        _log_abs_sinh(abs(F))
        + (L - 1) * float(_logcosh(F + a))
        + float(_logcosh(a))
        - 0.5 * log_det
    )
    return float(np.sign(F) * np.exp(log_C))


def eps_annealed(L: int, r: float) -> float:
    """Annealed basin capacity exponent rho_L - 2 * Sigma_L(r)."""
    return float(saddle_point(L)["rho_L"] - 2.0 * Sigma_L(L, r))


def eps_typical(L: int, r: float) -> float:
    """Typical basin capacity exponent rho_L - L(L-1)(1 - r^2)."""
    return float(saddle_point(L)["rho_L"] - L * (L - 1) * (1.0 - r * r))


def rc_annealed(L: int) -> float:
    """Annealed basin threshold r_c solving Sigma_L(r_c) = rho_L / 2."""
    rho_L = saddle_point(L)["rho_L"]
    return float(brentq(lambda r: Sigma_L(L, r) - 0.5 * rho_L, 0.01, 0.99))


def rc_typical(L: int) -> float:
    """Typical basin threshold sqrt(1 - rho_L / (L(L-1)))."""
    rho_L = saddle_point(L)["rho_L"]
    return float(np.sqrt(max(1.0 - rho_L / (L * (L - 1)), 0.0)))


def rc_load(L: int, N: int, P: int) -> float:
    """Load-consistent basin threshold solving L * I_R(r_c) = ln(P) / N."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    if P <= 1:
        return 0.0
    alpha = np.log(P) / float(N)
    if alpha >= L * np.log(2.0):
        return 1.0
    return float(brentq(lambda r: L * float(I_R(r)) - alpha, 1e-12, 1.0 - 1e-12))


def theory_basin_asymptotic(
    L: int, P: int, N: int, r: float, regime: str = "annealed"
) -> float:
    """Asymptotic one-step overlap curve erf(mu / sqrt(2 * var)) for corrupted cues."""
    if P <= 1:
        return 1.0 if r > 0 else 0.0
    if regime == "annealed":
        sig_val = C_L_prefactor(L, r) * np.exp(-N * Sigma_L(L, r))
    elif regime == "typical":
        sig_val = np.exp(-0.5 * N * L * (L - 1) * (1.0 - r * r))
    else:
        raise ValueError(
            f"Unknown regime {regime!r}; expected 'annealed' or 'typical'."
        )
    var = (P - 1) * noise_var_per_pattern_annealed(N, L)
    if var <= 0.0:
        return 1.0 if sig_val > 0 else (-1.0 if sig_val < 0 else 0.0)
    return float(sp_erf(sig_val / np.sqrt(2.0 * var)))


# ===========================================================================
# Proposition 2: Cue-retrieval theory
# ===========================================================================
def rho_cue(L: int) -> float:
    """Asymptotic cue-retrieval capacity rate rho_{L-1} (0.0 for L=2)."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    return 0.0 if L == 2 else float(saddle_point(L - 1)["rho_L"])


def log_s_cue(L: int, N: int, r: float = 1.0) -> float:
    """Log deterministic seed signal at cue overlap r."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    if r <= 0.0:
        return -np.inf
    K = L - 1
    return float(0.5 * N * K * (K - 1) * r * r + _log_abs_sinh(K * r))


def log_v_cue(L: int, N: int) -> float:
    """Log exact per-pattern noise variance of the cue-driven seed (Proposition 2)."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    if N < 1:
        raise ValueError("N must be >= 1.")
    K = L - 1
    if K == 1:
        return float(np.log(0.5) + np.log(np.expm1(N * _logcosh(2.0 / N))))
    m1 = (2.0 * np.arange(N + 1, dtype=np.float64) - N) / N
    lp1 = _log_binom_pmf(N)
    chunk_logs = []
    for idx in _iter_index_chunks(N + 1, K):
        m = m1[idx]
        lp = lp1[idx].sum(axis=1)
        S = m.sum(axis=1)
        S2 = (m * m).sum(axis=1)
        pair = 0.5 * (S * S - S2)
        logterm = lp + 2.0 * N * pair + 2.0 * _log_abs_sinh(S)
        valid = np.isfinite(logterm)
        if np.any(valid):
            chunk_logs.append(float(logsumexp(logterm[valid])))
    return float(logsumexp(chunk_logs)) if chunk_logs else -np.inf


def gamma_cue(L: int, N: int, P, r: float = 1.0):
    """Exact finite-N reduced load gamma_N^cue(P) of the cue-driven seed."""
    P_arr = np.maximum(np.asarray(P, dtype=np.float64) - 1.0, 0.0)
    log_ratio = log_v_cue(L, N) - 2.0 * log_s_cue(L, N, r=r)
    res = np.where(P_arr > 0.0, P_arr * np.exp(log_ratio), 0.0)
    return _as_float_or_array(res)


def cue_envelope(gamma):
    """Proposition 2(i) Cantelli lower bound E[m^(seed)] >= 1 - min(g, 2g/(1+g))."""
    g = np.maximum(np.asarray(gamma, dtype=np.float64), 0.0)
    return _as_float_or_array(1.0 - np.minimum(g, 2.0 * g / (1.0 + g)))


def P_for_gamma_cue(
    gamma: float, N: int, L: int, r: float = 1.0, P_min: int = 2
) -> int:
    """Invert gamma_cue: rounded P achieving a target cue reduced load gamma."""
    val = 1.0 + gamma * np.exp(2.0 * log_s_cue(L, N, r=r) - log_v_cue(L, N))
    return max(P_min, int(round(val)))


# ===========================================================================
# Reference dynamics (single state)
# ===========================================================================
def hamiltonian_energy(xi: np.ndarray, sigma: np.ndarray) -> float:
    """Exact multilayer pairwise exponential energy H(sigma) = -N sum_mu exp(N(E2_mu - C(L,2)))."""
    P, L, N = xi.shape
    if sigma.shape != (L, N):
        raise ValueError(f"sigma shape {sigma.shape} != (L,N)=({L},{N})")
    m = (xi.astype(np.float64) * sigma.astype(np.float64)[None, :, :]).sum(axis=2) / N
    S = m.sum(axis=1)
    e2 = 0.5 * (S * S - (m * m).sum(axis=1))
    binomL2 = L * (L - 1) / 2.0
    return float(-N * np.exp(N * (e2 - binomL2)).sum())


def compute_local_field(
    xi: np.ndarray, sigma: np.ndarray, mode: str = "cavity"
) -> np.ndarray:
    """Local field H (shape (L,N)) for a single state sigma.

    mode="cavity": Eq. (local-field) cavity decomposition used throughout the paper.
    mode="exact": exact single-flip energy difference gradient of H(sigma).
    """
    if mode not in ("cavity", "exact"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'cavity' or 'exact'.")
    P, L, N = xi.shape
    if sigma.shape != (L, N):
        raise ValueError(f"sigma shape {sigma.shape} != (L,N)=({L},{N})")
    binomL2 = L * (L - 1) // 2
    xf = xi.astype(np.float64)
    sf = sigma.astype(np.float64)
    m = (xf * sf[None, :, :]).sum(axis=2) / N
    H = np.zeros((L, N), dtype=np.float64)

    if mode == "exact":
        sum_m = m.sum(axis=1)
        sum_m2 = (m * m).sum(axis=1)
        E2_shifted = 0.5 * N * (sum_m * sum_m - sum_m2) - N * binomL2
        G = sum_m[:, None] - m
        for a in range(L):
            Ga = G[:, a : a + 1]
            arg = E2_shifted[:, None] - (xf[:, a, :] * sf[a, :][None, :]) * Ga
            mx = arg.max(axis=0, keepdims=True)
            sh = np.sinh(xf[:, a, :] * Ga)
            H[a, :] = np.exp(mx[0]) * (np.exp(arg - mx) * sh).sum(axis=0)
        return H

    for i in range(N):
        xi_i = xf[:, :, i]
        sig_i = sf[:, i]
        mhat = m - xi_i * sig_i[None, :] / N
        rowsum = mhat.sum(axis=1, keepdims=True)
        F = rowsum - mhat
        sum_m = mhat.sum(axis=1)
        sum_m2 = (mhat**2).sum(axis=1)
        Ehat = 0.5 * N * (sum_m**2 - sum_m2) - N * binomL2
        S = (xi_i * sig_i[None, :] * F).sum(axis=1)
        for a in range(L):
            logPhi_a = S - xi_i[:, a] * sig_i[a] * F[:, a]
            arg = Ehat + logPhi_a
            mx = arg.max()
            eA = np.exp(arg - mx)
            sh = np.sinh(xi_i[:, a] * F[:, a])
            H[a, i] = np.exp(mx) * (eA * sh).sum()
    return H


def one_step_parallel(
    xi: np.ndarray, sigma: np.ndarray, mode: str = "cavity"
) -> np.ndarray:
    """Computes one synchronous zero-temperature update of the spins."""
    H = compute_local_field(xi, sigma, mode=mode)
    sn = np.sign(H).astype(np.int8)
    sn[sn == 0] = 1
    return sn


def initial_state(xi, mu0=0, r=1.0, rng=None) -> np.ndarray:
    """Generates an initial state with a specified overlap r to pattern mu0."""
    rng = rng or np.random.default_rng()
    L, N = xi.shape[1], xi.shape[2]
    sigma = xi[mu0].copy().astype(np.int8)
    if r >= 1.0 - 1e-12:
        return sigma
    p_flip = 0.5 * (1.0 - r)
    flip = rng.random(size=(L, N)) < p_flip
    return np.where(flip, -sigma, sigma).astype(np.int8)


def mattis_overlap(xi, sigma, mu0=0, layer=None) -> float:
    """Mattis overlap m^mu_a = <xi^mu_a, sigma_a> / N."""
    L, N = sigma.shape
    if layer is not None:
        xi_l = xi[mu0, layer].astype(np.int32)
        sig_l = np.asarray(sigma[layer], dtype=np.int32)
        return float((xi_l * sig_l).sum() / N)
    xi_m = xi[mu0].astype(np.int32)
    sig_m = np.asarray(sigma, dtype=np.int32)
    return float((xi_m * sig_m).sum() / (L * N))


def run_mc_one_step(
    N, L, P, r=1.0, n_trials=30, seed=None, mode: str = "cavity"
) -> tuple[float, float]:
    """MC mean +/- SEM of the one-step overlap on synthetic i.i.d. Rademacher memory."""
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_trials):
        xi = rng.choice(np.array([-1, 1], dtype=np.int8), size=(P, L, N))
        sigma0 = initial_state(xi, mu0=0, r=r, rng=rng)
        sigma1 = one_step_parallel(xi, sigma0, mode=mode)
        res.append(mattis_overlap(xi, sigma1, mu0=0))
    a = np.array(res)
    return float(a.mean()), float(a.std() / max(np.sqrt(len(a)), 1))


# ===========================================================================
# Finite-N predictions (log-domain)
# ===========================================================================
def theory_m1_one_step(
    L,
    P,
    N,
    r=1.0,
    sigma2_per_pattern=None,
    use_finiteN_signal=True,
) -> float:
    """Gaussian one-step overlap prediction; uses C_L(r)*exp(-N Sigma_L(r)) when r < 1."""
    if r < 1.0 - 1e-12:
        mu1 = C_L_prefactor(L, r) * np.exp(-N * Sigma_L(L, r))
    else:
        mu1 = mu1_signal_finiteN(L, N) if use_finiteN_signal else mu1_signal(L)
    if sigma2_per_pattern is None:
        log_s2pp = log_noise_var_typ(L, N)
    elif sigma2_per_pattern <= 0:
        return 1.0
    else:
        log_s2pp = float(np.log(sigma2_per_pattern))
    if P <= 1:
        return 1.0
    log_sig = 0.5 * (np.log(P - 1) + log_s2pp)
    if mu1 == 0.0:
        return 0.0
    log_arg = np.log(abs(mu1)) - log_sig - 0.5 * np.log(2.0)
    return float(np.sign(mu1) * sp_erf(np.exp(np.clip(log_arg, -700.0, 700.0))))


def empirical_noise_std_mc(L, N, P, r=1.0, n_trials=30, seed=42) -> float:
    samples = []
    for trial in range(n_trials):
        rng_t = np.random.default_rng(seed + trial * 1000)
        xi = rng_t.choice([-1, 1], size=(P, L, N)).astype(np.int8)
        sigma0 = initial_state(xi, mu0=0, r=r, rng=rng_t)
        H_noise = compute_local_field(xi[1:], sigma0) if P > 1 else np.zeros((L, N))
        samples.append((xi[0] * H_noise).flatten())
    return float(np.concatenate(samples).std())


def _basin_from_log_noise_std(
    L: int, N: int, r: float, log_sigma_noise: float, prob_threshold: float = 1e-12
) -> float:
    if L == 1:
        return 0.0
    Nm1 = N - 1
    p = 0.5 * (1.0 + r)
    ks = np.arange(Nm1 + 1, dtype=np.float64)
    lb = gammaln(N) - gammaln(ks + 1) - gammaln(Nm1 - ks + 1)
    with np.errstate(divide="ignore"):
        if p >= 1.0:
            lbp = np.where(ks == Nm1, 0.0, -np.inf)
        elif p <= 0.0:
            lbp = np.where(ks == 0, 0.0, -np.inf)
        else:
            lbp = lb + ks * np.log(p) + (Nm1 - ks) * np.log(1.0 - p)
    bp = np.exp(lbp)
    mh = (2.0 * ks - Nm1) / float(N)

    eps = max(prob_threshold ** (1.0 / L), 1e-8)
    keep = np.where(bp >= eps)[0]
    if len(keep) == 0:
        keep = np.array([int(np.argmax(bp))])
    bk, mk = bp[keep], mh[keep]
    Nk = len(keep)

    grids = np.meshgrid(*[np.arange(Nk)] * L, indexing="ij")
    K = np.stack([g.reshape(-1) for g in grids], axis=1)
    pk = np.prod(bk[K], axis=1)
    mask = pk >= prob_threshold
    K, pk = K[mask], pk[mask]
    if len(K) == 0:
        return 0.0

    m = mk[K]
    F = m.sum(axis=1, keepdims=True) - m
    E = 0.5 * N * (m.sum(axis=1) ** 2 - (m * m).sum(axis=1)) - N * L * (L - 1) / 2.0

    S = np.array(list(itertools.product([-1, 1], repeat=L - 1)), dtype=np.float64)
    pp, pn = 0.5 * (1.0 + r), 0.5 * (1.0 - r)
    ps = (pp ** (S == 1).sum(axis=1)) * (pn ** (S == -1).sum(axis=1))
    arg = S @ F[:, 1:].T
    sgn_F0 = np.sign(F[:, 0])[None, :]
    if not np.isfinite(log_sigma_noise):
        local_m = sgn_F0
    else:
        lx = _log_abs_sinh(F[:, 0])[None, :] + E[None, :] + arg
        z = sgn_F0 * np.exp(np.clip(lx - log_sigma_noise, -700.0, 700.0))
        local_m = np.where(F[:, 0][None, :] == 0.0, 0.0, sp_erf(z / np.sqrt(2.0)))
    return float((pk * (ps[:, None] * local_m).sum(axis=0)).sum())


def basin_log(
    L: int,
    P: int,
    N: int,
    r: float,
    log_s2pp: float,
    prob_threshold: float = 1e-12,
) -> float:
    """Log-domain finite-N basin retrieval overlap for a given log per-pattern noise variance."""
    if P <= 1 or not np.isfinite(log_s2pp):
        log_sigma = -np.inf
    else:
        log_sigma = 0.5 * (np.log(P - 1) + log_s2pp)
    return _basin_from_log_noise_std(L, N, r, log_sigma, prob_threshold=prob_threshold)


def theory_basin_finiteN(
    L, P, N, r, sigma2_per_pattern=None, prob_threshold=1e-12
) -> float:
    """Vectorised finite-N basin theory (probability of correct retrieval)."""
    if sigma2_per_pattern is None:
        sigma_noise = empirical_noise_std_mc(L, N, P, r=r, n_trials=1000)
        log_sigma = float(np.log(sigma_noise)) if sigma_noise > 0.0 else -np.inf
        return _basin_from_log_noise_std(
            L, N, r, log_sigma, prob_threshold=prob_threshold
        )
    if P <= 1 or sigma2_per_pattern <= 0.0:
        return basin_log(L, P, N, r, -np.inf, prob_threshold=prob_threshold)
    return basin_log(
        L, P, N, r, float(np.log(sigma2_per_pattern)), prob_threshold=prob_threshold
    )


def theory_basin_finiteN_typical(
    L: int,
    P: int,
    N: int,
    r,
    n_draws: int = 16,
    n_calib: int | None = None,
    base_seed: int = 42,
    prob_threshold: float = 1e-12,
) -> dict:
    """R1 log-domain typical finite-N basin prediction averaged across independent store draws."""
    r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
    calib = max(P, 1) if n_calib is None else n_calib
    seeds = [base_seed] + list(range(1, n_draws))
    logv = np.array(
        [log_noise_var_typ(L, N, n_calib=calib, seed=s) for s in seeds],
        dtype=np.float64,
    )
    curves = np.array(
        [
            [
                basin_log(L, P, N, float(rv), float(lv), prob_threshold=prob_threshold)
                for rv in r_arr
            ]
            for lv in logv
        ]
    )
    mean_c = curves.mean(axis=0)
    sd_c = curves.std(axis=0, ddof=1) if n_draws > 1 else np.zeros_like(mean_c)
    min_c = curves.min(axis=0)
    max_c = curves.max(axis=0)
    is_scalar = np.ndim(r) == 0
    return dict(
        mean=float(mean_c[0]) if is_scalar else mean_c,
        std=float(sd_c[0]) if is_scalar else sd_c,
        min=float(min_c[0]) if is_scalar else min_c,
        max=float(max_c[0]) if is_scalar else max_c,
        log_v_typ_mean=float(logv.mean()),
        log_v_typ_std=float(logv.std(ddof=1)) if n_draws > 1 else 0.0,
    )


if __name__ == "__main__":  # pragma: no cover
    print("=== LAM engine (reference) self-test ===")
    for L in (2, 3, 4):
        sp = saddle_point(L)
        print(
            "L=%d  x*=%.4f  rho_L=%.4f  K_L=%.4f  r_c=%.4f  mu1=%.6f  mu1_finN(128)=%.6f"
            % (
                L,
                sp["x_star"],
                sp["rho_L"],
                K_L_prefactor(L),
                sp["r_c"],
                mu1_signal(L),
                mu1_signal_finiteN(L, 128),
            )
        )
