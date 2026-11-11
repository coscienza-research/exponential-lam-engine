import numpy as np
import pytest
from scipy.special import gammaln, logsumexp

import lam_engine
import lam_engine_torch


def test_saddle_point():
    with pytest.raises(ValueError, match="LAM requires L >= 2."):
        lam_engine.saddle_point(1)

    res = lam_engine.saddle_point(2)
    assert set(res) == {"x_star", "m_star", "phi_L", "rho_L", "r_c"}
    assert np.isclose(res["x_star"], 1.9150, atol=1e-4)
    assert np.isclose(res["rho_L"], 1.34695, atol=1e-4)
    assert np.isclose(res["r_c"], 0.5714, atol=1e-4)

    # Overflow-safe evaluation for large L
    sp_large = lam_engine.saddle_point(30)
    assert np.isfinite(sp_large["rho_L"])
    assert np.isclose(sp_large["rho_L"], 30 * np.log(2.0), rtol=1e-3)


def test_theory_functions_and_KL_prefactor():
    L, N = 2, 32
    mu1 = lam_engine.mu1_signal(L)
    mu1_fin = lam_engine.mu1_signal_finiteN(L, N)
    assert mu1 > 0
    assert mu1_fin > 0
    assert mu1_fin == lam_engine.s_N(L, N)

    # Paper Eq. (eq:KL-main) prefactors K_L
    expected_KL = {2: 0.721638, 3: 0.963293, 4: 0.995042, 5: 0.999329}
    for layer_count, ref_val in expected_KL.items():
        assert np.isclose(lam_engine.K_L_prefactor(layer_count), ref_val, atol=1e-4)
    assert np.isclose(lam_engine.K_L_prefactor(30), 1.0, atol=1e-6)

    noise = lam_engine.noise_var_per_pattern_annealed(N, L)
    assert noise > 0


def test_proposition1_exact_moments_and_bounds():
    # Appendix check: v_N * exp(N * rho_2) approaches K_2
    sp2 = lam_engine.saddle_point(2)
    scaled_12 = np.exp(lam_engine.log_v_N(2, 12) + 12 * sp2["rho_L"])
    scaled_50 = np.exp(lam_engine.log_v_N(2, 50) + 50 * sp2["rho_L"])
    assert np.isclose(scaled_12, 0.578, atol=0.01)
    assert np.isclose(scaled_50, 0.685, atol=0.01)
    assert scaled_50 < lam_engine.K_L_prefactor(2)
    assert lam_engine.log_noise_var_ann_exact(2, 12) == lam_engine.log_v_N(2, 12)
    assert lam_engine.log_v_N(2, 1) == -np.inf

    # Multi-layer exact v_N (chunked evaluation for L=3)
    sp3 = lam_engine.saddle_point(3)
    scaled_L3 = np.exp(lam_engine.log_v_N(3, 10) + 10 * sp3["rho_L"])
    assert 0.0 < scaled_L3 < 1.5

    # Cantelli stability bounds (Proposition 1(i))
    N, L = 8, 2
    Ps = np.array([1, 20, 100])
    gammas = lam_engine.gamma_N_exact(Ps, N, L)
    assert gammas[0] == 0.0
    assert gammas[1] > 0.0 and gammas[2] > gammas[1]
    assert np.isclose(lam_engine.gamma_N_exact(1, N, L), 0.0)
    flips = lam_engine.cantelli_flip_bound(gammas)
    overlaps = lam_engine.cantelli_overlap_bound(gammas)
    assert np.allclose(overlaps, 1.0 - 2.0 * flips)
    assert np.isclose(overlaps[0], 1.0)

    # Proposition 1(iii) non-asymptotic counting ceiling
    assert lam_engine.counting_flip_lower_bound(1, N, L) == 0.0
    c_small = lam_engine.counting_flip_lower_bound(100, 4, 2)
    c_large = lam_engine.counting_flip_lower_bound(1e9, 20, 3)
    assert 0.0 < c_small < 0.25
    assert c_large >= 0.0


def test_gamma_eff_and_P():
    L, N = 2, 32
    g_scalar = lam_engine.gamma_eff(10, N, L)
    assert isinstance(g_scalar, float)
    g_arr = lam_engine.gamma_eff([10, 20], N, L)
    assert g_arr.shape == (2,)

    P_inv = lam_engine.P_for_gamma(g_scalar, N, L)
    assert isinstance(P_inv, int)
    assert abs(P_inv - 10) <= 1

    P_min = lam_engine.P_for_gamma(-10.0, N, L, P_min=5)
    assert P_min == 5
    P_max = lam_engine.P_for_gamma(100.0, N, L, P_max=50)
    assert P_max == 50


def test_m1_theory_gamma():
    m1 = lam_engine.m1_theory_gamma(0.5)
    assert isinstance(m1, float)
    m1_arr = lam_engine.m1_theory_gamma([0.5, 1.0])
    assert m1_arr.shape == (2,)


def test_N_for_log_capacity():
    N = lam_engine.N_for_log_capacity(2, 5.0, N_min=10)
    assert N >= 10


def test_noise_var_typical_and_log_domain():
    # Small N: log_noise_var_typ matches log(noise_var_per_pattern_typical)
    lv_small = lam_engine.log_noise_var_typ(L=2, N=8, n_calib=10, seed=42)
    noise = lam_engine.noise_var_per_pattern_typical(L=2, N=8, n_calib=10, seed=42)
    assert noise > 0
    assert np.isclose(np.exp(lv_small), noise)

    # Large N where float64 noise_var_per_pattern_typical underflows (L=3, N=128)
    lv_large = lam_engine.log_noise_var_typ(L=3, N=128, n_calib=64, seed=42)
    assert np.isfinite(lv_large) and lv_large < -700.0
    m_basin = lam_engine.basin_log(L=3, P=600, N=128, r=0.5, log_s2pp=lv_large)
    assert 0.99 < m_basin <= 1.0


def test_corrupted_saddle_and_thresholds():
    assert np.isclose(lam_engine.I_R(0.0), 0.0)
    assert np.isclose(lam_engine.I_R(1.0), np.log(2.0))
    assert np.allclose(lam_engine.I_R([-0.5, 0.5]), [lam_engine.I_R(0.5)] * 2)

    assert np.isclose(lam_engine.Sigma_L(2, 0.5), 0.4781, atol=1e-4)
    assert lam_engine.Sigma_L(2, 1.0) == 0.0
    assert lam_engine.corrupted_saddle(2, 1.0) == 1.0
    assert np.isclose(lam_engine.C_L_prefactor(2, 0.5), 0.5368, atol=1e-4)
    assert np.isclose(lam_engine.C_L_prefactor(2, 0.7), 0.4896, atol=1e-4)
    assert np.isclose(lam_engine.C_L_prefactor(3, 0.5), 0.6503, atol=1e-3)
    assert np.isclose(lam_engine.C_L_prefactor(2, 1.0), lam_engine.mu1_signal(2))
    assert lam_engine.C_L_prefactor(2, 0.0) == 0.0
    assert lam_engine.C_L_prefactor(3, 0.0) > 0.0

    r_ann = lam_engine.rc_annealed(2)
    r_typ = lam_engine.rc_typical(2)
    assert np.isclose(r_ann, 0.3195, atol=1e-3)
    assert np.isclose(r_typ, 0.5714, atol=1e-3)
    assert np.isclose(lam_engine.eps_annealed(2, r_ann), 0.0, atol=1e-8)
    assert np.isclose(lam_engine.eps_typical(2, r_typ), 0.0, atol=1e-8)

    # Load-consistent threshold L * I_R(r_c) = ln(P) / N
    r_l = lam_engine.rc_load(L=3, N=30, P=35224)
    assert np.isclose(r_l, 0.47, atol=0.02)
    assert lam_engine.rc_load(L=3, N=30, P=1) == 0.0
    assert lam_engine.rc_load(L=2, N=4, P=10**6) == 1.0

    # Asymptotic basin helper
    m_ann = lam_engine.theory_basin_asymptotic(2, 50, 12, r=0.5, regime="annealed")
    m_typ = lam_engine.theory_basin_asymptotic(2, 50, 12, r=0.5, regime="typical")
    assert 0.0 < m_ann <= 1.0
    assert 0.0 < m_typ <= 1.0
    assert lam_engine.theory_basin_asymptotic(2, 1, 12, r=0.5) == 1.0


def test_proposition2_cue_retrieval_theory():
    assert lam_engine.rho_cue(2) == 0.0
    assert np.isclose(lam_engine.rho_cue(3), lam_engine.saddle_point(2)["rho_L"])
    assert np.isclose(lam_engine.rho_cue(4), lam_engine.saddle_point(3)["rho_L"])

    # L=2 closed form vs explicit binomial sum
    N = 12
    k = np.arange(N + 1)
    lp = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) - N * np.log(2.0)
    m = (2.0 * k - N) / N
    with np.errstate(divide="ignore"):
        lt = lp + 2.0 * np.log(np.abs(np.sinh(m)))
    lv_binom = float(logsumexp(lt[np.isfinite(lt)]))
    assert np.isclose(lam_engine.log_v_cue(2, N), lv_binom, atol=1e-12)

    # L=3 multi-layer cue variance & capacity scaling
    lv3 = lam_engine.log_v_cue(3, 6)
    assert np.isfinite(lv3)
    assert np.isclose(lam_engine.log_s_cue(2, N, r=1.0), np.log(np.sinh(1.0)))
    assert lam_engine.log_s_cue(2, N, r=0.0) == -np.inf

    g = lam_engine.gamma_cue(2, N=16, P=5, r=1.0)
    env = lam_engine.cue_envelope(g)
    assert 0.0 < env <= 1.0
    P_back = lam_engine.P_for_gamma_cue(g, N=16, L=2, r=1.0)
    assert abs(P_back - 5) <= 1


def test_dynamics_overlaps_and_exact_energy():
    rng = np.random.default_rng(42)
    P, L, N = 15, 2, 6
    xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)

    # Energy minimum at stored configuration
    E_stored = lam_engine.hamiltonian_energy(xi, xi[0])
    sig0_noisy = lam_engine.initial_state(xi, mu0=0, r=0.5, rng=rng)
    assert E_stored < lam_engine.hamiltonian_energy(xi, sig0_noisy)

    # initial_state
    sig0_full = lam_engine.initial_state(xi, mu0=0, r=1.0, rng=rng)
    assert np.array_equal(sig0_full, xi[0])
    assert sig0_noisy.shape == (L, N)

    with pytest.raises(ValueError):
        lam_engine.compute_local_field(xi, np.zeros((L, N + 1)))
    with pytest.raises(ValueError):
        lam_engine.hamiltonian_energy(xi, np.zeros((L, N + 1)))

    # Brute-force check: mode="exact" agrees 100% with single-flip energy decrease
    sig1_exact = lam_engine.one_step_parallel(xi, sig0_noisy, mode="exact")
    E0 = lam_engine.hamiltonian_energy(xi, sig0_noisy)
    for a in range(L):
        for i in range(N):
            flipped = sig0_noisy.copy()
            flipped[a, i] *= -1
            dH = lam_engine.hamiltonian_energy(xi, flipped) - E0
            if abs(dH) > 1e-12 * abs(E0):
                expected_spin = sig0_noisy[a, i] if dH > 0 else -sig0_noisy[a, i]
                assert sig1_exact[a, i] == expected_spin

    # Cavity vs exact mode match torch batched path
    sig1 = lam_engine.one_step_parallel(xi, sig0_noisy, mode="cavity")
    assert sig1.shape == (L, N)
    assert set(np.unique(sig1)).issubset({-1, 1})

    m_full = lam_engine.mattis_overlap(xi, sig1, mu0=0)
    assert isinstance(m_full, float)
    m_layer = lam_engine.mattis_overlap(xi, sig1, mu0=0, layer=0)
    assert isinstance(m_layer, float)


def test_mc_one_step():
    mean_val, sem_val = lam_engine.run_mc_one_step(8, 2, 3, r=1.0, n_trials=2, seed=42)
    assert isinstance(mean_val, float)
    assert isinstance(sem_val, float)


def test_theory_predictions():
    L, P, N = 2, 3, 8
    m1 = lam_engine.theory_m1_one_step(L, P, N, r=1.0, use_finiteN_signal=False)
    assert 0.0 <= m1 <= 1.0
    m1_corr = lam_engine.theory_m1_one_step(L, P, N, r=0.5, sigma2_per_pattern=0.01)
    assert 0.0 <= m1_corr <= 1.0
    assert lam_engine.theory_m1_one_step(L, P, N, r=0.0, sigma2_per_pattern=0.01) == 0.0
    assert lam_engine.theory_m1_one_step(L, 1, N, r=1.0) == 1.0
    assert lam_engine.theory_m1_one_step(L, P, N, r=1.0, sigma2_per_pattern=0.0) == 1.0

    noise_std = lam_engine.empirical_noise_std_mc(L, N, P, n_trials=2, seed=42)
    assert noise_std >= 0.0

    basin = lam_engine.theory_basin_finiteN(L, P, N, r=1.0, sigma2_per_pattern=0.1)
    assert 0.0 <= basin <= 1.0
    assert np.isclose(
        basin,
        lam_engine.basin_log(L, P, N, r=1.0, log_s2pp=float(np.log(0.1))),
    )
    assert lam_engine.basin_log(L, 1, N, r=1.0, log_s2pp=-10.0) == 1.0
    assert (
        lam_engine.theory_basin_finiteN(L, 1, N, r=1.0, sigma2_per_pattern=0.1) == 1.0
    )
    assert (
        lam_engine.theory_basin_finiteN(L, P, N, r=1.0, sigma2_per_pattern=0.0) == 1.0
    )
    assert np.isclose(
        lam_engine.theory_basin_finiteN(L, P, N, r=0.0, sigma2_per_pattern=0.1),
        0.0,
        atol=1e-12,
    )
    m1_default = lam_engine.theory_m1_one_step(L, P, 12, r=1.0)
    assert 0.0 <= m1_default <= 1.0
    assert np.isclose(
        lam_engine.theory_basin_finiteN(L, P, N, r=-1.0, sigma2_per_pattern=0.1),
        -basin,
        atol=1e-12,
    )
    assert (
        lam_engine.theory_basin_finiteN(L, 1, N, r=-1.0, sigma2_per_pattern=0.0) == -1.0
    )

    # Multi-draw R1 typical finite-N wrapper
    res_typ = lam_engine.theory_basin_finiteN_typical(
        L=2, P=6, N=10, r=[0.2, 0.8], n_draws=3, n_calib=10, base_seed=42
    )
    assert res_typ["mean"].shape == (2,)
    assert res_typ["mean"][1] >= res_typ["mean"][0]
    res_scalar = lam_engine.theory_basin_finiteN_typical(
        L=2, P=6, N=10, r=0.8, n_draws=1, n_calib=10, base_seed=42
    )
    assert isinstance(res_scalar["mean"], float)

    # Edge cases: L=1 and prob_threshold > 1
    assert (
        lam_engine.theory_basin_finiteN(1, P, N, r=1.0, sigma2_per_pattern=0.1) == 0.0
    )
    assert lam_engine.theory_basin_finiteN(L, P, N, r=1.0, prob_threshold=2.0) == 0.0


def test_torch_helpers():
    avail = lam_engine_torch.torch_available()
    assert isinstance(avail, bool)
    if avail:
        xi = np.ones((2, 2, 4), dtype=np.int8)
        t = lam_engine_torch.to_xi_tensor(xi, device="cpu")
        assert t.shape == (2, 2, 4)


def test_torch_dynamics_and_relax():
    P, L, N, B = 6, 2, 8, 3
    rng = np.random.default_rng(42)
    xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)
    sig0 = rng.choice([-1, 1], size=(B, L, N)).astype(np.int8)

    for mode in ("cavity", "exact"):
        sig_torch = lam_engine_torch.iterate_parallel_auto(
            xi, sig0, n_steps=1, device="cpu", mode=mode
        )
        sig_np = lam_engine_torch.iterate_parallel_batched_numpy(
            xi, sig0, n_steps=1, mode=mode
        )
        assert np.array_equal(sig_torch, sig_np)

    # Clamping & cue seed
    sig2 = lam_engine_torch.iterate_parallel_auto(
        xi, sig0, n_steps=2, clamp_layers=[0], device="cpu"
    )
    assert np.array_equal(sig2[:, 0, :], sig0[:, 0, :])

    cue_sig = lam_engine_torch.cue_seed_auto(xi, sig0, known_layers=[0], device="cpu")
    assert np.array_equal(cue_sig[:, 0, :], sig0[:, 0, :])

    sig_np_clamp = lam_engine_torch.iterate_parallel_batched_numpy(
        xi, sig0, n_steps=1, clamp_layers=[0]
    )
    assert np.array_equal(sig_np_clamp[:, 0, :], sig0[:, 0, :])

    cue_np = lam_engine_torch.cue_seed_numpy(xi, sig0, known_layers=[0])
    assert np.array_equal(cue_np[:, 0, :], sig0[:, 0, :])

    # Fixed-point relaxation matches between Torch and NumPy
    stored_batch = xi[:B].copy()
    rel_s, rel_steps, rel_done = lam_engine_torch.relax_parallel_auto(
        xi, stored_batch, max_steps=5, device="cpu"
    )
    assert rel_done.all()
    assert np.all(rel_steps == 0)
    assert np.array_equal(rel_s, stored_batch)

    s_t, st_t, dn_t = lam_engine_torch.relax_parallel_auto(
        xi, sig0, max_steps=10, clamp_layers=[0], device="cpu"
    )
    s_n, st_n, dn_n = lam_engine_torch.relax_parallel_batched_numpy(
        xi, sig0, max_steps=10, clamp_layers=[0]
    )
    assert np.array_equal(s_t, s_n)
    assert np.array_equal(st_t, st_n)
    assert np.array_equal(dn_t, dn_n)


def test_validation_guards():
    with pytest.raises(ValueError):
        lam_engine.I_R(1.5)
    with pytest.raises(ValueError):
        lam_engine.mu1_signal(1)
    with pytest.raises(ValueError):
        lam_engine.mu1_signal_finiteN(1, 8)
    with pytest.raises(ValueError):
        lam_engine.mu1_signal_finiteN(2, 0)
    with pytest.raises(ValueError):
        lam_engine.log_v_N(1, 4)
    with pytest.raises(ValueError):
        lam_engine.log_v_N(2, 0)
    with pytest.raises(ValueError):
        lam_engine.corrupted_saddle(1, 0.5)
    with pytest.raises(ValueError):
        lam_engine.Sigma_L(1, 0.5)
    with pytest.raises(ValueError):
        lam_engine.C_L_prefactor(1, 0.5)
    with pytest.raises(ValueError):
        lam_engine.rc_load(1, 10, 5)
    with pytest.raises(ValueError):
        lam_engine.rho_cue(1)
    with pytest.raises(ValueError):
        lam_engine.log_s_cue(1, 10)
    with pytest.raises(ValueError):
        lam_engine.log_v_cue(1, 10)
    with pytest.raises(ValueError):
        lam_engine.log_v_cue(2, 0)
    with pytest.raises(ValueError):
        lam_engine.theory_basin_asymptotic(2, 10, 8, 0.5, regime="invalid")
    xi = np.ones((2, 2, 4), dtype=np.int8)
    sig = np.ones((2, 4), dtype=np.int8)
    with pytest.raises(ValueError):
        lam_engine.compute_local_field(xi, sig, mode="invalid")
    if lam_engine_torch.torch_available():
        xt = lam_engine_torch.to_xi_tensor(xi, device="cpu")
        with pytest.raises(ValueError):
            lam_engine_torch.one_step_parallel_batched(xt, xt, mode="invalid")


def test_no_torch(monkeypatch):
    monkeypatch.setattr(lam_engine_torch, "_HAS_TORCH", False)

    with pytest.raises(RuntimeError):
        lam_engine_torch.to_xi_tensor(np.ones((2, 2, 4), dtype=np.int8))

    rng = np.random.default_rng(42)
    xi = rng.choice([-1, 1], size=(2, 2, 8)).astype(np.int8)
    sig0 = rng.choice([-1, 1], size=(3, 2, 8)).astype(np.int8)

    sig1 = lam_engine_torch.iterate_parallel_auto(xi, sig0, n_steps=1)
    assert sig1.shape == (3, 2, 8)

    rel_s, rel_steps, rel_done = lam_engine_torch.relax_parallel_auto(
        xi, sig0, max_steps=5
    )
    assert rel_s.shape == (3, 2, 8)
    assert rel_steps.shape == (3,)
    assert rel_done.shape == (3,)

    cue = lam_engine_torch.cue_seed_auto(xi, sig0, known_layers=[0])
    assert cue.shape == (3, 2, 8)
