# Exponential LAM: Multilayer Hetero-Associative Engine

[![arXiv](https://img.shields.io/badge/arXiv-2607.29554-b31b1b.svg)](https://arxiv.org/abs/2607.29554)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2607.29554-0077c2.svg)](https://doi.org/10.48550/arXiv.2607.29554)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)](test_lam.py)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen.svg)](test_lam.py)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!IMPORTANT]
> ### Official Reference Implementation
> This repository provides the official simulation engine and analytical solvers accompanying:
>
> **Exponential Capacity in Multilayer Hetero-Associative Neural Networks**  
> Elena Agliari, Adriano Barra, Andrea Ladiana, Andrea Lepre  
> *arXiv preprint* [arXiv:2607.29554](https://arxiv.org/abs/2607.29554) (2026).  
> **Preprint:** [https://arxiv.org/abs/2607.29554](https://arxiv.org/abs/2607.29554) | **DOI:** [10.48550/arXiv.2607.29554](https://doi.org/10.48550/arXiv.2607.29554)

---

## Overview

Classical dense/exponential Hopfield networks store exponentially many memories ($P \sim e^{cN}$) in an auto-associative architecture. The **Multilayer Hetero-Associative Layered Associative Memory (LAM)** extends this exponential storage and retrieval principle to $L \ge 2$ interacting binary layers linking distinct sensory channels or heterogeneous modalities.

This package contains the core engine and analytical theory aligned with Revision 1 (R1) of the manuscript:

1. **Reference NumPy Engine & Analytical Theory (`lam_engine.py`)**:
   - Exact zero-temperature cavity local-field dynamics (`mode="cavity"`) and exact single-flip energy gradient dynamics (`mode="exact"`), together with the exact pairwise exponential Hamiltonian (`hamiltonian_energy`).
   - Asymptotic stability saddle-point solver (`saddle_point`) and corrected Laplace prefactor `K_L_prefactor` ($2/\sqrt{\det \mathcal{H}_L^*}$).
   - **Proposition 1** non-asymptotic finite-$N$ moments (`mu1_signal_finiteN` / `s_N`, `log_v_N`), Cantelli stability bounds (`cantelli_flip_bound`, `cantelli_overlap_bound`), and counting error lower bound (`counting_flip_lower_bound`).
   - **Proposition 2** cue-driven retrieval theory (`rho_cue`, `log_s_cue`, `log_v_cue`, `gamma_cue`, `cue_envelope`).
   - Corrupted-basin saddle, annealed/typical/load-consistent thresholds (`corrupted_saddle`, `Sigma_L`, `C_L_prefactor`, `rc_annealed`, `rc_typical`, `rc_load`), and log-domain finite-$N$ typical-noise solvers (`log_noise_var_typ`, `basin_log`, `theory_basin_finiteN_typical`) avoiding double-precision underflow at large $N$.
2. **Batched GPU/CPU PyTorch Engine (`lam_engine_torch.py`)**:
   - Vectorized, chunked Monte Carlo retrieval supporting `mode="cavity"` and `mode="exact"`, cue seeding (`cue_seed_auto`), layer clamping (`iterate_parallel_auto`), fixed-point relaxation (`relax_parallel_auto`), and transparent pure-NumPy fallback.

---

## Scientific Scope & Mathematical Foundation

### 1. Pairwise Multilayer Exponential Energy

The network comprises $L \ge 2$ layers of $N$ binary spins $\sigma^a \in \{-1, +1\}^N$ storing $P$ associated tuples $\{\xi^{\mu,a}\}_{\mu=1}^P \subset \{-1, +1\}^{L \times N}$. With per-layer Mattis overlaps $m_\mu^a = \frac{1}{N}\sum_{i=1}^N \xi_i^{\mu,a}\sigma_i^a$, the pairwise exponential energy is:

$$\mathcal{H}(\sigma) = -N \sum_{\mu=1}^P \exp\!\left[ N \sum_{1 \le a < b \le L} \left(m_\mu^a m_\mu^b - 1\right) \right]$$

At zero temperature, evaluating either the cavity local field or exact single-flip update across all $NL$ neurons costs $\Theta(NLP)$ in time and memory per sweep. Under cue retrieval, recall of a unique target pattern requires a single-valued association rule $\Psi$ (or a class-level majority readout when identical cues map to multiple stored patterns).

### 2. Storage & Stability Capacity Bracket (Proposition 1)

Under i.i.d. Rademacher patterns, non-asymptotic exact signal and noise moments ($s_N, v_N$) together with Cantelli's one-sided inequality (without relying on a Gaussian noise approximation) and a combinatorial collision counting bound bracket the exponential storage capacity rate $\alpha_c = \frac{1}{N}\ln P_c$:

$$\rho_L \le \alpha_c \le L \log 2, \qquad \rho_L \sim L \log 2 \text{ as } L \to \infty$$

A single stored configuration remains stable with high probability for $\alpha < \rho_L$, and all $P$ stored configurations are simultaneously stable for $\alpha < \rho_L / 2$. Asymptotically, the per-pattern noise second moment scales as $v_N \simeq K_L e^{-N\rho_L}$ with:

$$K_L = \frac{2}{\sqrt{\det \mathcal{H}_L^*}}\,\sinh^2\!\big((L-1)m^*\big)\cosh\!\big(2(L-1)m^*\big)^{L-1}$$

giving $K_2 \approx 0.7216$, $K_3 \approx 0.9633$, $K_4 \approx 0.9950$, and $K_L \to 1$ as $L \to \infty$.

### 3. Cue-Driven Retrieval vs. Stability Capacity (Proposition 2)

When $L-1$ layers are presented as a cue to infer the remaining target layer, distinct cue patterns are bounded by $2^{(L-1)N}$ and retrieval performance separates into two regimes governed by $\rho_{\text{cue}}(L) = \rho_{L-1}$:

- **$L \ge 3$**: exponential cue-retrieval capacity ($P_c^{\text{cue}} \sim e^{N \rho_{L-1}}$).
- **$L = 2$**: $\rho_1 = 0$; the two-layer cue seed reduces to a Hebbian readout whose capacity scales linearly with layer width ($P / N \to 0$).

### 4. Basins of Attraction & Sub-Capacity Typical Noise

Tolerating initial overlap $r < 1$ reduces the capacity exponent—yielding annealed ($\varepsilon_L^{\text{ann}}(r) = \rho_L - 2\Sigma_L(r)$), typical ($\varepsilon_L^{\text{typ}}(r) = \rho_L - L(L-1)(1-r^2)$), and load-consistent ($L I_R(r_c) = \frac{1}{N}\ln P$) thresholds—while retaining exponential scaling inside the basin of attraction. At finite $N$ and sub-capacity loads ($P \ll e^{N L I_R(m^*)}$), realized noise fluctuations follow the typical-sample scale rather than the annealed rare-tail moment $v_N$; evaluating signal and noise in the log domain (`log_noise_var_typ`, `basin_log`) prevents float64 underflow (e.g., at $N=128, L=3$).

### 5. Structured Data Benchmarks

Across the empirical evaluations reported in the paper—synthetic nonlinear manifolds (Hidden Manifold Model), immunological $\text{TCR}\alpha/\beta$–epitope triplets (VDJdb under sequence-similarity splits), and natural language intent benchmarks (CLINC150 under standard and pooled splits)—analytical finite-$N$ curves contain **no parameters fitted to structured data**, and empirical retrieval accuracy is governed by whether the front-end binary encoder co-locates same-class representations closer than rival classes in Hamming geometry.

---

## Installation

Install in editable mode:

```bash
pip install -e .
```

With PyTorch and development/test tools:

```bash
pip install -e .[torch,dev]
```

### Core Requirements

- Python $\ge 3.10$
- `numpy >= 1.24`
- `scipy >= 1.10`
- Optional GPU acceleration: `torch >= 2.0`

---

## Quick Start

### 1. Stability Bounds, Prefactor, and Cue-Retrieval Rate

```python
import numpy as np
import lam_engine

L, N = 3, 12
sp = lam_engine.saddle_point(L)
K_L = lam_engine.K_L_prefactor(L)
print(f"rho_{L} = {sp['rho_L']:.4f}, K_{L} = {K_L:.4f}, rho_cue = {lam_engine.rho_cue(L):.4f}")

# Proposition 1 exact finite-N reduced load and Cantelli lower bound
gamma_100 = lam_engine.gamma_N_exact(P=100, N=N, L=L)
print(f"Cantelli one-step overlap >= {lam_engine.cantelli_overlap_bound(gamma_100):.4f}")

# Proposition 2 cue-seed exact variance & Cantelli bound (O(1) fast path for L=2)
lv_cue = lam_engine.log_v_cue(L=2, N=64)
g_cue = lam_engine.gamma_cue(L=2, N=64, P=10, r=1.0)
print(f"log v_cue(L=2, N=64) = {lv_cue:.4f}, cue envelope >= {lam_engine.cue_envelope(g_cue):.4f}")
```

### 2. Log-Domain Finite-N Typical Basins (Overflow/Underflow Safe)

```python
import lam_engine

# Stable even at N=128, L=3 where float64 variance (~e^-750) would underflow
L, N, P = 3, 128, 600
res = lam_engine.theory_basin_finiteN_typical(L, P, N, r=[0.2, 0.5, 0.9], n_draws=16)
print("Finite-N typical overlap mean:", res["mean"])
print("Load-consistent threshold r_c:", round(lam_engine.rc_load(L, N, P), 4))
```

### 3. Single-State Dynamics & Exact vs. Cavity Field

```python
import numpy as np
import lam_engine

P, L, N = 20, 2, 32
rng = np.random.default_rng(42)
xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)
sigma0 = lam_engine.initial_state(xi, mu0=0, r=0.7, rng=rng)

sigma_cavity = lam_engine.one_step_parallel(xi, sigma0, mode="cavity")
sigma_exact = lam_engine.one_step_parallel(xi, sigma0, mode="exact")
print("Cavity overlap:", lam_engine.mattis_overlap(xi, sigma_cavity, mu0=0))
print("Exact overlap :", lam_engine.mattis_overlap(xi, sigma_exact, mu0=0))
```

### 4. Batched GPU/CPU Dynamics, Cue Seeding, Clamping & Relaxation

```python
import numpy as np
import lam_engine_torch

P, L, N, B = 50, 3, 64, 8
rng = np.random.default_rng(42)
xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)
cues = xi[:B].copy()

# Seed unknown layers (1 and 2) from known cue layer 0
seeded = lam_engine_torch.cue_seed_auto(xi, cues, known_layers=[0])

# Clamp cue layer 0 and relax to a fixed point
final_states, steps, converged = lam_engine_torch.relax_parallel_auto(
    xi, seeded, max_steps=10, clamp_layers=[0]
)
print("Converged:", converged.all(), "Steps:", steps)
```

---

## Project Structure

```text
exponential-lam-engine/
├── lam_engine.py          # Reference NumPy engine, saddle-point & R1 finite-N/cue theory
├── lam_engine_torch.py    # Batched GPU/CPU PyTorch engine with cue-seeding, clamping & relax
├── config.py              # Device configuration (CUDA/CPU) and chunking defaults
├── test_lam.py            # Unit tests, theoretical identity checks & CPU/Torch parity
├── CITATION.cff           # Citation metadata
├── LICENSE                # MIT License
└── pyproject.toml         # Build & package configuration
```

---

## Verification & Testing

Run the test suite:

```bash
pytest test_lam.py -v
```

Run tests with line coverage:

```bash
pytest test_lam.py -v --cov=lam_engine --cov=lam_engine_torch --cov-report=term-missing
```

Run linter and formatter checks:

```bash
ruff check .
ruff format --check .
```

---

## Citation

If you use this software in your research, please cite the accompanying manuscript:

```bibtex
@article{agliari2026exponential,
  title={Exponential Capacity in Multilayer Hetero-Associative Neural Networks},
  author={Agliari, Elena and Barra, Adriano and Ladiana, Andrea and Lepre, Andrea},
  journal={arXiv preprint arXiv:2607.29554},
  year={2026},
  doi={10.48550/arXiv.2607.29554},
  url={https://arxiv.org/abs/2607.29554}
}
```

You can also use GitHub's **"Cite this repository"** button via [`CITATION.cff`](CITATION.cff).

---

## License

Released under the [MIT License](LICENSE).
