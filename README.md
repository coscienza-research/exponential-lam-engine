# Exponential LAM: Multilayer Hetero-Associative Engine

[![arXiv](https://img.shields.io/badge/arXiv-2607.29554-b31b1b.svg)](https://arxiv.org/abs/2607.29554)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2607.29554-0077c2.svg)](https://doi.org/10.48550/arXiv.2607.29554)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)](test_lam.py)
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen.svg)](test_lam.py)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!IMPORTANT]
> ### Official Reference Implementation
> This repository provides the official implementation, Monte Carlo simulation engines (CPU NumPy and GPU-accelerated PyTorch), and analytical saddle-point solvers for the research paper:
>
> **Exponential Capacity in Multilayer Hetero-Associative Neural Networks**  
> Elena Agliari, Adriano Barra, Andrea Ladiana, Andrea Lepre  
> *arXiv preprint* [arXiv:2607.29554](https://arxiv.org/abs/2607.29554) (2026).  
> **Preprint:** [https://arxiv.org/abs/2607.29554](https://arxiv.org/abs/2607.29554) | **DOI:** [10.48550/arXiv.2607.29554](https://doi.org/10.48550/arXiv.2607.29554)

---

## Overview

Exponential Hopfield networks store a number of patterns that scales exponentially with the number of neurons ($P \sim e^N$). In their classical formulation, however, they are auto-associative: they complete a corrupted copy of a memory into the memory itself.

The **Multilayer Hetero-Associative Layered Associative Memory (LAM)** extends exponential networks to multi-layer architectures capable of storing and retrieving associations across distinct sensory channels or heterogeneous data domains (e.g., mapping T-cell receptor chains to target epitopes, or natural language cues to conversational intents).

This repository provides two high-performance simulation backends and theoretical tools:

1. **Analytical & Reference NumPy Core (`lam_engine.py`)**: Exact 1-step local field dynamics, saddle-point solutions for the critical storage rate $\rho_L$, asymptotic log-capacity predictions, and finite-$N$ calculations.
2. **GPU-Accelerated PyTorch Engine (`lam_engine_torch.py`)**: Vectorized, batched Monte Carlo retrieval engine designed for large-scale GPU evaluations, cue-seeding, and cue-clamping across arbitrary numbers of layers.

---

## Scientific Scope & Mathematical Foundation

### 1. Multi-Layer Exponential Energy

The network consists of $L$ layers of $N$ binary neurons $\sigma_l = (\sigma_{l,1}, \dots, \sigma_{l,N}) \in \{-1, +1\}^N$, with each layer carrying its own dataset of $P$ patterns $\xi_l^\mu \in \{-1, +1\}^N$. The global energy is an exponential function of the product of per-layer Mattis overlaps $m_l^\mu = \frac{1}{N} \sum_{i=1}^N \xi_{l,i}^\mu \sigma_{l,i}$:

$$\mathcal{H}(\sigma) = - \sum_{\mu=1}^P \exp\left( \beta \prod_{l=1}^L m_l^\mu \right)$$

The Hamiltonian is minimized precisely when all layers simultaneously retrieve patterns sharing the exact same index $\mu$, establishing a robust hetero-associative bound.

### 2. Exponential Storage Capacity

Through exact cavity / signal-to-noise large-deviation analysis, the aligned hetero-associative state is proved to be a stable fixed point of the zero-temperature dynamics up to a critical number of stored associations:

$$P_c \sim e^{N \rho_L}, \quad \text{with } \rho_L \sim L \log 2$$

Expanding the basins of attraction to tolerate corrupted cues lowers the effective rate $\rho_L$ but **strictly preserves its exponential character**.

### 3. Domain Universality

The closed-form analytical predictions describe—without any parameter refitting—diverse empirical data modalities:
- Synthetic nonlinear manifolds (Hidden Manifold Model),
- Real-world immunological triplets (VDJdb T-cell-receptor $\alpha/\beta$ chains and target epitopes),
- Natural language intent benchmarks (CLINC150).

---

## Key Features

- **Dual-Engine Architecture**: Seamless transition between high-precision NumPy CPU reference routines and massively batched GPU PyTorch tensors.
- **Cue Seeding & Clamping**: Automatic initial-state seeding from partial cues and selective layer clamping during dynamical relaxation.
- **Exact Theoretical Solvers**: Numerical saddle-point root-finding for $\rho_L$ and theoretical 1-step overlap curves $m_1(r)$.
- **Strict Reproducibility**: Deterministic seed management, vectorization benchmarks, and self-contained test validations.

---

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

To install with PyTorch GPU support and development/testing dependencies:

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

### 1. Theoretical Predictions & Critical Capacity Rate

Evaluate the analytical critical storage rate $\rho_L$ and predict the expected 1-step Mattis overlap:

```python
import lam_engine

L, N = 3, 64
sp = lam_engine.saddle_point(L)
print(f"Critical rate rho_L: {sp['rho_L']:.4f}")
print(f"Log-capacity (N * rho_L): {N * sp['rho_L']:.2f} nats")

# Expected 1-step overlap for P = 20 stored patterns starting from pure cue (r = 1.0)
m1_theory = lam_engine.theory_m1_one_step(L, P=20, N=N, r=1.0)
print(f"Theoretical 1-step overlap m_1: {m1_theory:.4f}")
```

### 2. Single-State Dynamics (Reference NumPy Engine)

Simulate retrieval of a multi-layer state corrupted by noise:

```python
import numpy as np
import lam_engine

P, L, N = 20, 2, 64
rng = np.random.default_rng(42)

# Generate P random archetype associations across L layers
xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)

# Initialize a state corrupted with noise (overlap r = 0.7 with pattern mu0=0)
sigma_init = lam_engine.initial_state(xi, mu0=0, r=0.7, rng=rng)

# Perform one parallel zero-temperature synchronous update step
sigma_next = lam_engine.one_step_parallel(xi, sigma_init)

# Evaluate Mattis overlap with the target pattern
overlap = lam_engine.mattis_overlap(xi, sigma_next, mu0=0)
print(f"Target overlap across layers: {overlap}")
```

### 3. Batched GPU Dynamics with Cue Seeding & Clamping (PyTorch)

Process batched queries on GPU while clamping known sensory channels (e.g., layer 0):

```python
import numpy as np
import lam_engine_torch

P, L, N, B = 50, 3, 128, 10
rng = np.random.default_rng(42)

xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)
cues = rng.choice([-1, 1], size=(B, L, N)).astype(np.int8)

# Deterministically seed unobserved layers (layers 1 and 2) using cue layer 0
seeded_states = lam_engine_torch.cue_seed_auto(xi, cues, known_layers=[0])

# Dynamically evolve the network while clamping cue layer 0
final_states = lam_engine_torch.iterate_parallel_auto(
    xi, seeded_states, n_steps=3, clamp_layers=[0]
)
print("Batched GPU retrieval completed. Final shape:", final_states.shape)
```

---

## Reproducibility & Provenance

The simulation engine follows rigorous reproducibility practices:
- Fully deterministic initialization using isolated NumPy/PyTorch PRNG streams.
- Strict tensor contract enforcement across layer dimensions and batch shapes.
- Automated numerical consistency checks between CPU NumPy and GPU PyTorch implementations.

---

## Project Structure

```text
exponential-lam-engine/
├── lam_engine.py          # Reference NumPy engine, saddle-point theory & capacity solvers
├── lam_engine_torch.py    # Vectorized GPU PyTorch engine with cue-seeding & clamping
├── config.py              # Device management (CUDA/CPU) and chunking parameters
├── test_lam.py            # Unit, theoretical limit, and CPU/GPU consistency tests
├── CITATION.cff           # Citation File Format metadata (enables GitHub Cite button)
├── LICENSE                # Open-source MIT License
└── pyproject.toml         # Build specification, dependencies, and tool settings
```

---

## Verification & Testing

Execute the test suite with `pytest`:

```bash
pytest test_lam.py -v
```

Evaluate test coverage:

```bash
pytest test_lam.py --cov=lam_engine --cov=lam_engine_torch --cov-report=term-missing
```

Run code formatting and lint verification:

```bash
ruff check .
ruff format --check .
```

---

## Citation

If you use this software, the Exponential LAM architecture, or the theoretical derivations in your research, please cite our preprint:

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

You can also use GitHub's **"Cite this repository"** button in the sidebar (powered by [`CITATION.cff`](CITATION.cff)).

---

## License

MIT License — Copyright (c) 2026 Andrea Ladiana, Scuderia "Cò'Scienza" (Sapienza University of Rome). See [`LICENSE`](LICENSE) for details.
