# RL for Quantum Error Correction

## 1. Overview

This project compares two RL decoders (DQN and PPO) against the standard Minimum-Weight Perfect Matching (MWPM) decoder for surface code quantum error correction. The emphasis is on physically-motivated quantum simulation — diverse noise models, circuit-level analysis, syndrome visualization — combined with rigorous RL methodology.

**Key libraries:** Stim (circuit simulation & sampling), PyMatching (MWPM baseline), Sinter (parallelized Monte Carlo benchmarking), Stable-Baselines3 (DQN/PPO), Gymnasium (RL environment).

**Key design decisions:**
- Custom Gymnasium environment (not `gym-surfacecode`, which is limited to d≤5)
- On-the-fly batched sampling from Stim for training; deterministic pre-sampled sets for evaluation
- SB3 with custom CNN feature extractors (not from-scratch RL implementations)
- Sparse reward by default; potential-based shaping as principled fallback; heuristic shaping for ablation
- No Qiskit — Stim's noise models are more relevant, faster, and better reflect the state of the art

---

## 2. Project Structure (eventually)

```
qec-rl/
│
├── pyproject.toml
├── requirements.txt
├── Makefile                               # make train-d3, make evaluate, make plots
├── README.md
│
├── configs/
│   ├── base.yaml                          # Shared defaults (seed, device, logging)
│   ├── noise/
│   │   ├── depolarizing.yaml
│   │   ├── biased_z.yaml                  # Z-biased (dephasing-dominated)
│   │   ├── measurement.yaml               # Measurement-dominated noise
│   │   └── correlated.yaml                # Spatially correlated errors
│   ├── agents/
│   │   ├── dqn_d3.yaml
│   │   ├── dqn_d5.yaml
│   │   ├── ppo_d3.yaml
│   │   └── ppo_d5.yaml
│   └── experiments/
│       ├── threshold_sweep.yaml           # Main: logical vs physical error rate curves
│       ├── noise_robustness.yaml          # Biased/correlated noise evaluation
│       ├── generalization.yaml            # Train on depol, eval on biased
│       └── ablation_reward.yaml           # Reward shaping comparison
│
├── src/
│   ├── __init__.py
│   │
│   ├── quantum/                           # ── Quantum simulation (emphasis area) ──
│   │   ├── __init__.py
│   │   ├── surface_code.py                # Circuit generation & batched sampling via Stim
│   │   ├── noise_models.py                # 4 noise models with physical motivation
│   │   ├── syndrome.py                    # Syndrome reshaping, diff computation, visualization
│   │   ├── decoder_baseline.py            # MWPM via PyMatching + Sinter benchmarking
│   │   └── circuit_analysis.py            # Error budget, DEM analysis, circuit structure
│   │
│   ├── envs/                              # ── Gymnasium environment ──
│   │   ├── __init__.py
│   │   ├── surface_code_env.py            # Core MDP environment
│   │   ├── wrappers.py                    # Observation wrappers (flatten, pad)
│   │   └── reward.py                      # Reward variants: sparse, potential-based, heuristic
│   │
│   ├── agents/                            # ── RL agents via SB3 ──
│   │   ├── __init__.py
│   │   ├── networks.py                    # CNN feature extractors for syndrome grids
│   │   ├── dqn_agent.py                   # DQN config & creation via SB3
│   │   ├── ppo_agent.py                   # PPO config & creation via SB3
│   │   └── callbacks.py                   # Eval callback, optional curriculum
│   │
│   ├── evaluation/                        # ── Evaluation & plotting ──
│   │   ├── __init__.py
│   │   ├── evaluator.py                   # Systematic evaluation harness
│   │   ├── metrics.py                     # Logical error rate, CIs, threshold estimation
│   │   └── plots.py                       # Publication-quality figures
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                      # YAML config loading & merging
│       └── reproducibility.py             # Seed management, deterministic settings
│
├── scripts/
│   ├── train.py                           # Main training entry point (config-driven)
│   ├── evaluate.py                        # Full evaluation grid
│   ├── benchmark_mwpm.py                  # MWPM baselines via Sinter
│   └── plot_results.py                    # Generate all figures
│
├── notebooks/
│   ├── 01_surface_code_exploration.ipynb  # Stim circuits, syndrome visualization, MWPM
│   ├── 02_env_validation.ipynb            # Environment sanity checks
│   ├── 03_training_analysis.ipynb         # Training curves, debugging
│   └── 04_final_results.ipynb             # Paper figures and analysis
│
├── tests/
│   ├── test_surface_code.py
│   ├── test_noise_models.py
│   ├── test_env.py
│   └── test_evaluator.py
│
└── results/                               # Git-ignored
    ├── models/
    ├── logs/
    ├── data/
    └── figures/
```