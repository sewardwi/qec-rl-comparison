# Project Plan: RL for Quantum Error Correction

## 1. Project Structure

```
rl-qec/
├── README.md
├── requirements.txt
├── config/
│   ├── noise_models.yaml          # Noise model configurations
│   └── training.yaml              # Hyperparameters for DQN/PPO
├── src/
│   ├── simulation/
│   │   ├── stim_circuits.py       # Stim surface code circuit generation
│   │   ├── qiskit_noise.py        # [STRETCH] Qiskit Aer realistic noise models
│   │   └── syndrome_utils.py      # Syndrome extraction & manipulation
│   ├── environment/
│   │   ├── surface_code_env.py    # Gym-compatible QEC environment
│   │   ├── syndrome_loader.py     # Pre-sampled syndrome trajectory manager
│   │   └── reward.py              # Reward logic (logical error check + shaping)
│   ├── agents/
│   │   ├── dqn.py                 # DQN agent with CNN Q-network
│   │   ├── ppo.py                 # PPO agent with CNN policy
│   │   └── networks.py            # Shared CNN architectures
│   ├── baselines/
│   │   └── mwpm_decoder.py        # PyMatching MWPM baseline wrapper
│   ├── evaluation/
│   │   ├── metrics.py             # Logical error rate computation
│   │   ├── benchmark.py           # Run all decoders on same syndrome sets
│   │   └── noise_sweep.py         # Sweep physical error rates & code distances
│   └── visualization/
│       ├── plots.py               # Error rate curves, training curves
│       └── lattice_viz.py         # Syndrome/correction lattice visualization
├── notebooks/
│   ├── 01_stim_exploration.ipynb
│   ├── 02_env_validation.ipynb
│   ├── 03_training_analysis.ipynb
│   └── 04_final_results.ipynb
├── scripts/
│   ├── generate_syndromes.py      # Batch pre-generation of syndrome data
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── evaluate_all.py
└── tests/
    ├── test_env.py
    └── test_decoders.py
```

## 2. Implementation Phases

### Phase 1: Quantum Simulation Foundation (Week 1–2)

This is the core differentiator — build a robust, physically-motivated simulation layer that goes beyond toy models.

#### 1a. Stim Circuit Generation (`stim_circuits.py`)

- Write parameterized surface code circuit builder for distances d = 3, 5, 7
- Implement `generate_circuit(distance, rounds, noise_model)` that returns a `stim.Circuit`
- Support multiple noise models:
  - **Depolarizing:** Standard `DEPOLARIZE1(p)` / `DEPOLARIZE2(p)` on all gates
  - **Biased noise:** Asymmetric X/Z error rates (e.g., Z-biased with η = pZ/pX ratios of 10, 100)
  - **Spatially correlated:** Inject correlated two-qubit errors on adjacent data qubits to simulate crosstalk
- Extract the detector error model (DEM) from Stim for analysis and baseline decoding
- Implement batch syndrome sampling using `stim.CompiledDetectorSampler` for fast data generation (target: 10⁵–10⁶ samples per configuration)

#### 1b. Batch Syndrome Pre-Generation (`scripts/generate_syndromes.py`)

- For each `(d, p, noise_model)` configuration, pre-generate syndrome trajectory datasets
- Store as `.npz` files with fields:
  - `syndromes`: shape `(n_samples, rounds, num_stabilizers)`
  - `observable_flips`: shape `(n_samples,)` — ground truth logical observable outcomes
  - `metadata`: noise model params, distance, rounds
- Target: 10⁶ samples for training configs, 10⁵ for evaluation configs
- This runs once and is consumed by both the RL environment and MWPM baseline

#### 1c. Qiskit Aer Realistic Noise — STRETCH GOAL (`qiskit_noise.py`)

> **Deprioritized.** Transpiling surface codes onto heavy-hex topology is nontrivial and time-intensive. Only pursue this after Stim-based noise models (depolarizing, biased, correlated) are fully working and evaluated. The three Stim noise models alone provide plenty of material.

- Pull real backend noise parameters from IBM backends using `qiskit_ibm_runtime` (T1, T2, gate errors, readout errors)
- Build `NoiseModel` objects with thermal relaxation, gate-dependent depolarizing, and readout errors
- Transpile surface code circuits onto heavy-hex connectivity and simulate with Aer
- Convert output to unified syndrome format for downstream processing

#### 1d. Syndrome Utilities (`syndrome_utils.py`)

- Unified syndrome representation: binary array of shape `(rounds, num_stabilizers)` for d×d code
- Implement difference syndrome (temporal differencing between consecutive rounds) — this is what the RL agent actually sees
- Syndrome-to-grid mapping for CNN input: reshape flat syndrome into 2D spatial layout matching the stabilizer lattice
- Functions to compute the cumulative Pauli frame and check commutation with logical operators (for reward computation)

---

### Phase 2: RL Environment (Week 2–3)

#### 2a. Syndrome Loader (`syndrome_loader.py`)

Cleanly separates syndrome data from agent interaction, enabling fair benchmarking across decoders.

- `SyndromeDataset` class: loads pre-generated `.npz` files, serves indexed syndrome trajectories
- `SyndromeIterator`: yields `(syndrome_trajectory, observable_flip)` tuples
- Shared across the Gym environment (for RL agents) and the MWPM baseline wrapper
- Supports deterministic iteration (for evaluation — all decoders see identical syndrome sequences) and shuffled sampling (for training)

#### 2b. Gym Environment (`surface_code_env.py`)

- Subclass `gymnasium.Env`
- **Constructor** takes a `SyndromeDataset` instance (not raw Stim circuits) — the env never runs simulations itself
- **State space:** `Box(0, 1, shape=(2d-1, 2d-1, history_len))` — spatial syndrome grid stacked over `history_len` recent rounds
  - Channel 0: current X-stabilizer syndromes
  - Channel 1: current Z-stabilizer syndromes
  - Channels 2+: previous rounds (gives temporal context)
- **Action space:** `Discrete(3 * d² + 1)`
  - Actions `0` to `d²−1`: apply X correction on qubit i
  - Actions `d²` to `2d²−1`: apply Z correction on qubit i
  - Actions `2d²` to `3d²−1`: apply Y correction on qubit i
  - Action `3d²`: "advance" — request next syndrome round (no correction, proceed)
- **Step logic:**
  - Agent selects a Pauli correction → update internal Pauli frame tracker
  - On "advance" action → load next syndrome round from the pre-sampled trajectory
  - After `max_rounds` syndrome cycles → episode ends
- **Reset:** Pull next syndrome trajectory from the `SyndromeDataset`

#### 2c. Reward (`reward.py`)

**Default: shaped reward (not optional — this is the primary reward scheme).**

- **Terminal reward:** +1 if logical qubit preserved, −1 if logical error
  - Logical error check: multiply cumulative correction by residual error, check commutation with logical X and logical Z operators
- **Intermediate shaping (on by default):**
  - Small negative reward per correction applied: `−0.01` per Pauli correction (encourages efficiency, discourages random corrections)
  - Syndrome reduction bonus: `+0.005 × (syndromes_cleared)` when a correction reduces the number of active syndromes (provides credit assignment signal)
- **Ablation mode:** flag to disable shaping and use sparse-only reward, for ablation study comparison
- Rationale: sparse-only reward makes credit assignment extremely difficult at d ≥ 5 where episodes span many rounds. Shaping is the default; ablation demonstrates its impact.

#### 2d. Environment Variants

- `DepolarizingEnv(d, p)` — standard depolarizing noise at physical error rate p
- `BiasedNoiseEnv(d, p, eta)` — Z-biased noise
- `CorrelatedEnv(d, p, correlation_strength)` — spatially correlated errors
- `RealisticEnv(d, backend_name)` — [STRETCH] uses Qiskit Aer noise model from a real backend

---

### Phase 3: Agents & Training (Week 3–4)

**Week 3 checkpoint (end of week 3):** Evaluate d=3 and d=5 DQN training convergence. If d=5 is converging reasonably, proceed with d=7. If not, scope d=7 out and invest the time in deeper analysis at d=3,5 (more noise models, more ablations, failure mode analysis). Document the decision and the convergence data that informed it.

#### 3a. CNN Architecture (`networks.py`)

Shared backbone for both DQN and PPO:

```
Conv2d(in_channels=history_len*2, 64, 3x3, padding=1) → ReLU
Conv2d(64, 64, 3x3, padding=1) → ReLU
Conv2d(64, 32, 3x3, padding=1) → ReLU
Flatten → Linear(32 * grid_size², 256) → ReLU
```

- DQN head: `Linear(256, num_actions)` → Q-values
- PPO heads: `Linear(256, num_actions)` → policy logits; `Linear(256, 1)` → value estimate
- Scale architecture with code distance (grid grows as (2d−1)²)

#### 3b. DQN Agent (`dqn.py`)

Standard DQN with:
- Experience replay buffer (capacity ~100k transitions)
- Target network with hard update every N steps (or Polyak averaging)
- ε-greedy exploration: ε decays from 1.0 → 0.05 over training
- Huber loss on TD error

Hyperparameters (starting points):
- Learning rate: 1e-4 (Adam)
- Batch size: 64
- Discount γ: 0.95
- Target update freq: 1000 steps

Optional: Double DQN to reduce overestimation

#### 3c. PPO Agent (`ppo.py`)

PPO-Clip implementation:
- Clip ratio ε = 0.2
- Rollout buffer: collect 2048 steps per update
- 4 epochs per update, minibatch size 64
- GAE (λ = 0.95) for advantage estimation
- Entropy bonus coefficient: 0.01 (encourage exploration in early training)
- Value loss coefficient: 0.5
- Shared CNN backbone between policy and value heads
- Learning rate: 3e-4 (Adam), with optional linear decay

#### 3d. Training Pipeline (`scripts/train_dqn.py`, `scripts/train_ppo.py`)

- Load pre-generated syndrome `.npz` datasets for each `(d, p, noise_model)` config
- Training loop with logging (W&B or TensorBoard):
  - Episode reward (mean, std)
  - Logical error rate (rolling window)
  - Loss curves, gradient norms
  - ε schedule (DQN) / entropy (PPO)
- Checkpointing every N episodes
- Train each agent for each `(d, p)` combination independently

---

### Phase 4: Baselines & Evaluation (Week 4–5)

#### 4a. MWPM Baseline (`mwpm_decoder.py`)

- Wrap PyMatching decoder:
  - Input: Stim detector error model + syndrome samples **from the same `SyndromeDataset`** used by RL agents
  - Output: correction and logical observable prediction
- Consumes the same `SyndromeIterator` as the RL environment — guarantees identical syndrome sequences for fair comparison
- This is the gold standard — the RL agents need to match or beat this

#### 4b. Evaluation Suite (`evaluation/`)

**`metrics.py`:**
- `logical_error_rate(agent, env, n_episodes)` — decode n_episodes, return fraction of logical errors
- Confidence intervals via Wilson score or Clopper-Pearson (important for comparing decoders at low error rates)
- **Minimum 10⁵ episodes at the lower end of the p sweep** (p ≤ 0.01) to get statistically meaningful separation between decoders
- Decode latency measurement (wall-clock time per syndrome)

**`benchmark.py`:**
- Run all three decoders (MWPM, DQN, PPO) on the same `SyndromeDataset` for fair comparison
- Save raw results as DataFrame for analysis

**`noise_sweep.py`:**
- Sweep physical error rate p ∈ [0.001, 0.01, 0.02, ..., 0.10] for each d ∈ {3, 5} (and d=7 if checkpoint passed)
- For each (d, p): evaluate all decoders on 10⁴–10⁵ episodes
- **Priority sweep: biased noise (η ∈ {1, 10, 100}) and correlated noise settings** — this is where RL is most likely to beat MWPM
- Also sweep depolarizing for completeness
- Compute thresholds: find p* where logical error rate curves for different d cross

---

### Phase 5: Analysis & Visualization (Week 5–6)

#### 5a. Core Plots (`visualization/plots.py`)

- **Logical error rate vs. physical error rate** (log scale) for each decoder at d = 3, 5 (7 if available) — the main result figure
- Threshold estimation: fit curves to find crossing points
- Training curves: episode reward / logical error rate vs. training step for DQN and PPO
- Sample efficiency comparison: logical error rate vs. number of training episodes for DQN vs. PPO
- Noise model comparison: bar chart of logical error rates under depolarizing vs. biased vs. correlated for each decoder

#### 5b. Quantum-Focused Analysis

- **Syndrome visualization** (`lattice_viz.py`): plot the surface code lattice with syndromes highlighted and corrections overlaid — show what the agent "sees" and "does"
- **Decoder failure mode analysis:** for failure cases, visualize the error chain and the agent's correction chain to understand failure modes (e.g., does DQN create logical errors by completing error chains across the lattice?)
- **Noise model sensitivity / generalization:** how much does performance degrade when an agent trained on depolarizing noise is evaluated on biased/correlated noise? (Transfer analysis)
- Comparison to theoretical bounds: plot RL decoder performance against the hashing bound or ML decoder bound where known

#### 5c. RL-Focused Analysis (for CS 4180 requirements)

- **DQN vs. PPO:** training stability (variance of returns), sample efficiency, final performance
- **Ablation studies:**
  - Reward shaping: compare default shaped reward vs. sparse-only (demonstrate the impact quantitatively)
  - CNN depth: 2-layer vs. 3-layer backbone
  - History length: 1 vs. 3 vs. 5 rounds of temporal context
- **Q-value / advantage visualization:** what states does the agent consider high-value?

---

## 3. Key Dependencies

```
stim>=1.12
pymatching>=2.0
gymnasium>=0.29
torch>=2.0
numpy
matplotlib
seaborn
pandas
pyyaml
tensorboard (or wandb)

# Stretch goal only:
# qiskit>=1.0
# qiskit-aer>=0.13
# qiskit-ibm-runtime>=0.20
```

## 4. Timeline

| Week | Milestone | Deliverable | Decision Point |
|------|-----------|-------------|----------------|
| 1 | Stim circuit generation + batch syndrome pre-generation for d=3,5,7 | `stim_circuits.py`, `syndrome_utils.py`, `generate_syndromes.py`, notebook 01 | — |
| 2 | Gym environment + syndrome loader functional, reward shaping implemented | `syndrome_loader.py`, `surface_code_env.py`, `reward.py`, notebook 02 | — |
| 3 | DQN training on d=3 depolarizing, MWPM baseline running | `dqn.py`, `networks.py`, `mwpm_decoder.py`, training scripts | **d=7 go/no-go checkpoint** |
| 4 | PPO training, scale to d=5 (and d=7 if checkpoint passed), begin evaluation sweeps | `ppo.py`, `noise_sweep.py`, `benchmark.py` | — |
| 5 | Full evaluation across all noise models (prioritize biased/correlated), visualization | All evaluation + visualization code, notebook 03 | — |
| 6 | Analysis, ablation studies, writeup, polish | Final notebook 04, report | — |

## 5. Risk Mitigation

- **d=7 intractable:** Action space is 3(49)+1 = 148 actions, state grid is 13×13. **Formal go/no-go checkpoint at end of week 3** based on d=3,5 convergence data. If d=5 is struggling, cut d=7 and redirect effort to deeper analysis at d=3,5 (more noise models, ablations, failure mode visualization). Document the scaling challenge quantitatively.
- **RL doesn't beat MWPM under depolarizing noise:** Expected — MWPM is near-optimal for simple depolarizing. **Pivot emphasis to biased/correlated noise** where MWPM provably underperforms and RL should shine. This is the priority sweep.
- **Syndrome data generation:** Pre-generate everything with Stim upfront (Phase 1b). Stim samples ~10⁶ syndromes in seconds. The RL environment and MWPM baseline both consume from the same pre-generated `.npz` files.
- **Reward sparsity hurts training:** Shaped reward is now the **default**, not optional. Sparse-only is available as an ablation flag to demonstrate the impact.
- **Qiskit Aer is slow / complex:** Demoted to stretch goal. Three Stim-based noise models provide sufficient material.

## 6. Emphasis Strategy

Since this is for an RL course but you want to showcase the quantum side:

**Quantum depth (where you go beyond typical RL projects):**
- Three physically-motivated noise models (depolarizing, biased, correlated) with clear physical justification
- Proper surface code circuit construction and syndrome extraction via Stim
- Analysis of decoder failure modes in terms of quantum error chains
- Noise generalization analysis (train on one noise model, evaluate on another)
- Discussion of practical decoder requirements (latency, scalability)
- [Stretch] Real IBM backend noise parameters via Qiskit

**RL rigor (what CS 4180 expects):**
- Two well-implemented RL algorithms with clear comparison
- Proper hyperparameter choices with justification
- Training curves, sample efficiency, stability analysis
- Ablation studies (reward shaping, CNN depth, history length)
- Strong baseline (MWPM) that contextualizes RL performance