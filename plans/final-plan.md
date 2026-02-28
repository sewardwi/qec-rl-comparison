# RL for Quantum Error Correction: Unified Implementation Plan

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

## 2. Project Structure

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

---

## 3. Quantum Simulation Layer (Emphasis Area)

### 3.1 `surface_code.py` — Circuit Generation & Sampling

**Key types:**

```python
@dataclasses.dataclass(frozen=True)
class SurfaceCodeParams:
    distance: int          # 3, 5, or 7
    rounds: int            # typically = distance
    basis: str = "Z"       # "X" or "Z" memory experiment

    @property
    def num_data_qubits(self) -> int:
        return self.distance ** 2

    @property
    def num_stabilizers(self) -> int:
        return self.distance ** 2 - 1

    @property
    def task_string(self) -> str:
        return f"surface_code:rotated_memory_{self.basis.lower()}"


@dataclasses.dataclass
class SurfaceCodeCircuit:
    params: SurfaceCodeParams
    noise_params: dict[str, float]
    circuit: stim.Circuit
    _sampler: stim.CompiledDetectorSampler | None = None  # Lazy, cached

    def sample(self, shots: int = 1024, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Batch-sample (syndromes, observables). Compiles sampler on first call."""
        if self._sampler is None:
            self._sampler = self.circuit.compile_detector_sampler(seed=seed)
        syndromes, observables = self._sampler.sample(shots, separate_observables=True)
        return syndromes, observables

    @property
    def detector_error_model(self) -> stim.DetectorErrorModel:
        return self.circuit.detector_error_model(decompose_errors=True)

    @property
    def detector_coordinates(self) -> dict[int, tuple[float, ...]]:
        return self.circuit.get_detector_coordinates()
```

**Key functions:**

| Function | Purpose |
|---|---|
| `generate_circuit(params, noise_config)` | Build noisy Stim circuit; dispatch to noise model builder |
| `get_qubit_layout(params)` | Data qubit & stabilizer positions on rotated lattice |
| `analyze_circuit_structure(circuit)` | Gate counts, depth, qubit counts, DEM statistics |

**Sampling strategy:**
- Training: the environment calls `circuit.sample(1024)` to fill a batch. Each `reset()` advances an index. When the batch is exhausted, resample. This is fast (~ms for 1024 shots) and provides fresh data.
- Evaluation: pre-sample a fixed set of `(syndromes, observables)` with a fixed seed. All decoders (MWPM, DQN, PPO) evaluate on identical syndrome sequences for fair comparison.

---

### 3.2 `noise_models.py` — Noise Model Library

Four physically-motivated noise models:

| # | Model | Physical Motivation | Stim Implementation |
|---|---|---|---|
| 1 | **Depolarizing** | Standard benchmark; symmetric X/Y/Z errors | Pass `p` to all four Stim noise params (`after_clifford_depolarization`, `before_round_data_depolarization`, `before_measure_flip_probability`, `after_reset_flip_probability`) |
| 2 | **Biased-Z** | Dephasing-dominated (superconducting qubits, e.g. transmon); Z errors η× more likely | Generate noiseless circuit, insert custom `PAULI_CHANNEL_1(px, py, pz)` with `pz = η·px` after each gate |
| 3 | **Measurement-dominated** | Readout errors ≫ gate errors (common in NISQ devices) | `before_measure_flip_probability = p`, all gate noise params `= p/10` |
| 4 | **Correlated** | Crosstalk between neighboring qubits | Add `CORRELATED_ERROR` instructions for adjacent data qubit pairs |

```python
class NoiseModelType(Enum):
    DEPOLARIZING = "depolarizing"
    BIASED_Z = "biased_z"
    MEASUREMENT = "measurement"
    CORRELATED = "correlated"

@dataclasses.dataclass
class NoiseConfig:
    model_type: NoiseModelType
    physical_error_rate: float
    bias_ratio: float = 1.0            # η for biased-Z
    correlation_strength: float = 0.0  # For correlated model

def build_noisy_circuit(params: SurfaceCodeParams, config: NoiseConfig) -> stim.Circuit:
    """Dispatch to appropriate circuit builder based on noise model type."""
    ...
```

**Biased/correlated implementation approach:**
1. Generate noiseless circuit: `stim.Circuit.generated(...)` with all noise params = 0
2. Iterate through circuit instructions
3. After Clifford gates: insert `PAULI_CHANNEL_1` or `PAULI_CHANNEL_2` with adjusted probabilities
4. For correlated: add `CORRELATED_ERROR(p)` on adjacent qubit pairs after two-qubit gates

**Rationale for 4 models (not 6):** Biased-X is symmetric to biased-Z (just rotate basis). Asymmetric gate noise is a refinement of depolarizing. Four models gives enough variety for the paper without spreading analysis thin. Each model gets a proper physical justification in the writeup.

---

### 3.3 `syndrome.py` — Syndrome Processing

Converts Stim's flat detector array into spatial grids for CNN input.

```python
def get_grid_dimensions(circuit: stim.Circuit) -> tuple[int, int, int]:
    """From detector coordinates, compute (rounds, grid_h, grid_w)."""
    coords = circuit.get_detector_coordinates()
    # coords[i] = (x, y, t); unique (x,y) define spatial grid; unique t define rounds
    ...

def reshape_syndrome_to_grid(
    flat_syndrome: np.ndarray,  # shape (num_detectors,)
    coords: dict[int, tuple],
    grid_h: int, grid_w: int, rounds: int
) -> np.ndarray:
    """Flat bool array → (rounds, grid_h, grid_w) spatial grid."""
    ...

def reshape_syndrome_batch(flat_batch, ...) -> np.ndarray:
    """Batch version: (batch, num_detectors) → (batch, rounds, grid_h, grid_w)."""
    ...

def compute_syndrome_diff(grid: np.ndarray) -> np.ndarray:
    """XOR between consecutive rounds → (rounds-1, grid_h, grid_w).
    Diff syndromes highlight where NEW errors occurred."""
    ...

def visualize_syndrome(grid, distance, round_idx, ax=None):
    """Plot one syndrome round on the surface code lattice."""
    ...

def visualize_syndrome_spacetime(grid, distance):
    """3D space-time visualization across all rounds."""
    ...
```

**Grid dimensions by distance:**
- d=3: grid ≈ 5×5 (from detector coordinate ranges)
- d=5: grid ≈ 9×9
- d=7: grid ≈ 13×13

---

### 3.4 `decoder_baseline.py` — MWPM Baseline & Sinter Benchmarking

```python
class MWPMDecoder:
    """Wrapper around PyMatching for fair comparison with RL decoders."""

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit) -> MWPMDecoder:
        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        return cls(matching)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode batch of syndromes → predicted observable flips."""
        return self._matching.decode_batch(syndromes)

    def evaluate(self, syndromes: np.ndarray, observables: np.ndarray) -> dict:
        """Compute logical error rate on a fixed syndrome set."""
        predictions = self.decode_batch(syndromes)
        errors = np.any(predictions != observables, axis=1)
        n_errors = errors.sum()
        n_shots = len(errors)
        ler = n_errors / n_shots
        ci_lo, ci_hi = clopper_pearson(n_errors, n_shots, alpha=0.05)
        return {"logical_error_rate": ler, "ci_lower": ci_lo, "ci_upper": ci_hi,
                "num_errors": n_errors, "num_shots": n_shots}


def run_sinter_benchmark(
    distances: list[int],
    error_rates: list[float],
    noise_config: NoiseConfig,
    num_workers: int = 4,
    max_shots: int = 100_000,
    max_errors: int = 1000,
) -> pd.DataFrame:
    """High-throughput MWPM baseline generation via Sinter.

    Sinter handles parallelization, early stopping after max_errors logical
    errors, and statistical aggregation. Returns DataFrame with columns:
    distance, physical_error_rate, logical_error_rate, num_shots, num_errors.
    """
    tasks = []
    for d in distances:
        for p in error_rates:
            params = SurfaceCodeParams(distance=d, rounds=d)
            circuit = build_noisy_circuit(params, noise_config._replace(physical_error_rate=p))
            tasks.append(sinter.Task(circuit=circuit.circuit,
                                     json_metadata={"d": d, "p": p}))
    results = sinter.collect(
        num_workers=num_workers, tasks=tasks,
        max_shots=max_shots, max_errors=max_errors,
        decoders=["pymatching"],
    )
    return _sinter_results_to_dataframe(results)
```

This generates gold-standard baseline curves that RL decoders are compared against.

---

### 3.5 `circuit_analysis.py` — Error Budget & DEM Analysis

```python
def analyze_detector_error_model(dem: stim.DetectorErrorModel) -> dict:
    """Parse DEM: count error mechanisms, weight distribution,
    detector connectivity, boundary error fraction."""
    ...

def compute_error_budget(params, noise_config) -> dict:
    """Estimate each noise channel's contribution to logical error rate
    by toggling channels on/off and comparing logical error rates."""
    ...

def scaling_analysis(distances, noise_config) -> pd.DataFrame:
    """Circuit complexity vs distance: qubits, detectors, DEM size, depth."""
    ...
```

These tools support the quantum emphasis in the paper — showing you understand the physics, not just the RL.

---

## 4. Gymnasium Environment

### 4.1 `surface_code_env.py` — MDP Formulation

**Observation:**
- Shape: `(num_channels, grid_h, grid_w)` as `float32` (channels-first for PyTorch/SB3)
- Channel layout:
  - Channels `0` to `rounds-1`: syndrome grids for each round (binary 0/1)
  - Channel `rounds`: X-correction history (which qubits have been X-corrected)
  - Channel `rounds+1`: Z-correction history (which qubits have been Z-corrected)
- Total channels = `distance + 2` (since rounds = distance)
- Including correction history in the observation is critical — without it, the agent can't condition on its own past actions within a round.

**Actions:**
- `Discrete(3d² + 1)`
  - `0` to `d²-1`: apply X on data qubit `i`
  - `d²` to `2d²-1`: apply Z on data qubit `i - d²`
  - `2d²` to `3d²-1`: apply Y on data qubit `i - 2d²` (toggles both X and Z)
  - `3d²`: **commit** — finalize corrections, advance to next round (or end episode)

| Distance | Data qubits | Actions |
|---|---|---|
| d=3 | 9 | 28 |
| d=5 | 25 | 76 |
| d=7 | 49 | 148 |

**Episode flow:**
1. `reset()`: draw next pre-sampled syndrome; reset correction state; set round = 0
2. Agent observes syndrome grid + correction history → applies correction or commits
3. On commit: advance to next round. On correction: update correction channels, stay in same round.
4. Max `2d` corrections per round (safety limit to prevent degenerate exploration)
5. After all `d` rounds committed → episode ends; compute terminal reward.

**Reward computation:**
```python
# For rotated_memory_z:
# Logical Z operator runs along a column of data qubits.
# X corrections on that column flip the logical observable.
correction_flips_logical = np.bitwise_xor.reduce(x_corrections[logical_z_support])
actual_flip = observable[0]  # from Stim
success = (correction_flips_logical == actual_flip)
reward = +1.0 if success else -1.0
```

**Pre-sampling for efficiency:**
```python
class SurfaceCodeEnv(gymnasium.Env):
    BATCH_SIZE = 1024

    def __init__(self, circuit: SurfaceCodeCircuit, reward_fn: str = "sparse", ...):
        self._circuit = circuit
        self._batch_idx = self.BATCH_SIZE  # Force resample on first reset
        ...

    def _refill_batch(self):
        self._syndromes, self._observables = self._circuit.sample(self.BATCH_SIZE)
        self._batch_idx = 0

    def reset(self, seed=None, options=None):
        if self._batch_idx >= self.BATCH_SIZE:
            self._refill_batch()
        self._current_syndrome = self._syndromes[self._batch_idx]
        self._current_observable = self._observables[self._batch_idx]
        self._batch_idx += 1
        self._corrections = np.zeros(...)
        self._current_round = 0
        return self._build_observation(), {}
```

This is physically correct: in real QEC, the decoder observes syndromes and issues corrections without affecting the quantum state. The pre-sampled observables tell us whether the net correction preserves the logical qubit.

### 4.2 `wrappers.py`

| Wrapper | Purpose |
|---|---|
| `FlattenSyndromeWrapper` | Flatten `(C,H,W)` → `(C*H*W,)` for MLP ablation |
| `PadToSquareWrapper` | Pad non-square grids for uniform CNN input |

### 4.3 `reward.py` — Reward Variants

| Strategy | Description | Optimal-policy-preserving? | Use case |
|---|---|---|---|
| `sparse` | +1/-1 at episode end only | Yes (ground truth) | **Default** |
| `potential_based` | `γF(s') - F(s)` where `F = -syndrome_weight` | Yes (Ng et al. 1999) | Fallback if sparse training stalls |
| `heuristic_shaped` | -0.01 per correction + 0.005 × syndromes cleared | No | Ablation study only |

**Default is sparse.** Shaped reward is only activated if training convergence is poor, with the potential-based variant preferred because it provably preserves the optimal policy.

---

## 5. RL Agents (via Stable-Baselines3)

### 5.1 `networks.py` — Custom CNN Feature Extractor

Standard Atari-style CNNs (8×8, 4×4 kernels) would collapse these tiny grids. We use 3×3 kernels with `padding=1` to preserve spatial dimensions, and no pooling.

```python
class SyndromeCNN(BaseFeaturesExtractor):
    """3-layer CNN for syndrome grids. Extends SB3's BaseFeaturesExtractor."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]
        h, w = observation_space.shape[1], observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * h * w, features_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.cnn(x)
```

For d=7 (if pursued): consider `ResidualSyndromeCNN` with skip connections and global average pooling to handle the 13×13 grid and 10K+ flattened features. Only implement if the simple CNN shows gradient issues at d=7.

### 5.2 `dqn_agent.py`

```python
def create_dqn_agent(env, config: dict) -> DQN:
    policy_kwargs = dict(
        features_extractor_class=SyndromeCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[256, 256],
    )
    return DQN(
        "CnnPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,     # Long exploration (many correction patterns)
        exploration_final_eps=0.05,
        target_update_interval=1000,
        train_freq=4,
        verbose=1,
        tensorboard_log=config.get("log_dir"),
        seed=config.get("seed", 42),
        **config.get("overrides", {}),
    )
```

**Total timesteps:** 500K (d=3), 1M (d=5), 2M (d=7). Scaled by action space complexity.

### 5.3 `ppo_agent.py`

```python
def create_ppo_agent(env, config: dict) -> PPO:
    policy_kwargs = dict(
        features_extractor_class=SyndromeCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        share_features_extractor=True,  # Share CNN between actor/critic
    )
    return PPO(
        "CnnPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,    # Prevents collapsing to always-commit
        verbose=1,
        tensorboard_log=config.get("log_dir"),
        seed=config.get("seed", 42),
        **config.get("overrides", {}),
    )
```

**Same total timestep budget as DQN** for fair comparison.

### 5.4 `callbacks.py`

```python
class DecodingEvalCallback(BaseCallback):
    """Every eval_freq steps: run agent on eval_episodes episodes,
    compute logical error rate, log ratio to MWPM baseline, save best model."""
    def __init__(self, eval_env, mwpm_ler: float, eval_freq=10_000,
                 eval_episodes=1000, save_path=None): ...

class CurriculumCallback(BaseCallback):
    """Optional: linearly increase physical_error_rate from p_start to p_end
    over training. Helps agent learn basic patterns before hard cases."""
    ...
```

---

## 6. Evaluation Pipeline

### 6.1 `evaluator.py`

```python
class DecoderEvaluator:
    """Runs all decoders across a parameter grid on IDENTICAL syndrome sets."""

    def __init__(self, distances, error_rates, noise_configs, num_eval_shots=10_000):
        ...

    def add_decoder(self, name: str, decode_fn: Callable):
        """decode_fn(syndromes, circuit) → predicted_observable_flips"""
        ...

    def run(self) -> pd.DataFrame:
        """Returns DataFrame: decoder | distance | p_phys | noise_model |
        logical_error_rate | ci_lower | ci_upper | num_shots | num_errors"""
        # For each (d, p, noise_model):
        #   1. Generate circuit
        #   2. Pre-sample fixed evaluation set with deterministic seed
        #   3. Run every decoder on identical syndromes
        #   4. Compute metrics with Clopper-Pearson CIs
        ...
```

**Evaluation grid:**
- Distances: [3, 5] (+ 7 if checkpoint passed)
- Physical error rates: `np.logspace(-3, -0.7, 15)` (~0.001 to ~0.2)
- Noise models: depolarizing, biased_z, measurement, correlated
- Shots: 10,000 default; 50,000 for key threshold-region points

**For RL decoders:** wrap the trained SB3 model in a function that runs episodes and records success/failure. Report mean ± std over 3 random seeds.

**For MWPM:** use Sinter for high-throughput baselines (100K+ shots), supplemented by PyMatching on the shared evaluation sets for exact comparison.

### 6.2 `metrics.py`

```python
def logical_error_rate(n_errors: int, n_shots: int) -> tuple[float, float, float]:
    """Returns (rate, ci_lower, ci_upper) with Clopper-Pearson 95% CI."""
    ...

def estimate_threshold(error_rates, logical_rates_by_distance) -> float:
    """Find crossing point of log-log curves for different d via interpolation."""
    ...

def decoder_ratio(rl_ler: float, mwpm_ler: float) -> float:
    """RL error rate / MWPM error rate. <1 means RL wins."""
    ...
```

### 6.3 `plots.py`

**Paper figures (map to AAAI sections):**

| Figure | Description | Paper section |
|---|---|---|
| 1 | Logical vs physical error rate (log-log), one line per distance, panels for MWPM/DQN/PPO | Experiments |
| 2 | Decoder comparison overlay — all three decoders on same axes per noise model | Experiments |
| 3 | Decoder ratio (RL/MWPM) vs physical error rate; horizontal line at 1.0 | Experiments |
| 4 | Training curves: reward & logical error rate vs timestep, DQN vs PPO | Experiments |
| 5 | Noise model comparison: logical error rate under each noise model at fixed d, p | Experiments |
| 6 | Reward shaping ablation: sparse vs potential-based vs heuristic | Experiments |
| 7 | Syndrome pattern gallery under each noise model | Background / Quantum emphasis |
| 8 | Example decoder behavior: syndrome → corrections visualized on lattice | Analysis |
| 9 | Generalization matrix: train on noise A, eval on noise B | Experiments |

---

## 7. Experiments (Mapped to Paper)

### Experiment 1: Threshold Curves (Main Result)
- Train DQN and PPO on depolarizing noise for d=3, d=5
- Sweep p ∈ [0.001, 0.2], evaluate all three decoders
- Plot logical error rate vs physical error rate (Figure 1-2)
- Estimate thresholds (Figure 3)
- **Expected:** MWPM threshold ≈ 0.7-1%. RL decoders approach but likely slightly underperform.

### Experiment 2: DQN vs PPO Comparison
- Same training budget, same evaluation
- Compare: sample efficiency, final error rate, training stability (variance over 3 seeds)
- **Expected:** PPO more stable (entropy bonus), DQN potentially lower final error (replay buffer)

### Experiment 3: Non-Standard Noise (Key Differentiator)
- Train agents on biased-Z and correlated noise
- Compare against MWPM (which assumes symmetric/independent errors)
- **Expected:** RL decoders have potential advantage here. This is the strongest result if it works.

### Experiment 4: Generalization
- Train on depolarizing, evaluate on biased/correlated (and vice versa)
- Produce a generalization matrix
- **Expected:** Some performance degradation; quantify how much.

### Experiment 5: Ablation Studies (RL Rigor)
- Reward shaping: sparse vs potential-based vs heuristic
- CNN depth: 2-layer vs 3-layer
- History channels: with vs without correction history
- **Expected:** Shaped reward accelerates training; correction history is necessary.

---

## 8. Dependencies

```
# Quantum simulation
stim>=1.12
pymatching>=2.1
sinter>=1.12

# RL
stable-baselines3>=2.1
gymnasium>=0.29
torch>=2.0

# Data & visualization
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12

# Config & utilities
pyyaml>=6.0
tensorboard>=2.14
tqdm>=4.65

# Testing
pytest>=7.4
```

---

## 9. Timeline (8 Weeks: Feb 23 → Apr 22)

| Week | Dates | Focus | Deliverables | Checkpoint |
|---|---|---|---|---|
| 1 | Feb 23 – Mar 1 | Stim circuits, syndrome reshaping, MWPM baseline | `surface_code.py`, `syndrome.py`, `decoder_baseline.py`, notebook 01 | MWPM logical error rates match literature (~1% threshold) |
| 2 | Mar 2 – Mar 8 | Depolarizing noise verified, Gym environment core | `noise_models.py` (depolarizing only), `surface_code_env.py`, `reward.py` | Env passes sanity checks; oracle agent matches MWPM |
| 3 | Mar 9 – Mar 15 | Environment wrappers, DQN training on d=3 | `wrappers.py`, `dqn_agent.py`, `networks.py`, `callbacks.py`, `train.py` | DQN d=3 training reward increasing |
| 4 | Mar 16 – Mar 22 | PPO training on d=3, scale DQN to d=5 | `ppo_agent.py`, d=5 configs | Both agents converging on d=3; DQN d=5 started |
| 5 | Mar 23 – Mar 29 | Scale PPO to d=5, begin evaluation pipeline, Sinter baselines | `evaluator.py`, `metrics.py`, `benchmark_mwpm.py` | **d=7 go/no-go:** if d=5 converging for both agents, attempt d=7. Otherwise cut d=7, invest in depth. |
| 6 | Mar 30 – Apr 5 | Remaining noise models, train agents on biased/correlated | `noise_models.py` (all 4), train on non-depolarizing | Agents training on biased noise |
| 7 | Apr 6 – Apr 12 | Full evaluation sweeps, generalization experiment, ablations | `plots.py`, `noise_sweep` runs, ablation configs | All experiments complete; raw data in CSVs |
| 8 | Apr 13 – Apr 22 | Paper writing, final figures, polish, cleanup | AAAI-format paper, notebooks 03-04, final figures | **Submit** |

---

## 10. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| d=7 intractable (148 actions, 13×13 grid) | High | Medium | Formal go/no-go at week 5. Present d=3,5 with quantitative scaling analysis. d=7 failure is itself an interesting finding. |
| RL doesn't beat MWPM under depolarizing | High | Low | Expected. Pivot emphasis to biased/correlated noise where MWPM is provably suboptimal. |
| Sparse reward causes training failure at d≥5 | Medium | High | Potential-based shaping as principled fallback. Heuristic shaping as backup. Document ablation. |
| Training time exceeds budget | Medium | Medium | Start d=3 experiments ASAP. Use curriculum learning if needed. SB3 is well-optimized. |
| Environment reward computation has bugs | Medium | Critical | Oracle test: agent that plays MWPM corrections must reproduce MWPM error rate. Run this in week 2. |

---

## 11. Verification Plan

| Check | Method | When |
|---|---|---|
| Stim circuits correct | `circuit.num_detectors` == `(d²-1) × rounds`; visualize circuit diagram | Week 1 |
| Syndrome reshaping correct | Visualize d=3 syndromes on lattice; verify detector positions | Week 1 |
| MWPM matches literature | Threshold ≈ 0.7-1% for circuit-level depolarizing | Week 1 |
| Environment reward correct | Oracle agent (plays MWPM corrections) reproduces MWPM error rate | Week 2 |
| RL agents learning | Training reward increasing from ~0 toward +1 | Week 3 |
| CNN handles all distances | Forward pass test for d=3,5,7 observation shapes | Week 3 |
| Evaluation is fair | All decoders evaluated on identical syndrome sequences | Week 5 |
| Results reproducible | Same results within CI across 3 random seeds | Week 7 |

---

## 12. Paper Outline (AAAI Format)

1. **Abstract** — QEC decoding as MDP; DQN/PPO vs MWPM; main findings on noise robustness
2. **Introduction** — QEC importance, surface code, decoder challenge, RL opportunity, contribution summary
3. **Background** — Surface codes, syndrome extraction, MWPM, MDP formulation, DQN, PPO
4. **Related Work** — Sweke et al. [4], Colomer et al. [7], AlphaQubit [3], Fösel et al. [8], Nautrup et al. [9]
5. **Methods** — Environment design (state/action/reward), noise models, CNN architecture, training details
6. **Experiments** — Threshold curves, DQN vs PPO, noise robustness, generalization, ablations
7. **Conclusion** — Summary, when RL helps (biased/correlated noise), scaling challenges, future work