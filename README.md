# RL Decoders for the Surface Code: DQN vs PPO vs MWPM

Compares two deep RL decoders (DQN and PPO) against the classical MWPM baseline for surface code quantum error correction. Circuits are simulated with Stim, the RL environment is a custom Gymnasium MDP, and agents are trained with Stable-Baselines3.

---

## What it does

Physical errors corrupt qubits continuously. A decoder reads syndrome measurements and figures out what corrections to apply to preserve the logical qubit. MWPM is the standard classical decoder, optimal for depolarizing noise but requires an explicit noise model. The question here is whether RL agents can learn competitive decoding strategies without one.

The project trains DQN and PPO on four noise models and evaluates them against MWPM across a sweep of physical error rates and code distances.

---

## Noise models

| Model | Physical motivation |
|---|---|
| Depolarizing | Symmetric X/Y/Z errors, standard benchmark |
| Biased-Z | Dephasing dominates, as in superconducting transmons |
| Measurement | Readout errors 10× larger than gate errors |
| Correlated | Crosstalk between neighboring qubits — MWPM degrades here |

Correlated noise is the most interesting case: MWPM struggles because it assumes independent errors, while RL learns purely from experience.

---

## MDP formulation

- **State:** `(d+2) × H × W` tensor: syndrome grids for each round plus X/Z correction accumulators
- **Actions:** `Discrete(3d² + 1)`: apply X/Z/Y to any data qubit, or commit to end the round
- **Reward:** Combined: potential-based shaping on syndrome weight reduction + small correction penalty + sparse ±1 terminal
- **Episode:** d commit rounds, one syndrome instance per episode

---

## Key design choices

- Stim for circuit simulation, faster and more accurate noise modeling than Qiskit
- Syndromes pre-sampled in batches of 1024 for training efficiency; deterministic held-out sets for evaluation
- Custom `SyndromeCNN` with 3×3 kernels, standard Atari CNNs (8×8 kernels) collapse the small syndrome grids
- Evaluation always uses sparse reward regardless of training reward, for fair comparison
- All decoders evaluated on identical syndrome sets for fair LER comparison

---

## Running it

Train all agents on all noise models:
```bash
python scripts/run_experiments.py --timesteps 500000 --eval-shots 5000 --eval-episodes 500
```

Train a single agent:
```bash
python scripts/train.py --agent dqn --distance 3 --noise-model correlated --timesteps 500000
```

Regenerate figures from existing results:
```bash
python scripts/run_experiments.py --skip-training
```

---

## Results summary

Both agents learn above-random baselines quickly but remain 3–5× worse than MWPM on depolarizing noise at the training budgets used. The gap narrows on structured noise (biased-Z, measurement) where errors have more regular patterns. Correlated noise, where MWPM itself degrades significantly, is the setting most likely to favor RL.

---

## Stack

- [Stim](https://github.com/quantumlib/Stim): stabilizer circuit simulation and syndrome sampling
- [PyMatching](https://github.com/oscarhiggott/PyMatching): MWPM decoder
- [Sinter](https://github.com/quantumlib/Stim/tree/main/glue/sample): parallelized Monte Carlo benchmarking
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): DQN and PPO implementations
- [Gymnasium](https://gymnasium.farama.org/): RL environment interface

---

## Paper

See QEC-RL-Comparison-Paper.pdf for a more comprehensive write up of formulation,
experiments, and results. This was last updated on April 20, 2026.
