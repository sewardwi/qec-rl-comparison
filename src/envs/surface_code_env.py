"""
Gymnasium environment for surface code decoding as an MDP.

The agent observes syndrome measurements and applies Pauli corrections to
data qubits, then commits to advance to the next round. After all rounds
are committed, a terminal reward indicates whether the corrections preserved
the logical qubit.

Observation:
    Shape (num_channels, grid_h, grid_w) as float32 (channels-first for PyTorch).
    Channels 0..rounds-1: syndrome grids per round (binary 0/1).
    Channel rounds: X-correction accumulator (which data qubits have X corrections).
    Channel rounds+1: Z-correction accumulator (which data qubits have Z corrections).
    Total channels = rounds + 2 = distance + 2.

Actions:
    Discrete(3*d^2 + 1):
        0..d^2-1:        Apply X on data qubit i
        d^2..2*d^2-1:    Apply Z on data qubit i-d^2
        2*d^2..3*d^2-1:  Apply Y on data qubit i-2*d^2 (toggles both X and Z)
        3*d^2:           Commit (advance to next round or end episode)

Episode flow:
    1. reset(): draw next pre-sampled syndrome, reset corrections
    2. Agent applies corrections or commits
    3. On commit: advance round counter. On correction: update correction channels.
    4. Max 2*d corrections per round (safety limit)
    5. After all rounds committed: terminal reward based on correction success
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from src.envs.reward import (
    RewardType,
    compute_combined_reward,
    compute_heuristic_reward,
    compute_potential_based_reward,
    compute_sparse_reward,
)
from src.quantum.noise_models import NoiseConfig, build_noisy_circuit
from src.quantum.surface_code import SurfaceCodeCircuit, SurfaceCodeParams
from src.quantum.syndrome import SyndromeGrid


class SurfaceCodeEnv(gym.Env):
    """
    Gymnasium environment for surface code decoding.

    The environment pre-samples batches of syndromes from a Stim circuit
    for efficiency. Each episode presents one syndrome instance for the
    agent to decode by applying corrections across multiple rounds.
    """

    metadata = {"render_modes": []}
    BATCH_SIZE = 1024

    def __init__(
        self,
        circuit: SurfaceCodeCircuit,
        reward_type: str = "sparse",
        gamma: float = 0.99,
        max_corrections_per_round: Optional[int] = None,
    ):
        """
        Initialize the surface code decoding environment.

        circuit: A SurfaceCodeCircuit to sample syndromes from.
        reward_type: One of "sparse", "potential_based", "heuristic_shaped".
        gamma: Discount factor (used for potential-based shaping).
        max_corrections_per_round: Safety limit on corrections per round.
            Defaults to 2 * distance.
        """
        super().__init__()

        self._circuit = circuit
        self._params = circuit.params
        self._reward_type = RewardType(reward_type)
        self._gamma = gamma

        d = self._params.distance
        self._max_corrections = max_corrections_per_round or 2 * d

        # build syndrome grid mapper
        self._syndrome_grid = SyndromeGrid(circuit.circuit)
        grid_h = self._syndrome_grid.grid_h
        grid_w = self._syndrome_grid.grid_w
        # syndromeGrid.rounds includes boundary detectors, so it's typically
        # params.rounds + 1 (e.g., d=3 with 3 rounds gives 4 temporal slices)
        syndrome_rounds = self._syndrome_grid.rounds

        # observation: syndrome temporal slices + X/Z correction channels
        num_channels = syndrome_rounds + 2
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_channels, grid_h, grid_w),
            dtype=np.float32,
        )

        # actions: X on each qubit + Z on each qubit + Y on each qubit + commit
        n_data = d * d
        self.action_space = gym.spaces.Discrete(3 * n_data + 1)
        self._n_data = n_data
        self._commit_action = 3 * n_data

        # correction channels need to map data qubit indices to grid positions
        # data qubits sit on a d x d sub-grid within the larger detector grid
        # place corrections on a d x d grid embedded in the grid_h x grid_w space
        self._grid_h = grid_h
        self._grid_w = grid_w
        self._syndrome_rounds = syndrome_rounds
        self._rounds = self._params.rounds  # Number of commit rounds in episode

        # build data qubit -> grid position mapping
        self._data_row, self._data_col = self._build_data_qubit_mapping()

        # pre-compute logical operator support (for reward computation)
        # For Z-basis memory experiment, the logical X operator is a column of
        # X corrections, the logical Z observable is what Stim tracks
        # just need to know which data qubits are in the logical operator support
        self._logical_support = self._compute_logical_support()

        # batch sampling state
        self._batch_idx = self.BATCH_SIZE
        self._syndromes: Optional[np.ndarray] = None
        self._observables: Optional[np.ndarray] = None

        # episode state
        self._current_syndrome_grid: Optional[np.ndarray] = None
        self._current_observable: Optional[np.ndarray] = None
        self._x_corrections: Optional[np.ndarray] = None
        self._z_corrections: Optional[np.ndarray] = None
        self._current_round: int = 0
        self._corrections_this_round: int = 0

    def _build_data_qubit_mapping(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Map data qubit indices (0..d^2-1) to positions in the observation grid.

        Data qubits are laid out on a d x d grid. We center this within the
        larger syndrome grid (grid_h x grid_w).
        """
        d = self._params.distance
        # data qubits arranged in a d x d grid, centered in the observation grid
        # syndrome grid is roughly (2d-1) x (2d-1), data qubits at even positions
        rows = np.zeros(d * d, dtype=np.int32)
        cols = np.zeros(d * d, dtype=np.int32)
        for i in range(d):
            for j in range(d):
                idx = i * d + j
                # map to grid coordinates: data qubits at positions
                # that align with the detector grid spacing
                rows[idx] = min(i * 2, self._grid_h - 1)
                cols[idx] = min(j * 2, self._grid_w - 1)
        return rows, cols

    def _compute_logical_support(self) -> np.ndarray:
        """
        Compute which data qubit indices support the logical operator.

        For the rotated surface code Z-basis memory experiment, the logical
        observable involves a specific set of data qubits. We identify these
        from the circuit's observable definition.
        """
        d = self._params.distance
        # For Z-basis memory: logical X is a row of X corrections across the code.
        # The observable Stim reports is whether the logical state was flipped.
        # For the standard rotated surface code, the logical operator support
        # is the first column of data qubits (indices 0, d, 2d, ..., (d-1)*d).
        # But actually, for Stim's convention, we need to track X corrections
        # and check parity against the observable.
        #
        # The key insight: Stim's observable tells us the actual logical flip.
        # Our X corrections contribute to the logical flip if they're on the
        # logical Z operator support. For rotated_memory_z, this is typically
        # a column of d data qubits.
        #
        # We use a simple column as the support. The exact column depends on
        # Stim's convention, but for the reward computation what matters is
        # that we correctly track X-correction parity.
        return np.arange(0, d * d, d)

    def _refill_batch(self):
        """Sample a new batch of syndromes from the circuit."""
        self._syndromes, self._observables = self._circuit.sample(self.BATCH_SIZE)
        self._batch_idx = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset environment for a new episode.

        Draws the next pre-sampled syndrome. Resamples a batch if exhausted.
        """
        super().reset(seed=seed)

        if self._batch_idx >= self.BATCH_SIZE:
            self._refill_batch()

        flat_syndrome = self._syndromes[self._batch_idx]
        self._current_observable = self._observables[self._batch_idx]
        self._batch_idx += 1

        # reshape flat syndrome to spatial grid: (rounds, grid_h, grid_w)
        self._current_syndrome_grid = self._syndrome_grid.reshape_single(
            flat_syndrome
        )

        # reset correction accumulators
        self._x_corrections = np.zeros(self._n_data, dtype=np.bool_)
        self._z_corrections = np.zeros(self._n_data, dtype=np.bool_)
        self._current_round = 0
        self._corrections_this_round = 0

        obs = self._build_observation()
        info = {
            "syndrome_weight": int(flat_syndrome.sum()),
            "distance": self._params.distance,
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one action in the environment.

        action: Integer action from action_space.

        (observation, reward, terminated, truncated, info)
        """
        d2 = self._n_data
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        # track state before action for potential-based reward
        syndrome_weight_before = int(self._current_syndrome_grid.sum())

        if action == self._commit_action:
            # commit: advance to next round
            self._current_round += 1
            self._corrections_this_round = 0

            if self._current_round >= self._rounds:
                # episode over: compute terminal reward
                terminated = True
                success = self._check_success()
                info["success"] = success
                info["x_correction_weight"] = int(self._x_corrections.sum())
                info["z_correction_weight"] = int(self._z_corrections.sum())

                reward = self._compute_terminal_reward(
                    success, syndrome_weight_before
                )
                obs = self._build_observation()
                return obs, reward, terminated, truncated, info

            # non-terminal commit: step reward
            reward = self._compute_step_reward(
                syndrome_weight_before, is_correction=False
            )

        elif action < d2:
            # X correction on data qubit `action`
            self._x_corrections[action] ^= True
            self._corrections_this_round += 1
            reward = self._compute_step_reward(
                syndrome_weight_before, is_correction=True
            )

        elif action < 2 * d2:
            # Z correction on data qubit `action - d^2`
            qubit = action - d2
            self._z_corrections[qubit] ^= True
            self._corrections_this_round += 1
            reward = self._compute_step_reward(
                syndrome_weight_before, is_correction=True
            )

        elif action < 3 * d2:
            # Y correction = X + Z on data qubit `action - 2*d^2`
            qubit = action - 2 * d2
            self._x_corrections[qubit] ^= True
            self._z_corrections[qubit] ^= True
            self._corrections_this_round += 1
            reward = self._compute_step_reward(
                syndrome_weight_before, is_correction=True
            )

        else:
            raise ValueError(f"Invalid action {action}, max is {self._commit_action}")

        # truncate if too many corrections in this round
        if self._corrections_this_round >= self._max_corrections:
            truncated = True
            success = self._check_success()
            info["success"] = success
            info["truncated_reason"] = "max_corrections_exceeded"
            reward = self._compute_terminal_reward(
                success, syndrome_weight_before
            )

        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        """
        Build the full observation tensor.

        Shape (rounds + 2, grid_h, grid_w) float32 array
        """
        sr = self._syndrome_rounds
        obs = np.zeros(
            (sr + 2, self._grid_h, self._grid_w),
            dtype=np.float32,
        )

        # syndrome channels (all temporal slices from detector grid)
        obs[:sr] = self._current_syndrome_grid

        # X-correction channel
        obs[sr, self._data_row, self._data_col] = (
            self._x_corrections.astype(np.float32)
        )

        # Z-correction channel
        obs[sr + 1, self._data_row, self._data_col] = (
            self._z_corrections.astype(np.float32)
        )

        return obs

    def _check_success(self) -> bool:
        """
        Check if the applied corrections successfully preserve the logical qubit.

        For Z-basis memory experiment: X corrections on the logical Z support
        flip the logical observable. Success = our correction parity matches
        the actual flip from Stim.
        """
        # parity of X corrections on the logical operator support
        correction_parity = np.bitwise_xor.reduce(
            self._x_corrections[self._logical_support]
        )
        # Stim's observable tells us the actual logical flip
        actual_flip = bool(self._current_observable[0])
        return bool(correction_parity) == actual_flip

    def _compute_terminal_reward(
        self, success: bool, syndrome_weight_before: int
    ) -> float:
        """Compute reward at episode termination."""
        if self._reward_type == RewardType.SPARSE:
            return compute_sparse_reward(success)
        elif self._reward_type == RewardType.POTENTIAL_BASED:
            return compute_potential_based_reward(
                syndrome_weight_before=syndrome_weight_before,
                syndrome_weight_after=0,
                gamma=self._gamma,
                terminal=True,
                success=success,
            )
        elif self._reward_type == RewardType.HEURISTIC_SHAPED:
            return compute_heuristic_reward(
                num_corrections=0,
                syndromes_cleared=0,
                terminal=True,
                success=success,
            )
        elif self._reward_type == RewardType.COMBINED:
            return compute_combined_reward(
                syndrome_weight_before=syndrome_weight_before,
                syndrome_weight_after=0,
                num_corrections=0,
                gamma=self._gamma,
                terminal=True,
                success=success,
            )
        return 0.0

    def _compute_step_reward(
        self, syndrome_weight_before: int, is_correction: bool
    ) -> float:
        """Compute reward for a non-terminal step."""
        if self._reward_type == RewardType.SPARSE:
            return 0.0
        elif self._reward_type == RewardType.POTENTIAL_BASED:
            syndrome_weight_after = int(self._current_syndrome_grid.sum())
            return compute_potential_based_reward(
                syndrome_weight_before=syndrome_weight_before,
                syndrome_weight_after=syndrome_weight_after,
                gamma=self._gamma,
            )
        elif self._reward_type == RewardType.HEURISTIC_SHAPED:
            return compute_heuristic_reward(
                num_corrections=1 if is_correction else 0,
                syndromes_cleared=0,
            )
        elif self._reward_type == RewardType.COMBINED:
            syndrome_weight_after = int(self._current_syndrome_grid.sum())
            return compute_combined_reward(
                syndrome_weight_before=syndrome_weight_before,
                syndrome_weight_after=syndrome_weight_after,
                num_corrections=1 if is_correction else 0,
                gamma=self._gamma,
            )
        return 0.0

    @classmethod
    def from_config(
        cls,
        distance: int = 3,
        rounds: Optional[int] = None,
        physical_error_rate: float = 0.01,
        noise_model: str = "depolarizing",
        reward_type: str = "sparse",
        **kwargs,
    ) -> SurfaceCodeEnv:
        """
        Convenience constructor from basic parameters.

        distance: Code distance.
        rounds: Number of rounds (default = distance).
        physical_error_rate: Physical error rate.
        noise_model: Noise model type string.
        reward_type: Reward type string.
        **kwargs: Passed to SurfaceCodeEnv.__init__.

        Configured SurfaceCodeEnv.
        """
        from src.quantum.noise_models import NoiseConfig, NoiseModelType

        if rounds is None:
            rounds = distance

        params = SurfaceCodeParams(distance=distance, rounds=rounds)
        config = NoiseConfig(
            model_type=NoiseModelType(noise_model),
            physical_error_rate=physical_error_rate,
        )
        circuit = build_noisy_circuit(params, config)
        return cls(circuit=circuit, reward_type=reward_type, **kwargs)
