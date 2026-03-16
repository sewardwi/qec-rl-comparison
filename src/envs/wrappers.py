"""Observation wrappers for the surface code environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenSyndromeWrapper(gym.ObservationWrapper):
    """Flatten (C, H, W) observation to (C*H*W,) for MLP ablation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        flat_dim = int(np.prod(obs_shape))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten()


class PadToSquareWrapper(gym.ObservationWrapper):
    """Pad non-square grids to the nearest square for uniform CNN input.

    Pads (C, H, W) to (C, max(H,W), max(H,W)) with zeros.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        c, h, w = self.observation_space.shape
        side = max(h, w)
        self._pad_h = side - h
        self._pad_w = side - w
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(c, side, side),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self._pad_h == 0 and self._pad_w == 0:
            return obs
        return np.pad(
            obs,
            ((0, 0), (0, self._pad_h), (0, self._pad_w)),
            mode="constant",
            constant_values=0.0,
        )
