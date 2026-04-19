"""
Custom CNN feature extractors for syndrome grids.

Standard Atari-style CNNs (8x8, 4x4 kernels) would collapse these tiny grids.
We use 3x3 kernels with padding=1 to preserve spatial dimensions, and no pooling.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SyndromeCNN(BaseFeaturesExtractor):
    """
    3-layer CNN for syndrome grids.

    Designed for small spatial grids (4x4 to 8x8) with multiple channels
    (syndrome rounds + correction history). Uses 3x3 kernels with padding=1
    to preserve spatial dimensions through all layers.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]
        h, w = observation_space.shape[1], observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * h * w, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)
