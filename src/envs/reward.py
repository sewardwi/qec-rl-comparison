"""
Reward functions for the surface code decoding environment.

Four reward strategies:
  - sparse: +1/-1 at episode end (ground truth, preserves optimal policy)
  - potential_based: gamma*F(s') - F(s) shaping (provably preserves optimal policy)
  - heuristic_shaped: per-step bonuses (does NOT preserve optimal policy; ablation only)
  - combined: potential-based shaping + correction penalty + sparse terminal (default)
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class RewardType(Enum):
    SPARSE = "sparse"
    POTENTIAL_BASED = "potential_based"
    HEURISTIC_SHAPED = "heuristic_shaped"
    COMBINED = "combined"


def compute_sparse_reward(success: bool) -> float:
    """Terminal reward: +1 for successful correction, -1 for failure."""
    return 1.0 if success else -1.0


def compute_potential_based_reward(
    syndrome_weight_before: int,
    syndrome_weight_after: int,
    gamma: float = 0.99,
    terminal: bool = False,
    success: bool = False,
) -> float:
    """
    Potential-based reward shaping (Ng et al. 1999).

    F(s) = -syndrome_weight(s), so the agent gets positive reward for
    reducing the number of triggered detectors. This provably preserves
    the optimal policy.

    syndrome_weight_before: Number of triggered detectors before action.
    syndrome_weight_after: Number of triggered detectors after action.
    gamma: Discount factor (must match agent's gamma).
    terminal: Whether this is the terminal step.
    success: Whether decoding was successful (only used if terminal).

    Shaped reward value.
    """
    # potential function: F(s) = -syndrome_weight
    phi_before = -syndrome_weight_before
    phi_after = -syndrome_weight_after

    # shaping reward: gamma * F(s') - F(s)
    shaping = gamma * phi_after - phi_before

    if terminal:
        # at terminal state, add the sparse terminal reward
        # F(terminal) = 0 by convention, so shaping = -F(s) = syndrome_weight
        terminal_reward = compute_sparse_reward(success)
        return terminal_reward + shaping
    return shaping


def compute_heuristic_reward(
    num_corrections: int,
    syndromes_cleared: int,
    terminal: bool = False,
    success: bool = False,
) -> float:
    """
    Heuristic reward shaping (for ablation study only).

    NOT guaranteed to preserve optimal policy. Includes:
      - Small penalty per correction action (-0.01) to discourage unnecessary actions
      - Small bonus for each syndrome cleared (+0.005)
      - Terminal sparse reward

    num_corrections: Number of correction actions taken this step.
    syndromes_cleared: Change in number of cleared syndromes this step.
    terminal: Whether this is the terminal step.
    success: Whether decoding was successful.

    Shaped reward value.
    """
    reward = -0.01 * num_corrections + 0.005 * syndromes_cleared
    if terminal:
        reward += compute_sparse_reward(success)
    return reward


def compute_combined_reward(
    syndrome_weight_before: int,
    syndrome_weight_after: int,
    num_corrections: int,
    gamma: float = 0.99,
    terminal: bool = False,
    success: bool = False,
) -> float:
    """
    Combined reward: potential-based shaping + correction penalty + sparse terminal.

    Merges the dense signal of potential-based shaping with a small correction
    penalty to discourage unnecessary actions, plus the sparse terminal outcome.

    Components:
      - Potential shaping: gamma * (-w_after) - (-w_before)
          Positive when syndrome weight decreases, negative when it increases.
      - Correction penalty: -0.01 per correction action
          Discourages applying unnecessary corrections.
      - Terminal: +1/-1 on success/failure (sparse outcome signal).

    This does NOT provably preserve the optimal policy due to the correction
    penalty, but provides the richest per-step signal.

    syndrome_weight_before: Number of triggered syndromes before action.
    syndrome_weight_after: Number of triggered syndromes after action.
    num_corrections: Number of correction actions taken this step (0 or 1).
    gamma: Discount factor (should match agent's gamma).
    terminal: Whether this is the terminal step.
    success: Whether decoding was successful (only used if terminal).

    Combined reward value.
    """
    # potential-based shaping: F(s) = -syndrome_weight
    shaping = gamma * (-syndrome_weight_after) - (-syndrome_weight_before)

    # correction penalty: small cost per correction to discourage excess actions
    correction_penalty = -0.01 * num_corrections

    reward = shaping + correction_penalty

    if terminal:
        reward += compute_sparse_reward(success)

    return reward
