"""Syndrome extraction, spatial reshaping, and visualization.

The raw detector output from Stim is a flat boolean array. This module
reshapes it into spatially meaningful representations for CNN input and
analysis.

For a distance-d rotated surface code with r rounds:
  - Detectors have (x, y, t) coordinates from the circuit
  - The spatial syndrome at each round maps onto a grid determined
    by the unique (x, y) detector positions
  - Grid dimensions depend on distance:
      d=3: ~5x5,  d=5: ~9x9,  d=7: ~13x13
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import stim


class SyndromeGrid:
    """Precomputed mapping from flat detector indices to spatial grid.

    Constructed once per circuit, then used to reshape any number of
    syndrome samples efficiently.

    Attributes:
        rounds: Number of syndrome extraction rounds.
        grid_h: Height of the spatial grid.
        grid_w: Width of the spatial grid.
        num_detectors: Total number of detectors in the circuit.
    """

    def __init__(self, circuit: stim.Circuit):
        coords = circuit.get_detector_coordinates()
        self.num_detectors = circuit.num_detectors

        if not coords:
            raise ValueError("Circuit has no detector coordinates.")

        # Extract spatial and temporal coordinates
        xs = []
        ys = []
        ts = []
        for det_idx in range(self.num_detectors):
            c = coords[det_idx]
            xs.append(c[0])
            ys.append(c[1])
            ts.append(c[2] if len(c) > 2 else 0.0)

        # Build sorted unique coordinate axes
        unique_x = sorted(set(xs))
        unique_y = sorted(set(ys))
        unique_t = sorted(set(ts))

        self.grid_w = len(unique_x)
        self.grid_h = len(unique_y)
        self.rounds = len(unique_t)

        x_to_col = {x: i for i, x in enumerate(unique_x)}
        y_to_row = {y: i for i, y in enumerate(unique_y)}
        t_to_round = {t: i for i, t in enumerate(unique_t)}

        # Build index mapping: detector_idx -> (round, row, col)
        self._det_round = np.zeros(self.num_detectors, dtype=np.int32)
        self._det_row = np.zeros(self.num_detectors, dtype=np.int32)
        self._det_col = np.zeros(self.num_detectors, dtype=np.int32)

        for det_idx in range(self.num_detectors):
            self._det_round[det_idx] = t_to_round[ts[det_idx]]
            self._det_row[det_idx] = y_to_row[ys[det_idx]]
            self._det_col[det_idx] = x_to_col[xs[det_idx]]

        # Store raw coordinates for visualization
        self._unique_x = unique_x
        self._unique_y = unique_y
        self._coords = coords

    def reshape_single(self, flat_syndrome: np.ndarray) -> np.ndarray:
        """Reshape a single flat syndrome to spatial grid.

        Args:
            flat_syndrome: Bool array of shape (num_detectors,).

        Returns:
            Grid of shape (rounds, grid_h, grid_w) with float32 values.
        """
        grid = np.zeros(
            (self.rounds, self.grid_h, self.grid_w), dtype=np.float32
        )
        grid[self._det_round, self._det_row, self._det_col] = flat_syndrome.astype(
            np.float32
        )
        return grid

    def reshape_batch(self, flat_batch: np.ndarray) -> np.ndarray:
        """Reshape a batch of flat syndromes to spatial grids.

        Args:
            flat_batch: Bool array of shape (batch, num_detectors).

        Returns:
            Grid of shape (batch, rounds, grid_h, grid_w) with float32 values.
        """
        batch_size = flat_batch.shape[0]
        grids = np.zeros(
            (batch_size, self.rounds, self.grid_h, self.grid_w),
            dtype=np.float32,
        )
        grids[
            :, self._det_round, self._det_row, self._det_col
        ] = flat_batch.astype(np.float32)
        return grids


def compute_syndrome_diff(grid: np.ndarray) -> np.ndarray:
    """Compute temporal difference (XOR) between consecutive syndrome rounds.

    In surface code decoding, the change between consecutive rounds
    highlights where new errors occurred.

    Args:
        grid: Shape (..., rounds, grid_h, grid_w).

    Returns:
        Shape (..., rounds-1, grid_h, grid_w) of XOR between consecutive rounds.
    """
    # Works on both single and batched grids via ellipsis
    return np.logical_xor(grid[..., 1:, :, :], grid[..., :-1, :, :]).astype(
        np.float32
    )


def get_syndrome_statistics(
    syndromes: np.ndarray, syndrome_grid: SyndromeGrid
) -> dict:
    """Compute statistical properties of syndrome data.

    Args:
        syndromes: Flat bool array of shape (num_shots, num_detectors).
        syndrome_grid: SyndromeGrid instance for this circuit.

    Returns:
        Dict with:
            detection_fraction: fraction of detectors triggered (averaged over shots)
            mean_syndrome_weight: average number of triggered detectors per shot
            std_syndrome_weight: std dev of syndrome weight
            per_detector_rates: per-detector trigger rates, shape (num_detectors,)
    """
    weights = syndromes.sum(axis=1)
    per_det = syndromes.mean(axis=0)

    return {
        "detection_fraction": float(syndromes.mean()),
        "mean_syndrome_weight": float(weights.mean()),
        "std_syndrome_weight": float(weights.std()),
        "per_detector_rates": per_det,
        "num_shots": syndromes.shape[0],
    }


def visualize_syndrome(
    grid: np.ndarray,
    distance: int,
    round_idx: int = 0,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot a single syndrome round on a grid.

    Triggered detectors are shown as filled red squares; inactive
    detectors as open blue squares.

    Args:
        grid: Shape (rounds, grid_h, grid_w) from SyndromeGrid.reshape_single.
        distance: Code distance (for labeling).
        round_idx: Which round to plot.
        ax: Matplotlib axes; created if None.
        title: Optional plot title.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    round_data = grid[round_idx]
    h, w = round_data.shape

    # Plot grid
    for row in range(h):
        for col in range(w):
            val = round_data[row, col]
            color = "red" if val > 0.5 else "lightblue"
            edge = "darkred" if val > 0.5 else "steelblue"
            ax.add_patch(
                plt.Rectangle(
                    (col - 0.4, row - 0.4),
                    0.8,
                    0.8,
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=1.5,
                )
            )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)  # Flip y-axis so (0,0) is top-left
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    if title is None:
        title = f"Syndrome d={distance}, round={round_idx}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def visualize_syndrome_spacetime(
    grid: np.ndarray,
    distance: int,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """3D scatter plot of triggered detectors across all rounds.

    Args:
        grid: Shape (rounds, grid_h, grid_w) from SyndromeGrid.reshape_single.
        distance: Code distance (for labeling).
        ax: 3D matplotlib axes; created if None.

    Returns:
        The matplotlib Axes3D object.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    rounds, h, w = grid.shape
    triggered = np.argwhere(grid > 0.5)  # (round, row, col)

    if len(triggered) > 0:
        ax.scatter(
            triggered[:, 2],  # col -> x
            triggered[:, 1],  # row -> y
            triggered[:, 0],  # round -> z (time)
            c="red",
            s=50,
            alpha=0.7,
            depthshade=True,
        )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("Round")
    ax.set_title(f"Syndrome space-time, d={distance}")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_zlim(-0.5, rounds - 0.5)

    return ax
