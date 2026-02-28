"""Tests for syndrome.py — reshaping, statistics, and visualization."""

import numpy as np
import pytest

from src.quantum.surface_code import SurfaceCodeParams, generate_circuit
from src.quantum.syndrome import (
    SyndromeGrid,
    compute_syndrome_diff,
    get_syndrome_statistics,
)


@pytest.fixture
def circuit_d3():
    params = SurfaceCodeParams(distance=3, rounds=3)
    noise = {"after_clifford_depolarization": 0.01}
    return generate_circuit(params, noise)


@pytest.fixture
def circuit_d5():
    params = SurfaceCodeParams(distance=5, rounds=5)
    noise = {"after_clifford_depolarization": 0.01}
    return generate_circuit(params, noise)


class TestSyndromeGrid:
    def test_construction(self, circuit_d3):
        sg = SyndromeGrid(circuit_d3.circuit)
        # Stim generates d+1 unique time slices for d rounds
        assert sg.rounds == 4  # d=3 rounds + 1 boundary
        assert sg.grid_h > 0
        assert sg.grid_w > 0
        assert sg.num_detectors == circuit_d3.num_detectors

    @pytest.mark.parametrize("d", [3, 5])
    def test_grid_dimensions(self, d):
        """Grid should have approximately (2d-1) positions per spatial axis."""
        params = SurfaceCodeParams(distance=d, rounds=d)
        noise = {"after_clifford_depolarization": 0.01}
        sc = generate_circuit(params, noise)
        sg = SyndromeGrid(sc.circuit)

        # Stim generates d+1 unique time slices for d rounds
        assert sg.rounds == d + 1
        # Grid dimensions depend on detector placement; verify reasonable range
        assert sg.grid_h >= d - 1
        assert sg.grid_w >= d - 1
        assert sg.grid_h <= 2 * d
        assert sg.grid_w <= 2 * d

    def test_reshape_single(self, circuit_d3):
        sg = SyndromeGrid(circuit_d3.circuit)
        syndromes, _ = circuit_d3.sample(shots=1, seed=42)
        flat = syndromes[0]

        grid = sg.reshape_single(flat)
        assert grid.shape == (sg.rounds, sg.grid_h, sg.grid_w)
        assert grid.dtype == np.float32

        # Number of nonzero entries in grid should equal syndrome weight
        assert grid.sum() == flat.sum()

    def test_reshape_batch(self, circuit_d3):
        sg = SyndromeGrid(circuit_d3.circuit)
        syndromes, _ = circuit_d3.sample(shots=50, seed=42)

        grids = sg.reshape_batch(syndromes)
        assert grids.shape == (50, sg.rounds, sg.grid_h, sg.grid_w)
        assert grids.dtype == np.float32

        # Check consistency with single reshape
        for i in range(5):
            single = sg.reshape_single(syndromes[i])
            np.testing.assert_array_equal(grids[i], single)

    def test_reshape_preserves_total_weight(self, circuit_d5):
        sg = SyndromeGrid(circuit_d5.circuit)
        syndromes, _ = circuit_d5.sample(shots=100, seed=42)
        grids = sg.reshape_batch(syndromes)

        # Total weight should be preserved
        flat_weights = syndromes.sum(axis=1)
        grid_weights = grids.sum(axis=(1, 2, 3))
        np.testing.assert_array_almost_equal(flat_weights, grid_weights)

    def test_noiseless_gives_zero_grid(self):
        params = SurfaceCodeParams(distance=3, rounds=3)
        sc = generate_circuit(params, {})
        sg = SyndromeGrid(sc.circuit)
        syndromes, _ = sc.sample(shots=10, seed=42)
        grids = sg.reshape_batch(syndromes)
        assert grids.sum() == 0.0


class TestSyndromeDiff:
    def test_diff_shape(self, circuit_d3):
        sg = SyndromeGrid(circuit_d3.circuit)
        syndromes, _ = circuit_d3.sample(shots=1, seed=42)
        grid = sg.reshape_single(syndromes[0])

        diff = compute_syndrome_diff(grid)
        assert diff.shape == (sg.rounds - 1, sg.grid_h, sg.grid_w)
        assert diff.dtype == np.float32

    def test_diff_batch_shape(self, circuit_d3):
        sg = SyndromeGrid(circuit_d3.circuit)
        syndromes, _ = circuit_d3.sample(shots=10, seed=42)
        grids = sg.reshape_batch(syndromes)

        diffs = compute_syndrome_diff(grids)
        assert diffs.shape == (10, sg.rounds - 1, sg.grid_h, sg.grid_w)

    def test_identical_rounds_give_zero_diff(self):
        """If all rounds are identical, diff should be all zeros."""
        grid = np.ones((3, 5, 5), dtype=np.float32)
        diff = compute_syndrome_diff(grid)
        assert diff.sum() == 0.0


class TestSyndromeStatistics:
    def test_statistics_keys(self, circuit_d3):
        sg = SyndromeGrid(circuit_d3.circuit)
        syndromes, _ = circuit_d3.sample(shots=100, seed=42)
        stats = get_syndrome_statistics(syndromes, sg)

        assert "detection_fraction" in stats
        assert "mean_syndrome_weight" in stats
        assert "std_syndrome_weight" in stats
        assert "per_detector_rates" in stats
        assert "num_shots" in stats

    def test_noiseless_statistics(self):
        params = SurfaceCodeParams(distance=3, rounds=3)
        sc = generate_circuit(params, {})
        sg = SyndromeGrid(sc.circuit)
        syndromes, _ = sc.sample(shots=100, seed=42)
        stats = get_syndrome_statistics(syndromes, sg)

        assert stats["detection_fraction"] == 0.0
        assert stats["mean_syndrome_weight"] == 0.0

    def test_noisy_statistics(self, circuit_d3):
        sg = SyndromeGrid(circuit_d3.circuit)
        syndromes, _ = circuit_d3.sample(shots=1000, seed=42)
        stats = get_syndrome_statistics(syndromes, sg)

        assert stats["detection_fraction"] > 0.0
        assert stats["mean_syndrome_weight"] > 0.0
        assert stats["num_shots"] == 1000
        assert len(stats["per_detector_rates"]) == circuit_d3.num_detectors
