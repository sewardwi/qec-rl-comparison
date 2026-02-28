"""Tests for surface_code.py — circuit generation, sampling, and structure analysis."""

import numpy as np
import pytest
import stim

from src.quantum.surface_code import (
    SurfaceCodeCircuit,
    SurfaceCodeParams,
    analyze_circuit_structure,
    generate_circuit,
    get_qubit_layout,
)


# --- SurfaceCodeParams ---


class TestSurfaceCodeParams:
    def test_basic_properties(self):
        params = SurfaceCodeParams(distance=3, rounds=3)
        assert params.num_data_qubits == 9
        assert params.num_stabilizers == 8
        assert params.task_string == "surface_code:rotated_memory_z"

    def test_basis_x(self):
        params = SurfaceCodeParams(distance=5, rounds=5, basis="X")
        assert params.task_string == "surface_code:rotated_memory_x"

    def test_invalid_distance(self):
        with pytest.raises(ValueError, match="distance"):
            SurfaceCodeParams(distance=0, rounds=1)

    def test_invalid_basis(self):
        with pytest.raises(ValueError, match="basis"):
            SurfaceCodeParams(distance=3, rounds=3, basis="Y")

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_stabilizer_count(self, d):
        params = SurfaceCodeParams(distance=d, rounds=d)
        assert params.num_stabilizers == d**2 - 1


# --- Circuit generation ---


class TestGenerateCircuit:
    @pytest.fixture
    def depolarizing_noise(self):
        return {
            "after_clifford_depolarization": 0.01,
            "before_round_data_depolarization": 0.01,
            "before_measure_flip_probability": 0.01,
            "after_reset_flip_probability": 0.01,
        }

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_generates_valid_circuit(self, d, depolarizing_noise):
        params = SurfaceCodeParams(distance=d, rounds=d)
        sc = generate_circuit(params, depolarizing_noise)
        assert isinstance(sc, SurfaceCodeCircuit)
        assert isinstance(sc.circuit, stim.Circuit)
        assert sc.num_detectors > 0
        assert sc.num_observables >= 1

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_detector_count(self, d, depolarizing_noise):
        """num_detectors should be approximately (d^2 - 1) * rounds."""
        params = SurfaceCodeParams(distance=d, rounds=d)
        sc = generate_circuit(params, depolarizing_noise)
        expected = (d**2 - 1) * d
        # Allow some tolerance due to boundary effects in Stim
        assert sc.num_detectors == expected, (
            f"d={d}: expected {expected} detectors, got {sc.num_detectors}"
        )


# --- Sampling ---


class TestSampling:
    @pytest.fixture
    def circuit_d3(self):
        params = SurfaceCodeParams(distance=3, rounds=3)
        noise = {"after_clifford_depolarization": 0.01}
        return generate_circuit(params, noise)

    def test_sample_shapes(self, circuit_d3):
        syndromes, observables = circuit_d3.sample(shots=100, seed=42)
        assert syndromes.shape == (100, circuit_d3.num_detectors)
        assert observables.shape == (100, circuit_d3.num_observables)
        assert syndromes.dtype == np.bool_
        assert observables.dtype == np.bool_

    def test_sample_deterministic_with_seed(self, circuit_d3):
        """Same seed should give same results (fresh sampler each time)."""
        # Need fresh circuit objects to avoid cached sampler
        params = SurfaceCodeParams(distance=3, rounds=3)
        noise = {"after_clifford_depolarization": 0.01}

        sc1 = generate_circuit(params, noise)
        s1, o1 = sc1.sample(shots=50, seed=42)

        sc2 = generate_circuit(params, noise)
        s2, o2 = sc2.sample(shots=50, seed=42)

        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(o1, o2)

    def test_noiseless_has_no_detections(self):
        """A noiseless circuit should produce all-zero syndromes."""
        params = SurfaceCodeParams(distance=3, rounds=3)
        sc = generate_circuit(params, {})
        syndromes, observables = sc.sample(shots=100, seed=42)
        assert syndromes.sum() == 0
        assert observables.sum() == 0

    def test_noisy_has_some_detections(self):
        """A noisy circuit should produce some non-zero syndromes."""
        params = SurfaceCodeParams(distance=3, rounds=3)
        noise = {"after_clifford_depolarization": 0.05}
        sc = generate_circuit(params, noise)
        syndromes, _ = sc.sample(shots=1000, seed=42)
        assert syndromes.sum() > 0


# --- Detector coordinates ---


class TestDetectorCoordinates:
    def test_coordinates_exist(self):
        params = SurfaceCodeParams(distance=3, rounds=3)
        noise = {"after_clifford_depolarization": 0.01}
        sc = generate_circuit(params, noise)
        coords = sc.detector_coordinates
        assert len(coords) == sc.num_detectors
        # Each coordinate should have at least 2 values (x, y) or 3 (x, y, t)
        for det_idx, coord in coords.items():
            assert len(coord) >= 2

    @pytest.mark.parametrize("d", [3, 5])
    def test_coordinates_have_time_dimension(self, d):
        params = SurfaceCodeParams(distance=d, rounds=d)
        noise = {"after_clifford_depolarization": 0.01}
        sc = generate_circuit(params, noise)
        coords = sc.detector_coordinates
        # Stim generates d+1 unique time values for d rounds:
        # d rounds of mid-circuit comparisons + 1 final boundary comparison
        time_vals = set()
        for coord in coords.values():
            if len(coord) >= 3:
                time_vals.add(coord[2])
        assert len(time_vals) == d + 1, (
            f"Expected {d + 1} unique time values, got {len(time_vals)}"
        )


# --- DEM ---


class TestDetectorErrorModel:
    def test_dem_is_valid(self):
        params = SurfaceCodeParams(distance=3, rounds=3)
        noise = {"after_clifford_depolarization": 0.01}
        sc = generate_circuit(params, noise)
        dem = sc.detector_error_model
        assert isinstance(dem, stim.DetectorErrorModel)
        assert len(dem) > 0


# --- Circuit structure analysis ---


class TestAnalyzeCircuitStructure:
    def test_analysis_keys(self):
        params = SurfaceCodeParams(distance=3, rounds=3)
        noise = {"after_clifford_depolarization": 0.01}
        sc = generate_circuit(params, noise)
        result = analyze_circuit_structure(sc)
        expected_keys = {
            "distance",
            "rounds",
            "num_qubits",
            "num_data_qubits",
            "num_stabilizers",
            "num_detectors",
            "num_observables",
            "num_error_mechanisms",
            "gate_counts",
        }
        assert set(result.keys()) == expected_keys

    def test_data_qubits_consistent(self):
        params = SurfaceCodeParams(distance=5, rounds=5)
        noise = {"after_clifford_depolarization": 0.01}
        sc = generate_circuit(params, noise)
        result = analyze_circuit_structure(sc)
        assert result["num_data_qubits"] == 25
        assert result["num_stabilizers"] == 24
        assert result["distance"] == 5


# --- Qubit layout ---


class TestQubitLayout:
    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_layout_has_coordinates(self, d):
        layout = get_qubit_layout(d)
        assert "all_qubit_coords" in layout
        assert layout["all_qubit_coords"].ndim == 2
        assert layout["all_qubit_coords"].shape[1] == 2
