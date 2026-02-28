"""Tests for decoder_baseline.py — MWPM decoder and Sinter benchmarking."""

import numpy as np
import pytest

from src.quantum.decoder_baseline import MWPMDecoder, clopper_pearson
from src.quantum.surface_code import SurfaceCodeParams, generate_circuit


@pytest.fixture
def circuit_d3():
    params = SurfaceCodeParams(distance=3, rounds=3)
    noise = {
        "after_clifford_depolarization": 0.01,
        "before_round_data_depolarization": 0.01,
        "before_measure_flip_probability": 0.01,
        "after_reset_flip_probability": 0.01,
    }
    return generate_circuit(params, noise)


class TestClopperPearson:
    def test_zero_errors(self):
        lo, hi = clopper_pearson(0, 1000)
        assert lo == 0.0
        assert hi > 0.0

    def test_all_errors(self):
        lo, hi = clopper_pearson(1000, 1000)
        assert lo < 1.0
        assert hi == 1.0

    def test_half_errors(self):
        lo, hi = clopper_pearson(500, 1000)
        assert 0.45 < lo < 0.5
        assert 0.5 < hi < 0.55

    def test_zero_shots(self):
        lo, hi = clopper_pearson(0, 0)
        assert lo == 0.0
        assert hi == 1.0


class TestMWPMDecoder:
    def test_from_circuit(self, circuit_d3):
        decoder = MWPMDecoder.from_circuit(circuit_d3.circuit)
        assert decoder is not None

    def test_from_surface_code_circuit(self, circuit_d3):
        decoder = MWPMDecoder.from_surface_code_circuit(circuit_d3)
        assert decoder is not None

    def test_decode_batch_shape(self, circuit_d3):
        decoder = MWPMDecoder.from_surface_code_circuit(circuit_d3)
        syndromes, observables = circuit_d3.sample(shots=100, seed=42)
        predictions = decoder.decode_batch(syndromes)
        assert predictions.shape == observables.shape

    def test_evaluate_returns_valid_results(self, circuit_d3):
        decoder = MWPMDecoder.from_surface_code_circuit(circuit_d3)
        syndromes, observables = circuit_d3.sample(shots=1000, seed=42)
        results = decoder.evaluate(syndromes, observables)

        assert "logical_error_rate" in results
        assert "ci_lower" in results
        assert "ci_upper" in results
        assert "num_errors" in results
        assert "num_shots" in results

        assert results["num_shots"] == 1000
        assert 0.0 <= results["logical_error_rate"] <= 1.0
        assert results["ci_lower"] <= results["logical_error_rate"]
        assert results["ci_upper"] >= results["logical_error_rate"]

    def test_noiseless_perfect_decoding(self):
        """Noiseless circuit should have zero logical errors."""
        params = SurfaceCodeParams(distance=3, rounds=3)
        sc = generate_circuit(params, {})
        decoder = MWPMDecoder.from_surface_code_circuit(sc)
        syndromes, observables = sc.sample(shots=100, seed=42)
        results = decoder.evaluate(syndromes, observables)
        assert results["logical_error_rate"] == 0.0
        assert results["num_errors"] == 0

    def test_error_rate_below_threshold(self, circuit_d3):
        """At p=0.01, well below threshold, MWPM should have low logical error rate."""
        decoder = MWPMDecoder.from_surface_code_circuit(circuit_d3)
        syndromes, observables = circuit_d3.sample(shots=10000, seed=42)
        results = decoder.evaluate(syndromes, observables)
        # At p=0.01, d=3, logical error rate should be well below 50%
        assert results["logical_error_rate"] < 0.5

    def test_higher_distance_lower_error(self):
        """Larger distance should give lower logical error rate below threshold."""
        noise = {
            "after_clifford_depolarization": 0.005,
            "before_round_data_depolarization": 0.005,
            "before_measure_flip_probability": 0.005,
            "after_reset_flip_probability": 0.005,
        }
        lers = {}
        for d in [3, 5]:
            params = SurfaceCodeParams(distance=d, rounds=d)
            sc = generate_circuit(params, noise)
            decoder = MWPMDecoder.from_surface_code_circuit(sc)
            syndromes, observables = sc.sample(shots=10000, seed=42)
            results = decoder.evaluate(syndromes, observables)
            lers[d] = results["logical_error_rate"]

        # d=5 should have lower error rate than d=3 below threshold
        assert lers[5] < lers[3], (
            f"Expected d=5 ({lers[5]:.4f}) < d=3 ({lers[3]:.4f})"
        )
