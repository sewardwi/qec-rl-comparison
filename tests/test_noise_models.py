"""Tests for noise_models.py — noise configuration and circuit building."""

import numpy as np
import pytest

from src.quantum.noise_models import (
    NoiseConfig,
    NoiseModelType,
    build_noisy_circuit,
)
from src.quantum.surface_code import SurfaceCodeParams


class TestNoiseConfig:
    def test_depolarizing_config(self):
        config = NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01)
        assert config.physical_error_rate == 0.01
        assert config.bias_ratio == 1.0

    def test_invalid_error_rate(self):
        with pytest.raises(ValueError):
            NoiseConfig(NoiseModelType.DEPOLARIZING, -0.1)
        with pytest.raises(ValueError):
            NoiseConfig(NoiseModelType.DEPOLARIZING, 1.5)

    def test_with_error_rate(self):
        config = NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01)
        new = config.with_error_rate(0.05)
        assert new.physical_error_rate == 0.05
        assert config.physical_error_rate == 0.01  # Original unchanged

    def test_zero_error_rate(self):
        config = NoiseConfig(NoiseModelType.DEPOLARIZING, 0.0)
        assert config.physical_error_rate == 0.0


class TestBuildNoisyCircuit:
    @pytest.mark.parametrize("distance", [3, 5])
    def test_depolarizing_circuit_generation(self, distance):
        params = SurfaceCodeParams(distance=distance, rounds=distance)
        config = NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01)
        sc = build_noisy_circuit(params, config)

        assert sc.params.distance == distance
        assert sc.circuit.num_detectors == (distance**2 - 1) * distance
        assert sc.circuit.num_observables == 1

    def test_depolarizing_produces_errors(self):
        """Noisy circuit should produce non-trivial syndromes."""
        params = SurfaceCodeParams(distance=3, rounds=3)
        config = NoiseConfig(NoiseModelType.DEPOLARIZING, 0.05)
        sc = build_noisy_circuit(params, config)

        syndromes, observables = sc.sample(1000, seed=42)
        assert syndromes.shape == (1000, sc.circuit.num_detectors)
        # At p=0.05, we expect some detections
        assert syndromes.sum() > 0

    def test_noiseless_circuit(self):
        """Zero noise should produce no detections."""
        params = SurfaceCodeParams(distance=3, rounds=3)
        config = NoiseConfig(NoiseModelType.DEPOLARIZING, 0.0)
        sc = build_noisy_circuit(params, config)

        syndromes, observables = sc.sample(100, seed=42)
        assert syndromes.sum() == 0
        assert observables.sum() == 0

    def test_measurement_dominated(self):
        """Measurement-dominated noise should produce valid circuits."""
        params = SurfaceCodeParams(distance=3, rounds=3)
        config = NoiseConfig(NoiseModelType.MEASUREMENT, 0.05)
        sc = build_noisy_circuit(params, config)
        syndromes, observables = sc.sample(100, seed=42)
        assert syndromes.shape[1] == sc.circuit.num_detectors

    def test_unimplemented_noise_model(self):
        params = SurfaceCodeParams(distance=3, rounds=3)
        config = NoiseConfig(NoiseModelType.BIASED_Z, 0.01)
        with pytest.raises(NotImplementedError):
            build_noisy_circuit(params, config)

    def test_error_rate_scaling(self):
        """Higher error rate should produce more detections on average."""
        params = SurfaceCodeParams(distance=3, rounds=3)

        low = NoiseConfig(NoiseModelType.DEPOLARIZING, 0.001)
        high = NoiseConfig(NoiseModelType.DEPOLARIZING, 0.1)

        sc_low = build_noisy_circuit(params, low)
        sc_high = build_noisy_circuit(params, high)

        syn_low, _ = sc_low.sample(2000, seed=42)
        syn_high, _ = sc_high.sample(2000, seed=42)

        assert syn_low.mean() < syn_high.mean()
