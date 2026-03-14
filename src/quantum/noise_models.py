"""Noise model library for surface code simulations.

Provides physically-motivated noise models for surface code circuits:
  1. Depolarizing: Standard symmetric X/Y/Z errors (benchmark)
  2. Biased-Z: Dephasing-dominated errors (superconducting qubits)
  3. Measurement-dominated: Readout errors >> gate errors (NISQ devices)
  4. Correlated: Spatially-correlated crosstalk between neighbors

For now, only depolarizing is fully implemented. Others are planned for week 6.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import stim

from src.quantum.surface_code import SurfaceCodeCircuit, SurfaceCodeParams


class NoiseModelType(Enum):
    DEPOLARIZING = "depolarizing"
    BIASED_Z = "biased_z"
    MEASUREMENT = "measurement"
    CORRELATED = "correlated"


@dataclasses.dataclass(frozen=True)
class NoiseConfig:
    """Configuration for a noise model.

    Attributes:
        model_type: Which noise model to use.
        physical_error_rate: Overall physical error rate p.
        bias_ratio: For biased-Z model, ratio pz/px (eta). Default 1.0 (symmetric).
        correlation_strength: For correlated model, crosstalk strength. Default 0.0.
    """

    model_type: NoiseModelType
    physical_error_rate: float
    bias_ratio: float = 1.0
    correlation_strength: float = 0.0

    def __post_init__(self):
        if self.physical_error_rate < 0 or self.physical_error_rate > 1:
            raise ValueError(
                f"physical_error_rate must be in [0, 1], got {self.physical_error_rate}"
            )

    def with_error_rate(self, p: float) -> NoiseConfig:
        """Return a copy with a different physical error rate."""
        return dataclasses.replace(self, physical_error_rate=p)


def build_noisy_circuit(
    params: SurfaceCodeParams, config: NoiseConfig
) -> SurfaceCodeCircuit:
    """Build a noisy surface code circuit for the given noise model.

    Args:
        params: Surface code parameters (distance, rounds, basis).
        config: Noise configuration specifying model type and error rate.

    Returns:
        SurfaceCodeCircuit with noise applied.
    """
    builders = {
        NoiseModelType.DEPOLARIZING: _build_depolarizing,
        NoiseModelType.MEASUREMENT: _build_measurement_dominated,
    }

    builder = builders.get(config.model_type)
    if builder is None:
        raise NotImplementedError(
            f"Noise model {config.model_type.value} not yet implemented. "
            f"Available: {[t.value for t in builders]}"
        )

    return builder(params, config)


def _build_depolarizing(
    params: SurfaceCodeParams, config: NoiseConfig
) -> SurfaceCodeCircuit:
    """Standard depolarizing noise: p applied uniformly to all noise channels.

    Physical motivation: symmetric X/Y/Z errors at equal rates. This is the
    standard benchmark noise model for surface code decoders.
    """
    p = config.physical_error_rate
    noise_params = {
        "after_clifford_depolarization": p,
        "before_round_data_depolarization": p,
        "before_measure_flip_probability": p,
        "after_reset_flip_probability": p,
    }

    circuit = stim.Circuit.generated(
        params.task_string,
        distance=params.distance,
        rounds=params.rounds,
        **noise_params,
    )

    return SurfaceCodeCircuit(
        params=params,
        noise_params=noise_params,
        circuit=circuit,
    )


def _build_measurement_dominated(
    params: SurfaceCodeParams, config: NoiseConfig
) -> SurfaceCodeCircuit:
    """Measurement-dominated noise: readout errors >> gate errors.

    Physical motivation: in many NISQ devices, measurement fidelity is the
    dominant error source, with gate errors roughly 10x lower.
    """
    p = config.physical_error_rate
    noise_params = {
        "after_clifford_depolarization": p / 10,
        "before_round_data_depolarization": p / 10,
        "before_measure_flip_probability": p,
        "after_reset_flip_probability": p / 10,
    }

    circuit = stim.Circuit.generated(
        params.task_string,
        distance=params.distance,
        rounds=params.rounds,
        **noise_params,
    )

    return SurfaceCodeCircuit(
        params=params,
        noise_params=noise_params,
        circuit=circuit,
    )
