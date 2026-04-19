"""
Noise model library for surface code simulations.

Provides physically-motivated noise models for surface code circuits:
  1. Depolarizing: Standard symmetric X/Y/Z errors (benchmark)
  2. Biased-Z: Dephasing-dominated errors (superconducting qubits)
  3. Measurement-dominated: Readout errors >> gate errors (NISQ devices)
  4. Correlated: Spatially-correlated crosstalk between neighbors
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
    """
    Configuration for a noise model.

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
    """
    Build a noisy surface code circuit for the given noise model.

    params: Surface code parameters (distance, rounds, basis).
    config: Noise configuration specifying model type and error rate.

    SurfaceCodeCircuit with noise applied.
    """
    builders = {
        NoiseModelType.DEPOLARIZING: _build_depolarizing,
        NoiseModelType.MEASUREMENT: _build_measurement_dominated,
        NoiseModelType.BIASED_Z: _build_biased_z,
        NoiseModelType.CORRELATED: _build_correlated,
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
    """
    Standard depolarizing noise: p applied uniformly to all noise channels.

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
    """
    Measurement-dominated noise: readout errors >> gate errors.

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


def _build_biased_z(
    params: SurfaceCodeParams, config: NoiseConfig
) -> SurfaceCodeCircuit:
    """
    Biased-Z (dephasing-dominated) noise.

    Physical motivation: in superconducting qubits (e.g., transmon),
    T2 << T1 means dephasing (Z) errors dominate over bit-flip (X) errors.
    We model this with a bias ratio eta = pz/px, so Z errors are eta times
    more likely.

    Implementation: generate noiseless circuit, then insert custom
    PAULI_CHANNEL_1(px, py, pz) after each gate layer, with pz = eta * px.
    The total error rate per qubit is p = px + py + pz.
    """
    p = config.physical_error_rate
    eta = config.bias_ratio

    # with bias eta: pz = eta * px, py = px, total = px + py + pz = (2 + eta)*px
    px = p / (2 + eta)
    py = px
    pz = eta * px

    # generate noiseless circuit first
    noiseless = stim.Circuit.generated(
        params.task_string,
        distance=params.distance,
        rounds=params.rounds,
    )

    # insert biased noise into the circuit
    noisy = stim.Circuit()
    for instruction in noiseless.flattened():
        noisy.append(instruction)
        name = instruction.name
        targets = instruction.targets_copy()

        # after single-qubit Clifford gates: add biased Pauli channel
        if name in ("H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
                     "SQRT_Y", "SQRT_Y_DAG", "SQRT_Z", "SQRT_Z_DAG"):
            qubit_indices = [t.value for t in targets if not t.is_combiner]
            for q in qubit_indices:
                noisy.append("PAULI_CHANNEL_1", [q], [px, py, pz])

        # after two-qubit gates: add independent biased noise on each qubit
        elif name in ("CX", "CY", "CZ", "CNOT"):
            qubit_indices = [t.value for t in targets if not t.is_combiner]
            for q in qubit_indices:
                noisy.append("PAULI_CHANNEL_1", [q], [px, py, pz])

        # after resets: add biased flip
        elif name in ("R", "RX", "RY", "RZ"):
            qubit_indices = [t.value for t in targets if not t.is_combiner]
            for q in qubit_indices:
                noisy.append("PAULI_CHANNEL_1", [q], [px, py, pz])

    noise_params = {"px": px, "py": py, "pz": pz, "bias_ratio": eta}
    return SurfaceCodeCircuit(
        params=params,
        noise_params=noise_params,
        circuit=noisy,
    )


def _build_correlated(
    params: SurfaceCodeParams, config: NoiseConfig
) -> SurfaceCodeCircuit:
    """
    Correlated (crosstalk) noise model.

    Physical motivation: in multi-qubit processors, two-qubit gates on
    adjacent qubits can induce crosstalk errors. We model this by starting
    with standard depolarizing noise and adding CORRELATED_ERROR instructions
    on pairs of data qubits that are neighbors on the lattice.

    Implementation: generate depolarizing circuit, then append correlated
    errors between neighboring data qubit pairs.
    """
    p = config.physical_error_rate
    p_corr = config.correlation_strength
    if p_corr <= 0:
        p_corr = p / 5  # default: correlation is 20% of physical error rate

    # start with standard depolarizing circuit
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

    # get data qubit pairs that are neighbors
    coords = circuit.get_final_qubit_coordinates()
    qubit_positions = {}
    for q_idx, coord in coords.items():
        qubit_positions[q_idx] = (coord[0], coord[1])

    # find adjacent pairs (Manhattan distance <= 2 on the lattice)
    neighbor_pairs = []
    qubit_list = sorted(qubit_positions.keys())
    for i in range(len(qubit_list)):
        for j in range(i + 1, len(qubit_list)):
            q1, q2 = qubit_list[i], qubit_list[j]
            x1, y1 = qubit_positions[q1]
            x2, y2 = qubit_positions[q2]
            dist = abs(x1 - x2) + abs(y1 - y2)
            if dist <= 2:
                neighbor_pairs.append((q1, q2))

    # append correlated errors after TICK instructions
    noisy = stim.Circuit()
    for instruction in circuit.flattened():
        noisy.append(instruction)
        if instruction.name == "TICK":
            for q1, q2 in neighbor_pairs:
                noisy.append(
                    "CORRELATED_ERROR",
                    [stim.target_x(q1), stim.target_x(q2)],
                    [p_corr],
                )

    noise_params["correlation_strength"] = p_corr
    return SurfaceCodeCircuit(
        params=params,
        noise_params=noise_params,
        circuit=noisy,
    )
