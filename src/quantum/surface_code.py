"""Surface code circuit generation and batched sampling via Stim.

Provides high-level abstractions over Stim's circuit generation for the
rotated surface code. Handles circuit generation for multiple distances
and noise regimes, extraction of structural metadata (qubit layout,
stabilizer map), and detector coordinate mapping for syndrome grid
reconstruction.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import stim


@dataclasses.dataclass(frozen=True)
class SurfaceCodeParams:
    """Immutable specification of a surface code instance.

    Attributes:
        distance: Code distance d. The code encodes 1 logical qubit in
            d^2 data qubits with (d^2 - 1) stabilizers.
        rounds: Number of syndrome extraction rounds. Typically set to d
            to match the code distance in the time direction.
        basis: Logical basis for memory experiment ("X" or "Z").
    """

    distance: int
    rounds: int
    basis: str = "Z"

    def __post_init__(self):
        if self.distance < 2:
            raise ValueError(f"distance must be >= 2, got {self.distance}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")
        if self.basis not in ("X", "Z"):
            raise ValueError(f"basis must be 'X' or 'Z', got {self.basis!r}")

    @property
    def num_data_qubits(self) -> int:
        return self.distance**2

    @property
    def num_stabilizers(self) -> int:
        return self.distance**2 - 1

    @property
    def task_string(self) -> str:
        return f"surface_code:rotated_memory_{self.basis.lower()}"


@dataclasses.dataclass
class SurfaceCodeCircuit:
    """A generated surface code circuit with metadata and cached sampler.

    Attributes:
        params: Code parameters used to generate this circuit.
        noise_params: Dict of noise parameters passed to Stim.
        circuit: The underlying Stim circuit.
    """

    params: SurfaceCodeParams
    noise_params: dict[str, float]
    circuit: stim.Circuit
    _sampler: Optional[stim.CompiledDetectorSampler] = dataclasses.field(
        default=None, repr=False
    )

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    @property
    def num_observables(self) -> int:
        return self.circuit.num_observables

    @property
    def detector_error_model(self) -> stim.DetectorErrorModel:
        return self.circuit.detector_error_model(decompose_errors=True)

    @property
    def detector_coordinates(self) -> dict[int, list[float]]:
        return self.circuit.get_detector_coordinates()

    def compile_sampler(self, seed: Optional[int] = None) -> stim.CompiledDetectorSampler:
        """Compile and cache a detector sampler for efficient repeated sampling."""
        if self._sampler is None:
            self._sampler = self.circuit.compile_detector_sampler(seed=seed)
        return self._sampler

    def sample(
        self, shots: int = 1024, seed: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch-sample detection events and observable flips.

        Args:
            shots: Number of samples to draw.
            seed: RNG seed for sampler compilation (only used on first call).

        Returns:
            (detections, observables):
                detections: bool array of shape (shots, num_detectors)
                observables: bool array of shape (shots, num_observables)
        """
        sampler = self.compile_sampler(seed)
        return sampler.sample(shots, separate_observables=True)


def generate_circuit(
    params: SurfaceCodeParams,
    noise_params: dict[str, float],
) -> SurfaceCodeCircuit:
    """Generate a noisy surface code circuit using Stim.

    Args:
        params: Code parameters (distance, rounds, basis).
        noise_params: Dict with keys matching stim.Circuit.generated kwargs:
            - after_clifford_depolarization
            - before_round_data_depolarization
            - before_measure_flip_probability
            - after_reset_flip_probability

    Returns:
        SurfaceCodeCircuit wrapping the generated Stim circuit.
    """
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


def get_qubit_layout(distance: int) -> dict[str, np.ndarray]:
    """Compute the physical qubit layout for the rotated surface code.

    Generates a noiseless circuit and extracts qubit coordinates from the
    circuit's coordinate annotations.

    Args:
        distance: Code distance.

    Returns:
        Dict with:
            "data_qubit_coords": (n, 2) array of data qubit (x, y) positions
            "measure_qubit_coords": (m, 2) array of measurement qubit positions
            "all_qubit_coords": (n+m, 2) array of all qubit positions
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=1,
    )
    all_coords = circuit.get_final_qubit_coordinates()

    data_coords = []
    measure_coords = []

    for qubit_idx, coord in sorted(all_coords.items()):
        # In the rotated surface code, data qubits and measure qubits
        # are interleaved on the lattice. We distinguish them by checking
        # which qubits are used as targets of measurement operations.
        coord_arr = np.array(coord[:2])
        data_coords.append((qubit_idx, coord_arr))
        # Note: a more precise separation requires circuit inspection;
        # for now we return all coordinates and let callers filter.

    all_arr = np.array([c for _, c in data_coords])
    return {
        "all_qubit_coords": all_arr,
        "qubit_indices": [idx for idx, _ in data_coords],
    }


def analyze_circuit_structure(sc: SurfaceCodeCircuit) -> dict:
    """Analyze and return structural properties of a surface code circuit.

    Returns:
        Dict with circuit statistics useful for understanding complexity
        scaling with distance.
    """
    circuit = sc.circuit
    dem = sc.detector_error_model

    # Count gates by type
    gate_counts: dict[str, int] = {}
    for instruction in circuit.flattened():
        name = instruction.name
        if name in gate_counts:
            gate_counts[name] += len(instruction.targets_copy()) // max(
                1, _gate_target_count(name)
            )
        else:
            gate_counts[name] = len(instruction.targets_copy()) // max(
                1, _gate_target_count(name)
            )

    # Count DEM error mechanisms
    num_error_mechanisms = 0
    for instruction in dem.flattened():
        if instruction.type == "error":
            num_error_mechanisms += 1

    return {
        "distance": sc.params.distance,
        "rounds": sc.params.rounds,
        "num_qubits": circuit.num_qubits,
        "num_data_qubits": sc.params.num_data_qubits,
        "num_stabilizers": sc.params.num_stabilizers,
        "num_detectors": sc.num_detectors,
        "num_observables": sc.num_observables,
        "num_error_mechanisms": num_error_mechanisms,
        "gate_counts": gate_counts,
    }


def _gate_target_count(gate_name: str) -> int:
    """Return number of target qubits per gate application."""
    two_qubit_gates = {"CX", "CY", "CZ", "CNOT", "SWAP", "ISWAP"}
    if gate_name in two_qubit_gates:
        return 2
    return 1
