"""Error budget, DEM analysis, and circuit structure analysis.

Tools for understanding the quantum simulation layer — showing
understanding of the physics, not just the RL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import stim

from src.quantum.noise_models import NoiseConfig, NoiseModelType, build_noisy_circuit
from src.quantum.surface_code import SurfaceCodeCircuit, SurfaceCodeParams


def analyze_detector_error_model(dem: stim.DetectorErrorModel) -> dict:
    """Parse a detector error model and extract statistics.

    Args:
        dem: Stim DetectorErrorModel.

    Returns:
        Dict with error mechanism counts, weight distribution,
        detector connectivity, and boundary error info.
    """
    num_errors = 0
    error_weights = []
    num_boundary = 0
    detector_degrees = {}

    for instruction in dem.flattened():
        if instruction.type == "error":
            num_errors += 1
            targets = instruction.targets_copy()
            detectors = [t for t in targets if t.is_relative_detector_id()]
            weight = len(detectors)
            error_weights.append(weight)

            has_logical = any(t.is_logical_observable_id() for t in targets)
            if has_logical:
                num_boundary += 1

            for det in detectors:
                d_id = det.val
                detector_degrees[d_id] = detector_degrees.get(d_id, 0) + 1

    weight_arr = np.array(error_weights) if error_weights else np.array([0])
    degrees = np.array(list(detector_degrees.values())) if detector_degrees else np.array([0])

    return {
        "num_error_mechanisms": num_errors,
        "num_boundary_errors": num_boundary,
        "boundary_fraction": num_boundary / max(num_errors, 1),
        "weight_distribution": {
            int(w): int(c) for w, c in
            zip(*np.unique(weight_arr, return_counts=True))
        },
        "mean_error_weight": float(weight_arr.mean()),
        "mean_detector_degree": float(degrees.mean()),
        "max_detector_degree": int(degrees.max()),
        "num_detectors": len(detector_degrees),
    }


def compute_error_budget(
    params: SurfaceCodeParams,
    physical_error_rate: float = 0.01,
    num_shots: int = 10_000,
    seed: int = 42,
) -> dict:
    """Estimate each noise channel's contribution to logical error rate.

    Toggles channels on/off and compares logical error rates.

    Args:
        params: Surface code parameters.
        physical_error_rate: Base physical error rate.
        num_shots: Shots per evaluation.
        seed: Random seed.

    Returns:
        Dict mapping noise channel -> logical error rate.
    """
    from src.quantum.decoder_baseline import MWPMDecoder

    p = physical_error_rate
    channels = {
        "full": {
            "after_clifford_depolarization": p,
            "before_round_data_depolarization": p,
            "before_measure_flip_probability": p,
            "after_reset_flip_probability": p,
        },
        "gate_only": {
            "after_clifford_depolarization": p,
            "before_round_data_depolarization": 0,
            "before_measure_flip_probability": 0,
            "after_reset_flip_probability": 0,
        },
        "idle_only": {
            "after_clifford_depolarization": 0,
            "before_round_data_depolarization": p,
            "before_measure_flip_probability": 0,
            "after_reset_flip_probability": 0,
        },
        "measurement_only": {
            "after_clifford_depolarization": 0,
            "before_round_data_depolarization": 0,
            "before_measure_flip_probability": p,
            "after_reset_flip_probability": 0,
        },
        "reset_only": {
            "after_clifford_depolarization": 0,
            "before_round_data_depolarization": 0,
            "before_measure_flip_probability": 0,
            "after_reset_flip_probability": p,
        },
    }

    from src.quantum.surface_code import generate_circuit

    results = {}
    for name, noise_params in channels.items():
        sc = generate_circuit(params, noise_params)
        decoder = MWPMDecoder.from_surface_code_circuit(sc)
        syn, obs = sc.sample(num_shots, seed=seed)
        result = decoder.evaluate(syn, obs)
        results[name] = result["logical_error_rate"]

    return results


def scaling_analysis(
    distances: list[int],
    noise_config: NoiseConfig | None = None,
) -> pd.DataFrame:
    """Circuit complexity vs distance: qubits, detectors, DEM size, depth.

    Args:
        distances: Code distances to analyze.
        noise_config: Noise config (default: depolarizing at p=0.01).

    Returns:
        DataFrame with circuit statistics per distance.
    """
    if noise_config is None:
        noise_config = NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01)

    rows = []
    for d in distances:
        params = SurfaceCodeParams(distance=d, rounds=d)
        circuit = build_noisy_circuit(params, noise_config)

        dem = circuit.detector_error_model
        dem_stats = analyze_detector_error_model(dem)

        # Count gates
        gate_counts = {}
        for inst in circuit.circuit.flattened():
            name = inst.name
            if name not in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE",
                            "QUBIT_COORDS", "SHIFT_COORDS"):
                gate_counts[name] = gate_counts.get(name, 0) + 1

        rows.append({
            "distance": d,
            "data_qubits": d * d,
            "total_qubits": circuit.circuit.num_qubits,
            "rounds": d,
            "detectors": circuit.num_detectors,
            "action_space": 3 * d * d + 1,
            "obs_channels": d + 2,  # syndrome_rounds + 2 correction channels
            "dem_errors": dem_stats["num_error_mechanisms"],
            "dem_boundary": dem_stats["num_boundary_errors"],
            "total_gates": sum(gate_counts.values()),
        })

    return pd.DataFrame(rows)
