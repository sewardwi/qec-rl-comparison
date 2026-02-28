"""MWPM decoder baseline via PyMatching and Sinter benchmarking.

Provides the gold-standard classical decoder against which RL decoders
are compared. Includes both:
  - MWPMDecoder: for evaluating on shared syndrome sets (fair comparison)
  - run_sinter_benchmark: for high-throughput baseline curve generation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pymatching
import sinter
import stim

from src.quantum.surface_code import (
    SurfaceCodeCircuit,
    SurfaceCodeParams,
    generate_circuit,
)


def clopper_pearson(
    k: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval.

    Args:
        k: Number of successes (errors).
        n: Number of trials (shots).
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        (lower, upper) confidence interval bounds.
    """
    from scipy.stats import beta as beta_dist

    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        lo = 0.0
    else:
        lo = beta_dist.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


class MWPMDecoder:
    """Minimum Weight Perfect Matching decoder wrapping PyMatching.

    Usage:
        decoder = MWPMDecoder.from_circuit(surface_code_circuit)
        predictions = decoder.decode_batch(syndromes)
        results = decoder.evaluate(syndromes, observables)
    """

    def __init__(self, matching: pymatching.Matching):
        self._matching = matching

    @classmethod
    def from_detector_error_model(
        cls, dem: stim.DetectorErrorModel
    ) -> MWPMDecoder:
        matching = pymatching.Matching.from_detector_error_model(dem)
        return cls(matching)

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit) -> MWPMDecoder:
        dem = circuit.detector_error_model(decompose_errors=True)
        return cls.from_detector_error_model(dem)

    @classmethod
    def from_surface_code_circuit(
        cls, sc: SurfaceCodeCircuit
    ) -> MWPMDecoder:
        return cls.from_circuit(sc.circuit)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndromes.

        Args:
            syndromes: Bool array of shape (batch, num_detectors).

        Returns:
            predictions: Bool array of shape (batch, num_observables).
        """
        return self._matching.decode_batch(syndromes)

    def evaluate(
        self,
        syndromes: np.ndarray,
        observables: np.ndarray,
        alpha: float = 0.05,
    ) -> dict:
        """Decode and compute logical error rate with confidence interval.

        Args:
            syndromes: Bool array of shape (num_shots, num_detectors).
            observables: Bool array of shape (num_shots, num_observables).
            alpha: Significance level for Clopper-Pearson CI.

        Returns:
            Dict with logical_error_rate, ci_lower, ci_upper, num_errors, num_shots.
        """
        predictions = self.decode_batch(syndromes)
        errors = np.any(predictions != observables, axis=1)
        n_errors = int(errors.sum())
        n_shots = len(errors)
        ler = n_errors / n_shots if n_shots > 0 else 0.0
        ci_lo, ci_hi = clopper_pearson(n_errors, n_shots, alpha)

        return {
            "logical_error_rate": ler,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "num_errors": n_errors,
            "num_shots": n_shots,
        }


def run_sinter_benchmark(
    distances: list[int],
    error_rates: list[float],
    noise_params_fn=None,
    basis: str = "Z",
    num_workers: int = 4,
    max_shots: int = 100_000,
    max_errors: int = 1000,
) -> pd.DataFrame:
    """High-throughput MWPM baseline generation via Sinter.

    Sinter handles parallelization, early stopping after max_errors
    logical errors, and statistical aggregation.

    Args:
        distances: List of code distances to evaluate.
        error_rates: List of physical error rates to sweep.
        noise_params_fn: Optional callable(p) -> dict of Stim noise params.
            Defaults to standard depolarizing (p applied to all channels).
        basis: Logical basis ("X" or "Z").
        num_workers: Number of parallel worker processes.
        max_shots: Maximum samples per (distance, error_rate) point.
        max_errors: Stop after this many logical errors per point.

    Returns:
        DataFrame with columns: distance, physical_error_rate,
        logical_error_rate, num_shots, num_errors.
    """
    if noise_params_fn is None:
        def noise_params_fn(p):
            return {
                "after_clifford_depolarization": p,
                "before_round_data_depolarization": p,
                "before_measure_flip_probability": p,
                "after_reset_flip_probability": p,
            }

    tasks = []
    for d in distances:
        for p in error_rates:
            params = SurfaceCodeParams(distance=d, rounds=d, basis=basis)
            noise_params = noise_params_fn(p)
            sc = generate_circuit(params, noise_params)
            tasks.append(
                sinter.Task(
                    circuit=sc.circuit,
                    json_metadata={"d": d, "p": float(p)},
                )
            )

    results = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        max_shots=max_shots,
        max_errors=max_errors,
        decoders=["pymatching"],
        print_progress=True,
    )

    return _sinter_results_to_dataframe(results)


def _sinter_results_to_dataframe(
    results: list[sinter.TaskStats],
) -> pd.DataFrame:
    """Convert Sinter TaskStats to a pandas DataFrame."""
    rows = []
    for stat in results:
        meta = stat.json_metadata
        n_shots = stat.shots
        n_errors = stat.errors
        ler = n_errors / n_shots if n_shots > 0 else 0.0
        ci_lo, ci_hi = clopper_pearson(n_errors, n_shots)
        rows.append(
            {
                "distance": meta["d"],
                "physical_error_rate": meta["p"],
                "logical_error_rate": ler,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "num_shots": n_shots,
                "num_errors": n_errors,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["distance", "physical_error_rate"]).reset_index(
        drop=True
    )
    return df
