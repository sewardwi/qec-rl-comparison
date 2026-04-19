"""
Systematic decoder evaluation harness.

Runs all registered decoders across a parameter grid on IDENTICAL syndrome
sets for fair comparison. Produces a DataFrame with per-decoder, per-distance,
per-error-rate logical error rates and confidence intervals.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import clopper_pearson, decoder_ratio
from src.quantum.noise_models import NoiseConfig, NoiseModelType, build_noisy_circuit
from src.quantum.surface_code import SurfaceCodeParams


class DecoderEvaluator:
    """
    Runs all decoders across a parameter grid on identical syndrome sets.

    Usage:
        evaluator = DecoderEvaluator(
            distances=[3, 5],
            error_rates=[0.001, 0.01, 0.1],
            noise_configs=[NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01)],
        )
        evaluator.add_decoder("mwpm", mwpm_decode_fn)
        evaluator.add_decoder("dqn", dqn_decode_fn)
        results = evaluator.run()
    """

    def __init__(
        self,
        distances: list[int],
        error_rates: list[float],
        noise_configs: list[NoiseConfig],
        num_eval_shots: int = 10_000,
        seed: int = 42,
    ):
        """
        distances: Code distances to evaluate.
        error_rates: Physical error rates to sweep.
        noise_configs: Noise model configurations. Each config's
            physical_error_rate is overridden by the error_rates sweep.
        num_eval_shots: Number of syndrome samples per evaluation point.
        seed: Seed for deterministic syndrome sampling.
        """
        self.distances = distances
        self.error_rates = error_rates
        self.noise_configs = noise_configs
        self.num_eval_shots = num_eval_shots
        self.seed = seed
        self._decoders: dict[str, Callable] = {}

    def add_decoder(self, name: str, decode_fn: Callable) -> None:
        """
        Register a decoder for evaluation

        name: Decoder name (e.g., "mwpm", "dqn", "ppo")
        decode_fn: Callable(syndromes, circuit) -> predicted_observable_flips
            syndromes: bool array (num_shots, num_detectors)
            circuit: SurfaceCodeCircuit
            Returns: bool array (num_shots, num_observables)
        """
        self._decoders[name] = decode_fn

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run all decoders across the full parameter grid

        DataFrame with columns: decoder, distance, physical_error_rate,
        noise_model, logical_error_rate, ci_lower, ci_upper,
        num_shots, num_errors.
        """
        rows = []

        for noise_config in self.noise_configs:
            noise_name = noise_config.model_type.value
            for d in self.distances:
                for p in self.error_rates:
                    config = noise_config.with_error_rate(p)
                    params = SurfaceCodeParams(distance=d, rounds=d)

                    try:
                        circuit = build_noisy_circuit(params, config)
                    except NotImplementedError:
                        if verbose:
                            print(
                                f"  Skipping {noise_name} d={d} p={p:.4f} "
                                f"(not implemented)"
                            )
                        continue

                    # Sample deterministic evaluation set
                    syndromes, observables = circuit.sample(
                        self.num_eval_shots, seed=self.seed
                    )

                    if verbose:
                        print(
                            f"  Evaluating d={d}, p={p:.4f}, "
                            f"noise={noise_name}...",
                            end="",
                        )

                    for decoder_name, decode_fn in self._decoders.items():
                        predictions = decode_fn(syndromes, circuit)
                        errors = np.any(predictions != observables, axis=1)
                        n_errors = int(errors.sum())
                        n_shots = len(errors)
                        ler = n_errors / n_shots if n_shots > 0 else 0.0
                        ci_lo, ci_hi = clopper_pearson(n_errors, n_shots)

                        rows.append(
                            {
                                "decoder": decoder_name,
                                "distance": d,
                                "physical_error_rate": p,
                                "noise_model": noise_name,
                                "logical_error_rate": ler,
                                "ci_lower": ci_lo,
                                "ci_upper": ci_hi,
                                "num_shots": n_shots,
                                "num_errors": n_errors,
                            }
                        )

                    if verbose:
                        print(" done")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                ["noise_model", "decoder", "distance", "physical_error_rate"]
            ).reset_index(drop=True)
        return df


def make_mwpm_decode_fn():
    """
    Create a MWPM decode function compatible with DecoderEvaluator.

    Callable(syndromes, circuit) -> predictions.
    """
    from src.quantum.decoder_baseline import MWPMDecoder

    def decode(syndromes, circuit):
        decoder = MWPMDecoder.from_surface_code_circuit(circuit)
        return decoder.decode_batch(syndromes)

    return decode


def make_rl_decode_fn(model, env_cls, env_kwargs: Optional[dict] = None):
    """
    Create an RL agent decode function compatible with DecoderEvaluator.

    The RL agent plays episodes in the environment. Each episode corresponds
    to one syndrome instance. Success/failure is determined by the environment.

    model: Trained SB3 model with .predict(obs, deterministic=True).
    env_cls: Environment class (e.g., SurfaceCodeEnv).
    env_kwargs: Additional kwargs for env construction.

    Callable(syndromes, circuit) -> predictions.
    """

    def decode(syndromes, circuit):
        kwargs = env_kwargs or {}
        env = env_cls(circuit=circuit, **kwargs)
        n_shots = len(syndromes)
        predictions = np.zeros((n_shots, circuit.num_observables), dtype=bool)

        for i in range(n_shots):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            # if agent succeeded the correction was correct (no flip needed)
            # if failed the logical observable was flipped
            predictions[i, 0] = not info.get("success", False)

        return predictions

    return decode
