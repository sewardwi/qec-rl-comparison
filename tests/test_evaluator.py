"""Tests for the evaluation pipeline: metrics, evaluator, and decode functions."""

import numpy as np
import pytest

from src.evaluation.evaluator import DecoderEvaluator, make_mwpm_decode_fn
from src.evaluation.metrics import (
    aggregate_seed_results,
    clopper_pearson,
    decoder_ratio,
    estimate_threshold,
    logical_error_rate,
)
from src.quantum.noise_models import NoiseConfig, NoiseModelType


# --- Metrics ---


class TestClopperPearson:
    def test_zero_errors(self):
        lo, hi = clopper_pearson(0, 100)
        assert lo == 0.0
        assert 0.0 < hi < 0.1

    def test_all_errors(self):
        lo, hi = clopper_pearson(100, 100)
        assert 0.9 < lo < 1.0
        assert hi == 1.0

    def test_half_errors(self):
        lo, hi = clopper_pearson(50, 100)
        assert 0.3 < lo < 0.5
        assert 0.5 < hi < 0.7

    def test_zero_shots(self):
        lo, hi = clopper_pearson(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_ci_contains_true_rate(self):
        lo, hi = clopper_pearson(10, 1000)
        assert lo < 0.01 < hi


class TestLogicalErrorRate:
    def test_basic(self):
        result = logical_error_rate(10, 100)
        assert result["logical_error_rate"] == 0.1
        assert result["num_errors"] == 10
        assert result["num_shots"] == 100
        assert result["ci_lower"] < 0.1 < result["ci_upper"]

    def test_zero_shots(self):
        result = logical_error_rate(0, 0)
        assert result["logical_error_rate"] == 0.0


class TestDecoderRatio:
    def test_equal(self):
        assert decoder_ratio(0.05, 0.05) == pytest.approx(1.0)

    def test_rl_better(self):
        assert decoder_ratio(0.03, 0.05) < 1.0

    def test_rl_worse(self):
        assert decoder_ratio(0.10, 0.05) > 1.0

    def test_zero_mwpm(self):
        assert decoder_ratio(0.05, 0.0) == float("inf")


class TestEstimateThreshold:
    def test_crossing(self):
        error_rates = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
        # d=3: worse at low p, better at high p
        # d=5: better at low p, worse at high p
        # crossing around p=0.01
        ler_d3 = np.array([0.001, 0.01, 0.05, 0.3, 0.45])
        ler_d5 = np.array([0.0001, 0.005, 0.08, 0.4, 0.48])

        threshold = estimate_threshold(
            error_rates, {3: ler_d3, 5: ler_d5}
        )
        assert threshold is not None
        assert 0.001 < threshold < 0.1

    def test_no_crossing(self):
        error_rates = np.array([0.001, 0.01, 0.1])
        ler_d3 = np.array([0.1, 0.2, 0.4])
        ler_d5 = np.array([0.2, 0.3, 0.45])

        threshold = estimate_threshold(
            error_rates, {3: ler_d3, 5: ler_d5}
        )
        assert threshold is None

    def test_single_distance(self):
        error_rates = np.array([0.01])
        assert estimate_threshold(error_rates, {3: np.array([0.1])}) is None


class TestAggregateSeedResults:
    def test_basic(self):
        results = [
            {"logical_error_rate": 0.10, "num_errors": 10, "num_shots": 100},
            {"logical_error_rate": 0.08, "num_errors": 8, "num_shots": 100},
            {"logical_error_rate": 0.12, "num_errors": 12, "num_shots": 100},
        ]
        agg = aggregate_seed_results(results)
        assert agg["num_seeds"] == 3
        assert agg["total_errors"] == 30
        assert agg["total_shots"] == 300
        assert agg["logical_error_rate_pooled"] == pytest.approx(0.1)
        assert agg["logical_error_rate_mean"] == pytest.approx(0.1)


# --- Evaluator ---


class TestDecoderEvaluator:
    def test_mwpm_only(self):
        """Run MWPM-only evaluation on a small grid."""
        evaluator = DecoderEvaluator(
            distances=[3],
            error_rates=[0.01],
            noise_configs=[NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01)],
            num_eval_shots=500,
            seed=42,
        )
        evaluator.add_decoder("mwpm", make_mwpm_decode_fn())
        df = evaluator.run(verbose=False)

        assert len(df) == 1
        assert df.iloc[0]["decoder"] == "mwpm"
        assert df.iloc[0]["distance"] == 3
        assert 0.0 <= df.iloc[0]["logical_error_rate"] <= 1.0
        assert df.iloc[0]["num_shots"] == 500

    def test_multiple_distances(self):
        evaluator = DecoderEvaluator(
            distances=[3, 5],
            error_rates=[0.01],
            noise_configs=[NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01)],
            num_eval_shots=200,
            seed=42,
        )
        evaluator.add_decoder("mwpm", make_mwpm_decode_fn())
        df = evaluator.run(verbose=False)

        assert len(df) == 2
        assert set(df["distance"]) == {3, 5}

    def test_multiple_error_rates(self):
        evaluator = DecoderEvaluator(
            distances=[3],
            error_rates=[0.001, 0.01, 0.1],
            noise_configs=[NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01)],
            num_eval_shots=200,
            seed=42,
        )
        evaluator.add_decoder("mwpm", make_mwpm_decode_fn())
        df = evaluator.run(verbose=False)

        assert len(df) == 3
        # higher error rate should generally have higher LER
        lers = df.sort_values("physical_error_rate")["logical_error_rate"].values
        assert lers[-1] > lers[0]

    def test_custom_decoder(self):
        """Register a trivial always-flip decoder."""

        def always_flip(syndromes, circuit):
            return np.ones(
                (len(syndromes), circuit.num_observables), dtype=bool
            )

        evaluator = DecoderEvaluator(
            distances=[3],
            error_rates=[0.01],
            noise_configs=[NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01)],
            num_eval_shots=200,
            seed=42,
        )
        evaluator.add_decoder("mwpm", make_mwpm_decode_fn())
        evaluator.add_decoder("always_flip", always_flip)
        df = evaluator.run(verbose=False)

        assert len(df) == 2
        mwpm_ler = df[df["decoder"] == "mwpm"]["logical_error_rate"].values[0]
        flip_ler = df[df["decoder"] == "always_flip"][
            "logical_error_rate"
        ].values[0]
        # MWPM should be much better than always-flip
        assert mwpm_ler < flip_ler

    def test_measurement_noise(self):
        """Evaluator handles measurement noise model."""
        evaluator = DecoderEvaluator(
            distances=[3],
            error_rates=[0.01],
            noise_configs=[NoiseConfig(NoiseModelType.MEASUREMENT, 0.01)],
            num_eval_shots=200,
            seed=42,
        )
        evaluator.add_decoder("mwpm", make_mwpm_decode_fn())
        df = evaluator.run(verbose=False)

        assert len(df) == 1
        assert df.iloc[0]["noise_model"] == "measurement"

    def test_unimplemented_noise_skipped(self):
        """Evaluator gracefully skips unimplemented noise models."""
        evaluator = DecoderEvaluator(
            distances=[3],
            error_rates=[0.01],
            noise_configs=[
                NoiseConfig(NoiseModelType.DEPOLARIZING, 0.01),
                NoiseConfig(NoiseModelType.BIASED_Z, 0.01),
            ],
            num_eval_shots=200,
            seed=42,
        )
        evaluator.add_decoder("mwpm", make_mwpm_decode_fn())
        df = evaluator.run(verbose=False)

        # biased-Z not yet implemented, should be skipped
        assert set(df["noise_model"]).issubset({"depolarizing", "biased_z"})
