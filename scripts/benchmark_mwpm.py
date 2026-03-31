"""Generate MWPM baseline curves via Sinter for all noise models.

Usage:
    python scripts/benchmark_mwpm.py
    python scripts/benchmark_mwpm.py --distances 3 5 --workers 8
    python scripts/benchmark_mwpm.py --output results/data/mwpm_baselines.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantum.decoder_baseline import run_sinter_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Generate MWPM baselines via Sinter"
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=[3, 5],
        help="Code distances",
    )
    parser.add_argument(
        "--error-rates",
        type=float,
        nargs="+",
        default=None,
        help="Physical error rates (default: logspace)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=100_000,
        help="Max shots per point",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=1000,
        help="Stop after this many errors per point",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/data/mwpm_baselines.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    if args.error_rates is None:
        args.error_rates = np.logspace(-3, -0.7, 15).tolist()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Distances: {args.distances}")
    print(f"Error rates: {len(args.error_rates)} points")
    print(f"Workers: {args.workers}")
    print(f"Max shots: {args.max_shots:,}, Max errors: {args.max_errors}")
    print()

    # Depolarizing noise (standard benchmark)
    print("Running depolarizing benchmark...")
    df_depol = run_sinter_benchmark(
        distances=args.distances,
        error_rates=args.error_rates,
        num_workers=args.workers,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
    )
    df_depol["noise_model"] = "depolarizing"

    # Measurement-dominated noise
    print("\nRunning measurement-dominated benchmark...")

    def measurement_noise(p):
        return {
            "after_clifford_depolarization": p / 10,
            "before_round_data_depolarization": p / 10,
            "before_measure_flip_probability": p,
            "after_reset_flip_probability": p / 10,
        }

    df_meas = run_sinter_benchmark(
        distances=args.distances,
        error_rates=args.error_rates,
        noise_params_fn=measurement_noise,
        num_workers=args.workers,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
    )
    df_meas["noise_model"] = "measurement"

    # Combine
    df = pd.concat([df_depol, df_meas], ignore_index=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    main()
