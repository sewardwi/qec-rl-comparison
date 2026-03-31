"""Full evaluation grid: run all decoders on identical syndrome sets.

Usage:
    python scripts/evaluate.py --model-dir results/models/dqn_d3_p0.01
    python scripts/evaluate.py --distances 3 5 --error-rates 0.001 0.01 0.1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import (
    DecoderEvaluator,
    make_mwpm_decode_fn,
    make_rl_decode_fn,
)
from src.quantum.noise_models import NoiseConfig, NoiseModelType


def main():
    parser = argparse.ArgumentParser(description="Evaluate decoders")
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
        "--noise-models",
        type=str,
        nargs="+",
        default=["depolarizing"],
        help="Noise model types",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=10_000,
        help="Evaluation shots per point",
    )
    parser.add_argument(
        "--dqn-model",
        type=str,
        default=None,
        help="Path to trained DQN model (.zip)",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default=None,
        help="Path to trained PPO model (.zip)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="results/data/evaluation_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    if args.error_rates is None:
        args.error_rates = np.logspace(-3, -0.7, 15).tolist()

    # Build noise configs
    noise_configs = []
    for nm in args.noise_models:
        noise_configs.append(NoiseConfig(NoiseModelType(nm), 0.01))

    # Create evaluator
    evaluator = DecoderEvaluator(
        distances=args.distances,
        error_rates=args.error_rates,
        noise_configs=noise_configs,
        num_eval_shots=args.num_shots,
        seed=args.seed,
    )

    # Always add MWPM
    evaluator.add_decoder("mwpm", make_mwpm_decode_fn())

    # Add RL models if provided
    if args.dqn_model:
        from stable_baselines3 import DQN

        from src.envs.surface_code_env import SurfaceCodeEnv

        model = DQN.load(args.dqn_model)
        evaluator.add_decoder(
            "dqn", make_rl_decode_fn(model, SurfaceCodeEnv)
        )

    if args.ppo_model:
        from stable_baselines3 import PPO

        from src.envs.surface_code_env import SurfaceCodeEnv

        model = PPO.load(args.ppo_model)
        evaluator.add_decoder(
            "ppo", make_rl_decode_fn(model, SurfaceCodeEnv)
        )

    print(f"Distances: {args.distances}")
    print(f"Error rates: {len(args.error_rates)} points")
    print(f"Noise models: {args.noise_models}")
    print(f"Decoders: {list(evaluator._decoders.keys())}")
    print(f"Shots per point: {args.num_shots:,}")
    print()

    # Run evaluation
    df = evaluator.run(verbose=True)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(f"Total rows: {len(df)}")

    # Print summary
    print("\nSummary:")
    for decoder in df["decoder"].unique():
        subset = df[df["decoder"] == decoder]
        mean_ler = subset["logical_error_rate"].mean()
        print(f"  {decoder}: mean LER = {mean_ler:.4f}")


if __name__ == "__main__":
    main()
