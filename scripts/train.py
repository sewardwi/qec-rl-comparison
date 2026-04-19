"""
Main training entry point for surface code RL decoders.

Usage:
    python scripts/train.py --agent dqn --distance 3 --error-rate 0.01
    python scripts/train.py --agent ppo --distance 5 --timesteps 1000000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.callbacks import DecodingEvalCallback
from src.agents.dqn_agent import create_dqn_agent
from src.agents.ppo_agent import create_ppo_agent
from src.envs.surface_code_env import SurfaceCodeEnv
from src.quantum.decoder_baseline import MWPMDecoder
from src.quantum.noise_models import NoiseConfig, NoiseModelType, build_noisy_circuit
from src.quantum.surface_code import SurfaceCodeParams
from src.utils.reproducibility import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Train RL decoder for surface codes")
    parser.add_argument(
        "--agent", type=str, default="dqn", choices=["dqn", "ppo"],
        help="Agent type"
    )
    parser.add_argument("--distance", type=int, default=3, help="Code distance")
    parser.add_argument(
        "--error-rate", type=float, default=0.01, help="Physical error rate"
    )
    parser.add_argument(
        "--noise-model", type=str, default="depolarizing",
        help="Noise model type"
    )
    parser.add_argument(
        "--reward-type", type=str, default="combined",
        choices=["sparse", "potential_based", "heuristic_shaped", "combined"],
        help="Reward type"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps (default: auto based on distance)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval-freq", type=int, default=10_000, help="Evaluation frequency"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=500, help="Episodes per evaluation"
    )
    parser.add_argument(
        "--log-dir", type=str, default="results/logs", help="Log directory"
    )
    parser.add_argument(
        "--save-dir", type=str, default="results/models", help="Model save directory"
    )
    args = parser.parse_args()

    # auto-set timesteps based on distance
    if args.timesteps is None:
        timestep_map = {3: 500_000, 5: 1_000_000, 7: 2_000_000}
        args.timesteps = timestep_map.get(args.distance, 500_000)

    set_global_seed(args.seed)

    # create circuit
    params = SurfaceCodeParams(distance=args.distance, rounds=args.distance)
    noise_config = NoiseConfig(
        model_type=NoiseModelType(args.noise_model),
        physical_error_rate=args.error_rate,
    )
    circuit = build_noisy_circuit(params, noise_config)

    # create training and eval environments
    train_env = SurfaceCodeEnv(
        circuit=circuit, reward_type=args.reward_type
    )
    eval_env = SurfaceCodeEnv(
        circuit=circuit, reward_type="sparse"
    )

    # compute MWPM baseline for comparison
    print("Computing MWPM baseline...")
    decoder = MWPMDecoder.from_surface_code_circuit(circuit)
    syn, obs = circuit.sample(5000, seed=args.seed)
    mwpm_result = decoder.evaluate(syn, obs)
    mwpm_ler = mwpm_result["logical_error_rate"]
    print(f"MWPM LER: {mwpm_ler:.4f}")

    # create agent
    run_name = f"{args.agent}_d{args.distance}_p{args.error_rate}"
    log_dir = os.path.join(args.log_dir, run_name)
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "seed": args.seed,
        "log_dir": log_dir,
    }

    if args.agent == "dqn":
        agent = create_dqn_agent(train_env, config)
    else:
        agent = create_ppo_agent(train_env, config)

    # create eval callback
    eval_callback = DecodingEvalCallback(
        eval_env=eval_env,
        mwpm_ler=mwpm_ler,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_path=save_dir,
    )

    print(f"\nTraining {args.agent.upper()} on d={args.distance}, p={args.error_rate}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Eval every {args.eval_freq:,} steps ({args.eval_episodes} episodes)")
    print(f"Logging to: {log_dir}")
    print(f"Saving to: {save_dir}")
    print()

    agent.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # save final model
    final_path = os.path.join(save_dir, "final_model")
    agent.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    if eval_callback.eval_results:
        best = min(eval_callback.eval_results, key=lambda x: x["logical_error_rate"])
        print(f"\nBest LER: {best['logical_error_rate']:.4f} "
              f"(MWPM ratio: {best['mwpm_ratio']:.3f}) "
              f"at timestep {best['timestep']:,}")


if __name__ == "__main__":
    main()
