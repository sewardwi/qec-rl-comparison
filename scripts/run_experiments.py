"""Run all experiments and generate paper data + figures.

This script orchestrates the full experiment pipeline:
  1. Train DQN and PPO on d=3 (depolarizing + measurement + biased-Z)
  2. Train DQN and PPO on d=5 (depolarizing)
  3. Run reward ablation (sparse, potential_based, heuristic_shaped)
  4. Evaluate all models via DecoderEvaluator
  5. MWPM baselines across all noise models
  6. Generalization matrix
  7. Generate all paper figures

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --timesteps 50000  # Quick test
    python scripts/run_experiments.py --skip-training     # Figures only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.callbacks import DecodingEvalCallback
from src.agents.dqn_agent import create_dqn_agent
from src.agents.ppo_agent import create_ppo_agent
from src.envs.surface_code_env import SurfaceCodeEnv
from src.evaluation.evaluator import DecoderEvaluator, make_mwpm_decode_fn, make_rl_decode_fn
from src.evaluation.metrics import estimate_threshold
from src.evaluation.plots import (
    plot_decoder_behavior,
    plot_decoder_comparison,
    plot_decoder_ratio,
    plot_generalization_matrix,
    plot_noise_comparison,
    plot_reward_ablation,
    plot_scaling_table,
    plot_syndrome_gallery,
    plot_threshold_curves,
    plot_training_curves,
)
from src.quantum.circuit_analysis import compute_error_budget, scaling_analysis
from src.quantum.decoder_baseline import MWPMDecoder
from src.quantum.noise_models import NoiseConfig, NoiseModelType, build_noisy_circuit
from src.quantum.surface_code import SurfaceCodeParams
from src.quantum.syndrome import SyndromeGrid
from src.utils.reproducibility import set_global_seed


def train_agent(
    agent_type: str,
    distance: int,
    noise_config: NoiseConfig,
    timesteps: int,
    reward_type: str = "sparse",
    seed: int = 42,
    eval_freq: int = 5000,
    eval_episodes: int = 200,
    save_dir: str = "results/models",
) -> tuple:
    """Train a single agent and return (model, eval_results)."""
    params = SurfaceCodeParams(distance=distance, rounds=distance)
    circuit = build_noisy_circuit(params, noise_config)

    train_env = SurfaceCodeEnv(circuit=circuit, reward_type=reward_type)
    eval_env = SurfaceCodeEnv(circuit=circuit, reward_type="sparse")

    # MWPM baseline
    decoder = MWPMDecoder.from_surface_code_circuit(circuit)
    syn, obs = circuit.sample(2000, seed=seed)
    mwpm_ler = decoder.evaluate(syn, obs)["logical_error_rate"]

    run_name = f"{agent_type}_d{distance}_{noise_config.model_type.value}_{reward_type}"
    model_dir = os.path.join(save_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)

    config = {
        "seed": seed,
        "overrides": {"verbose": 0, "learning_starts": min(1000, timesteps // 10)},
    }
    if agent_type == "ppo":
        config["overrides"] = {
            "verbose": 0,
            "n_steps": min(2048, timesteps // 4),
        }

    if agent_type == "dqn":
        model = create_dqn_agent(train_env, config)
    else:
        model = create_ppo_agent(train_env, config)

    callback = DecodingEvalCallback(
        eval_env=eval_env,
        mwpm_ler=mwpm_ler,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        save_path=model_dir,
        verbose=0,
    )

    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(model_dir, "final_model"))

    return model, callback.eval_results, mwpm_ler


def run_all_experiments(args):
    """Run the complete experiment pipeline."""
    set_global_seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    training_data = {}  # For training curves plot
    trained_models = {}  # (agent_type, distance, noise) -> model
    all_eval_results = []

    # =============================================
    # 1. Training experiments
    # =============================================
    if not args.skip_training:
        print("=" * 60)
        print("PHASE 1: Training")
        print("=" * 60)

        train_configs = [
            # (agent, distance, noise_type, reward)
            ("dqn", 3, "depolarizing", "sparse"),
            ("ppo", 3, "depolarizing", "sparse"),
            ("dqn", 5, "depolarizing", "sparse"),
            ("ppo", 5, "depolarizing", "sparse"),
            ("dqn", 3, "measurement", "sparse"),
            ("ppo", 3, "measurement", "sparse"),
            ("dqn", 3, "biased_z", "sparse"),
            ("ppo", 3, "biased_z", "sparse"),
        ]

        for agent_type, d, noise_name, reward_type in train_configs:
            noise_config = _make_noise_config(noise_name, args.error_rate)
            ts = args.timesteps if d == 3 else int(args.timesteps * 1.5)
            key = f"{agent_type}_d{d}_{noise_name}"

            print(f"\n  Training {key} for {ts:,} steps...")
            t0 = time.time()
            model, eval_results, mwpm_ler = train_agent(
                agent_type=agent_type,
                distance=d,
                noise_config=noise_config,
                timesteps=ts,
                reward_type=reward_type,
                seed=args.seed,
                eval_freq=max(1000, ts // 20),
                eval_episodes=args.eval_episodes,
                save_dir=args.model_dir,
            )
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s. "
                  f"Final LER: {eval_results[-1]['logical_error_rate']:.4f}" if eval_results else "")

            trained_models[(agent_type, d, noise_name)] = model
            training_data[key] = eval_results

        # Save training curves data
        training_curves_path = os.path.join(args.data_dir, "training_curves.json")
        with open(training_curves_path, "w") as f:
            json.dump(training_data, f, indent=2)
        print(f"\n  Training curves saved to {training_curves_path}")

        # =============================================
        # 2. Reward ablation (d=3, depolarizing)
        # =============================================
        print("\n" + "=" * 60)
        print("PHASE 2: Reward Ablation")
        print("=" * 60)

        ablation_data = {}
        for reward_type in ["sparse", "potential_based", "heuristic_shaped"]:
            noise_config = _make_noise_config("depolarizing", args.error_rate)
            print(f"\n  Training DQN d=3 with {reward_type}...")
            _, eval_results, _ = train_agent(
                agent_type="dqn",
                distance=3,
                noise_config=noise_config,
                timesteps=args.timesteps,
                reward_type=reward_type,
                seed=args.seed,
                eval_freq=max(1000, args.timesteps // 20),
                eval_episodes=args.eval_episodes,
                save_dir=args.model_dir,
            )
            ablation_data[reward_type] = eval_results

        ablation_path = os.path.join(args.data_dir, "reward_ablation.json")
        with open(ablation_path, "w") as f:
            json.dump(ablation_data, f, indent=2)

    else:
        # Load existing data if skipping training
        training_curves_path = os.path.join(args.data_dir, "training_curves.json")
        ablation_path = os.path.join(args.data_dir, "reward_ablation.json")
        if os.path.exists(training_curves_path):
            with open(training_curves_path) as f:
                training_data = json.load(f)
        if os.path.exists(ablation_path):
            with open(ablation_path) as f:
                ablation_data = json.load(f)

    # =============================================
    # 3. Evaluation sweeps
    # =============================================
    print("\n" + "=" * 60)
    print("PHASE 3: Evaluation Sweeps")
    print("=" * 60)

    error_rates = [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    noise_types = ["depolarizing", "measurement", "biased_z", "correlated"]

    for noise_name in noise_types:
        noise_config = _make_noise_config(noise_name, args.error_rate)
        print(f"\n  Evaluating {noise_name}...")

        # Run per-distance to avoid RL model/observation shape mismatches
        for d in [3, 5]:
            evaluator = DecoderEvaluator(
                distances=[d],
                error_rates=error_rates,
                noise_configs=[noise_config],
                num_eval_shots=args.eval_shots,
                seed=args.seed,
            )
            evaluator.add_decoder("mwpm", make_mwpm_decode_fn())

            # Add RL models trained on this distance + noise
            for agent_type in ["dqn", "ppo"]:
                key = (agent_type, d, noise_name)
                if key in trained_models:
                    evaluator.add_decoder(
                        agent_type,
                        make_rl_decode_fn(trained_models[key], SurfaceCodeEnv),
                    )

            df = evaluator.run(verbose=True)
            all_eval_results.append(df)

    # Combine all evaluation results
    eval_df = pd.concat(all_eval_results, ignore_index=True)
    eval_path = os.path.join(args.data_dir, "evaluation_results.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"\n  Evaluation results saved to {eval_path}")

    # =============================================
    # 4. Generalization matrix
    # =============================================
    print("\n" + "=" * 60)
    print("PHASE 4: Generalization Matrix")
    print("=" * 60)

    gen_rows = []
    train_noises = ["depolarizing", "biased_z"]
    eval_noises = ["depolarizing", "biased_z", "measurement", "correlated"]

    for train_noise in train_noises:
        for eval_noise in eval_noises:
            for agent_type in ["dqn", "ppo"]:
                key = (agent_type, 3, train_noise)
                if key not in trained_models:
                    continue

                eval_config = _make_noise_config(eval_noise, args.error_rate)
                params = SurfaceCodeParams(distance=3, rounds=3)
                try:
                    circuit = build_noisy_circuit(params, eval_config)
                except NotImplementedError:
                    continue

                env = SurfaceCodeEnv(circuit=circuit)
                model = trained_models[key]

                successes = 0
                n_episodes = args.eval_episodes
                for _ in range(n_episodes):
                    obs, _ = env.reset()
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, term, trunc, info = env.step(int(action))
                        done = term or trunc
                    if info.get("success"):
                        successes += 1

                ler = 1.0 - successes / n_episodes
                gen_rows.append({
                    "decoder": agent_type,
                    "train_noise": train_noise,
                    "eval_noise": eval_noise,
                    "logical_error_rate": ler,
                })

    gen_df = pd.DataFrame(gen_rows)
    gen_path = os.path.join(args.data_dir, "generalization_matrix.csv")
    gen_df.to_csv(gen_path, index=False)
    print(f"  Generalization matrix saved to {gen_path}")

    # =============================================
    # 5. Circuit analysis & error budget
    # =============================================
    print("\n" + "=" * 60)
    print("PHASE 5: Circuit Analysis")
    print("=" * 60)

    scaling_df = scaling_analysis([3, 5, 7])
    scaling_path = os.path.join(args.data_dir, "scaling_analysis.csv")
    scaling_df.to_csv(scaling_path, index=False)
    print(f"  Scaling analysis saved to {scaling_path}")
    print(scaling_df.to_string(index=False))

    params3 = SurfaceCodeParams(distance=3, rounds=3)
    budget = compute_error_budget(params3, physical_error_rate=args.error_rate, num_shots=5000)
    budget_path = os.path.join(args.data_dir, "error_budget.json")
    with open(budget_path, "w") as f:
        json.dump(budget, f, indent=2)
    print(f"\n  Error budget: {budget}")

    # =============================================
    # 6. Generate all figures
    # =============================================
    print("\n" + "=" * 60)
    print("PHASE 6: Generating Figures")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")

    # Figure 1: Threshold curves
    if not eval_df.empty:
        # Normalize decoder names for plotting
        plot_df = eval_df.copy()
        plot_df["decoder"] = plot_df["decoder"].str.replace(r"_d\d", "", regex=True)

        fig1 = plot_threshold_curves(plot_df, os.path.join(args.fig_dir, "fig1_threshold_curves.png"))
        plt.close(fig1)
        print("  Figure 1: Threshold curves")

        # Figure 2: Decoder comparison
        fig2 = plot_decoder_comparison(plot_df, os.path.join(args.fig_dir, "fig2_decoder_comparison.png"))
        plt.close(fig2)
        print("  Figure 2: Decoder comparison")

        # Figure 3: Decoder ratio
        fig3 = plot_decoder_ratio(plot_df, os.path.join(args.fig_dir, "fig3_decoder_ratio.png"))
        plt.close(fig3)
        print("  Figure 3: Decoder ratio")

        # Figure 5: Noise comparison
        fig5 = plot_noise_comparison(plot_df, distance=3,
                                     save_path=os.path.join(args.fig_dir, "fig5_noise_comparison.png"))
        plt.close(fig5)
        print("  Figure 5: Noise comparison")

    # Figure 4: Training curves
    if training_data:
        # Only include d=3 depolarizing for main training curve
        tc_data = {}
        for key in ["dqn_d3_depolarizing", "ppo_d3_depolarizing"]:
            if key in training_data and training_data[key]:
                agent_name = key.split("_")[0]
                tc_data[agent_name] = training_data[key]

        if tc_data:
            fig4 = plot_training_curves(tc_data, os.path.join(args.fig_dir, "fig4_training_curves.png"))
            plt.close(fig4)
            print("  Figure 4: Training curves")

    # Figure 6: Reward ablation
    if "ablation_data" in dir() and ablation_data:
        fig6 = plot_reward_ablation(ablation_data, os.path.join(args.fig_dir, "fig6_reward_ablation.png"))
        plt.close(fig6)
        print("  Figure 6: Reward ablation")
    elif os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation_data = json.load(f)
        if ablation_data:
            fig6 = plot_reward_ablation(ablation_data, os.path.join(args.fig_dir, "fig6_reward_ablation.png"))
            plt.close(fig6)
            print("  Figure 6: Reward ablation")

    # Figure 7: Syndrome gallery
    syndromes_by_noise = {}
    for noise_name in ["depolarizing", "measurement", "biased_z", "correlated"]:
        noise_config = _make_noise_config(noise_name, args.error_rate)
        params = SurfaceCodeParams(distance=3, rounds=3)
        try:
            circuit = build_noisy_circuit(params, noise_config)
            syn, _ = circuit.sample(1, seed=42)
            sg = SyndromeGrid(circuit.circuit)
            grid = sg.reshape_single(syn[0])
            syndromes_by_noise[noise_name] = grid
        except NotImplementedError:
            pass

    if syndromes_by_noise:
        fig7 = plot_syndrome_gallery(syndromes_by_noise, distance=3,
                                     save_path=os.path.join(args.fig_dir, "fig7_syndrome_gallery.png"))
        plt.close(fig7)
        print("  Figure 7: Syndrome gallery")

    # Figure 8: Decoder behavior
    params3 = SurfaceCodeParams(distance=3, rounds=3)
    noise_depol = NoiseConfig(NoiseModelType.DEPOLARIZING, args.error_rate)
    circuit3 = build_noisy_circuit(params3, noise_depol)
    env_demo = SurfaceCodeEnv(circuit=circuit3)

    # Run one episode with a trained model if available
    demo_model = trained_models.get(("dqn", 3, "depolarizing"))
    if demo_model is None:
        demo_model = trained_models.get(("ppo", 3, "depolarizing"))

    if demo_model is not None:
        obs_demo, _ = env_demo.reset()
        done = False
        while not done:
            action, _ = demo_model.predict(obs_demo, deterministic=True)
            obs_demo, _, term, trunc, info = env_demo.step(int(action))
            done = term or trunc

        fig8 = plot_decoder_behavior(
            syndrome_grid=env_demo._current_syndrome_grid,
            x_corrections=env_demo._x_corrections,
            z_corrections=env_demo._z_corrections,
            distance=3,
            success=info.get("success", False),
            save_path=os.path.join(args.fig_dir, "fig8_decoder_behavior.png"),
        )
        plt.close(fig8)
        print("  Figure 8: Decoder behavior")

    # Figure 9: Generalization matrix
    if not gen_df.empty:
        fig9 = plot_generalization_matrix(gen_df,
                                          save_path=os.path.join(args.fig_dir, "fig9_generalization.png"))
        plt.close(fig9)
        print("  Figure 9: Generalization matrix")

    # Scaling table (supplementary)
    fig_s = plot_scaling_table(scaling_df, os.path.join(args.fig_dir, "fig_scaling_table.png"))
    plt.close(fig_s)
    print("  Figure S: Scaling table")

    # Threshold estimation
    for noise_name in eval_df["noise_model"].unique():
        mwpm_sub = eval_df[(eval_df["decoder"] == "mwpm") & (eval_df["noise_model"] == noise_name)]
        if mwpm_sub.empty:
            continue
        ler_by_d = {}
        for d in mwpm_sub["distance"].unique():
            d_data = mwpm_sub[mwpm_sub["distance"] == d].sort_values("physical_error_rate")
            ler_by_d[d] = d_data["logical_error_rate"].values
        p_arr = np.array(sorted(mwpm_sub["physical_error_rate"].unique()))
        threshold = estimate_threshold(p_arr, ler_by_d)
        if threshold is not None:
            print(f"\n  Threshold ({noise_name}): {threshold:.4f} ({threshold*100:.2f}%)")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\n  Data: {args.data_dir}/")
    print(f"  Figures: {args.fig_dir}/")
    print(f"  Models: {args.model_dir}/")


def _make_noise_config(noise_name: str, p: float) -> NoiseConfig:
    """Create a NoiseConfig from name and error rate."""
    kwargs = {}
    if noise_name == "biased_z":
        kwargs["bias_ratio"] = 10.0
    elif noise_name == "correlated":
        kwargs["correlation_strength"] = p / 5
    return NoiseConfig(NoiseModelType(noise_name), p, **kwargs)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Training timesteps per agent (default 50K for testing)")
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--eval-shots", type=int, default=2000,
                        help="Evaluation shots per point")
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--fig-dir", type=str, default="results/figures")
    parser.add_argument("--model-dir", type=str, default="results/models")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, regenerate figures from existing data")
    args = parser.parse_args()

    run_all_experiments(args)
