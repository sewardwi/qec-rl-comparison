"""
Regenerate all paper figures from saved experiment data.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --data-dir results/data --fig-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.plots import (
    plot_decoder_comparison,
    plot_decoder_ratio,
    plot_generalization_matrix,
    plot_noise_comparison,
    plot_reward_ablation,
    plot_scaling_table,
    plot_threshold_curves,
    plot_training_curves,
)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--fig-dir", type=str, default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    # load data
    eval_path = os.path.join(args.data_dir, "evaluation_results.csv")
    if os.path.exists(eval_path):
        eval_df = pd.read_csv(eval_path)
        plot_df = eval_df.copy()
        plot_df["decoder"] = plot_df["decoder"].str.replace(r"_d\d", "", regex=True)

        fig1 = plot_threshold_curves(plot_df, os.path.join(args.fig_dir, "fig1_threshold_curves.png"))
        plt.close(fig1)
        print("Generated: fig1_threshold_curves.png")

        fig2 = plot_decoder_comparison(plot_df, os.path.join(args.fig_dir, "fig2_decoder_comparison.png"))
        plt.close(fig2)
        print("Generated: fig2_decoder_comparison.png")

        fig3 = plot_decoder_ratio(plot_df, os.path.join(args.fig_dir, "fig3_decoder_ratio.png"))
        plt.close(fig3)
        print("Generated: fig3_decoder_ratio.png")

        fig5 = plot_noise_comparison(plot_df, distance=3,
                                     save_path=os.path.join(args.fig_dir, "fig5_noise_comparison.png"))
        plt.close(fig5)
        print("Generated: fig5_noise_comparison.png")

    # training curves
    tc_path = os.path.join(args.data_dir, "training_curves.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            training_data = json.load(f)
        tc_data = {}
        for key in ["dqn_d3_depolarizing", "ppo_d3_depolarizing"]:
            if key in training_data and training_data[key]:
                tc_data[key.split("_")[0]] = training_data[key]
        if tc_data:
            fig4 = plot_training_curves(tc_data, os.path.join(args.fig_dir, "fig4_training_curves.png"))
            plt.close(fig4)
            print("Generated: fig4_training_curves.png")

    # reward ablation
    abl_path = os.path.join(args.data_dir, "reward_ablation.json")
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            ablation_data = json.load(f)
        if ablation_data:
            fig6 = plot_reward_ablation(ablation_data, os.path.join(args.fig_dir, "fig6_reward_ablation.png"))
            plt.close(fig6)
            print("Generated: fig6_reward_ablation.png")

    # generalization matrix
    gen_path = os.path.join(args.data_dir, "generalization_matrix.csv")
    if os.path.exists(gen_path):
        gen_df = pd.read_csv(gen_path)
        if not gen_df.empty:
            fig9 = plot_generalization_matrix(gen_df,
                                              save_path=os.path.join(args.fig_dir, "fig9_generalization.png"))
            plt.close(fig9)
            print("Generated: fig9_generalization.png")

    # scaling table
    sc_path = os.path.join(args.data_dir, "scaling_analysis.csv")
    if os.path.exists(sc_path):
        scaling_df = pd.read_csv(sc_path)
        fig_s = plot_scaling_table(scaling_df, os.path.join(args.fig_dir, "fig_scaling_table.png"))
        plt.close(fig_s)
        print("Generated: fig_scaling_table.png")

    print("\nAll figures regenerated.")


if __name__ == "__main__":
    main()
