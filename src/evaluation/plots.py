"""Publication-quality figures for the paper.

Generates all 9 figures mapped to paper sections:
  1. Threshold curves (LER vs p_phys, per distance)
  2. Decoder comparison overlay (all decoders on same axes)
  3. Decoder ratio (RL/MWPM) vs physical error rate
  4. Training curves (reward & LER vs timestep)
  5. Noise model comparison
  6. Reward shaping ablation
  7. Syndrome pattern gallery
  8. Decoder behavior visualization
  9. Generalization matrix
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent style
STYLE_KWARGS = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

DECODER_COLORS = {
    "mwpm": "#d62728",
    "dqn": "#1f77b4",
    "ppo": "#ff7f0e",
}
DECODER_MARKERS = {"mwpm": "s", "dqn": "o", "ppo": "^"}
DISTANCE_COLORS = {3: "#1f77b4", 5: "#ff7f0e", 7: "#2ca02c"}
NOISE_COLORS = {
    "depolarizing": "#1f77b4",
    "biased_z": "#ff7f0e",
    "measurement": "#2ca02c",
    "correlated": "#d62728",
}


def _apply_style():
    plt.rcParams.update(STYLE_KWARGS)
    sns.set_palette("colorblind")


def plot_threshold_curves(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Figure 1: Logical vs physical error rate, one line per distance.

    Creates a panel for each decoder.
    """
    _apply_style()
    decoders = sorted(df["decoder"].unique())
    n_panels = len(decoders)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)

    for idx, dec in enumerate(decoders):
        ax = axes[0, idx]
        subset = df[df["decoder"] == dec]
        for d in sorted(subset["distance"].unique()):
            d_data = subset[subset["distance"] == d].sort_values("physical_error_rate")
            color = DISTANCE_COLORS.get(d, "gray")
            ax.errorbar(
                d_data["physical_error_rate"],
                d_data["logical_error_rate"],
                yerr=[
                    d_data["logical_error_rate"] - d_data["ci_lower"],
                    d_data["ci_upper"] - d_data["logical_error_rate"],
                ],
                fmt="o-",
                color=color,
                label=f"d={d}",
                capsize=3,
                markersize=5,
            )
        p_range = subset["physical_error_rate"]
        ax.plot([p_range.min(), p_range.max()],
                [p_range.min(), p_range.max()],
                "k--", alpha=0.2, label="LER=p")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate")
        ax.set_title(dec.upper())
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Threshold Curves", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_decoder_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Figure 2: All decoders overlaid per noise model."""
    _apply_style()
    noise_models = sorted(df["noise_model"].unique())
    n_panels = len(noise_models)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(5 * max(n_panels, 1), 4.5), squeeze=False)

    for idx, nm in enumerate(noise_models):
        ax = axes[0, idx]
        subset = df[(df["noise_model"] == nm) & (df["distance"] == df["distance"].max())]
        for dec in sorted(subset["decoder"].unique()):
            d_data = subset[subset["decoder"] == dec].sort_values("physical_error_rate")
            ax.plot(
                d_data["physical_error_rate"],
                d_data["logical_error_rate"],
                f"{DECODER_MARKERS.get(dec, 'x')}-",
                color=DECODER_COLORS.get(dec, "gray"),
                label=dec.upper(),
                markersize=6,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate")
        ax.set_title(f"{nm.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Decoder Comparison", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_decoder_ratio(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Figure 3: Decoder ratio (RL/MWPM) vs physical error rate."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    mwpm_data = df[df["decoder"] == "mwpm"].set_index(
        ["distance", "physical_error_rate", "noise_model"]
    )["logical_error_rate"]

    for dec in sorted(df["decoder"].unique()):
        if dec == "mwpm":
            continue
        subset = df[df["decoder"] == dec]
        ratios = []
        p_values = []
        for _, row in subset.iterrows():
            key = (row["distance"], row["physical_error_rate"], row["noise_model"])
            if key in mwpm_data.index:
                mwpm_ler = mwpm_data[key]
                if mwpm_ler > 0:
                    ratios.append(row["logical_error_rate"] / mwpm_ler)
                    p_values.append(row["physical_error_rate"])

        if p_values:
            ax.plot(
                p_values, ratios,
                f"{DECODER_MARKERS.get(dec, 'x')}-",
                color=DECODER_COLORS.get(dec, "gray"),
                label=dec.upper(),
                markersize=6,
            )

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="MWPM parity")
    ax.set_xscale("log")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("LER / MWPM LER")
    ax.set_title("Decoder Ratio vs MWPM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_training_curves(
    training_data: dict[str, list[dict]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Figure 4: Training curves — reward & LER vs timestep.

    Args:
        training_data: Dict mapping agent name -> list of eval result dicts
            with keys: timestep, logical_error_rate, mean_reward.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for agent_name, results in training_data.items():
        steps = [r["timestep"] for r in results]
        lers = [r["logical_error_rate"] for r in results]
        rewards = [r["mean_reward"] for r in results]
        color = DECODER_COLORS.get(agent_name.lower(), "gray")

        ax1.plot(steps, lers, "-", color=color, label=agent_name.upper(), linewidth=1.5)
        ax2.plot(steps, rewards, "-", color=color, label=agent_name.upper(), linewidth=1.5)

    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Logical Error Rate")
    ax1.set_title("LER During Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Mean Reward During Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_noise_comparison(
    df: pd.DataFrame,
    distance: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Figure 5: LER under each noise model at fixed distance."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    subset = df[(df["distance"] == distance) & (df["decoder"] == "mwpm")]
    for nm in sorted(subset["noise_model"].unique()):
        nm_data = subset[subset["noise_model"] == nm].sort_values("physical_error_rate")
        color = NOISE_COLORS.get(nm, "gray")
        ax.plot(
            nm_data["physical_error_rate"],
            nm_data["logical_error_rate"],
            "o-",
            color=color,
            label=nm.replace("_", " ").title(),
            markersize=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title(f"MWPM Performance Across Noise Models (d={distance})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_reward_ablation(
    ablation_data: dict[str, list[dict]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Figure 6: Reward shaping ablation — sparse vs potential vs heuristic.

    Args:
        ablation_data: Dict mapping reward_type -> list of eval result dicts.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    reward_colors = {
        "sparse": "#1f77b4",
        "potential_based": "#ff7f0e",
        "heuristic_shaped": "#2ca02c",
    }

    for reward_type, results in ablation_data.items():
        steps = [r["timestep"] for r in results]
        lers = [r["logical_error_rate"] for r in results]
        rewards = [r["mean_reward"] for r in results]
        color = reward_colors.get(reward_type, "gray")
        label = reward_type.replace("_", " ").title()

        ax1.plot(steps, lers, "-", color=color, label=label, linewidth=1.5)
        ax2.plot(steps, rewards, "-", color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Logical Error Rate")
    ax1.set_title("Reward Ablation: LER")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Reward Ablation: Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_syndrome_gallery(
    syndromes_by_noise: dict[str, np.ndarray],
    distance: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Figure 7: Syndrome pattern gallery under each noise model.

    Args:
        syndromes_by_noise: Dict mapping noise_model -> syndrome grid
            of shape (rounds, grid_h, grid_w).
    """
    _apply_style()
    n_models = len(syndromes_by_noise)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for idx, (nm, grid) in enumerate(syndromes_by_noise.items()):
        ax = axes[idx]
        # Show round 0
        round_data = grid[0]
        h, w = round_data.shape

        im = ax.imshow(
            round_data, cmap="RdBu_r", vmin=0, vmax=1,
            interpolation="nearest", aspect="equal",
        )
        ax.set_title(nm.replace("_", " ").title())
        ax.set_xlabel("Column")
        if idx == 0:
            ax.set_ylabel("Row")

        # Annotate triggered detectors
        triggered = np.argwhere(round_data > 0.5)
        for r, c in triggered:
            ax.plot(c, r, "rx", markersize=8, markeredgewidth=2)

    fig.suptitle(f"Syndrome Patterns (d={distance}, round 0)", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_decoder_behavior(
    syndrome_grid: np.ndarray,
    x_corrections: np.ndarray,
    z_corrections: np.ndarray,
    distance: int,
    success: bool,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Figure 8: Example decoder behavior — syndrome + corrections on lattice.

    Args:
        syndrome_grid: Shape (rounds, grid_h, grid_w).
        x_corrections: Bool array of shape (d*d,).
        z_corrections: Bool array of shape (d*d,).
        distance: Code distance.
        success: Whether decoding was successful.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: syndrome
    round_data = syndrome_grid[0]
    ax1.imshow(round_data, cmap="RdBu_r", vmin=0, vmax=1,
               interpolation="nearest", aspect="equal")
    triggered = np.argwhere(round_data > 0.5)
    for r, c in triggered:
        ax1.plot(c, r, "rx", markersize=10, markeredgewidth=2)
    ax1.set_title("Syndrome (Round 0)")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")

    # Right: corrections on d x d grid
    corr_grid = np.zeros((distance, distance, 3))  # RGB
    for i in range(distance):
        for j in range(distance):
            idx = i * distance + j
            if x_corrections[idx] and z_corrections[idx]:
                corr_grid[i, j] = [0.8, 0.0, 0.8]  # Purple for Y
            elif x_corrections[idx]:
                corr_grid[i, j] = [1.0, 0.0, 0.0]  # Red for X
            elif z_corrections[idx]:
                corr_grid[i, j] = [0.0, 0.0, 1.0]  # Blue for Z
            else:
                corr_grid[i, j] = [0.9, 0.9, 0.9]  # Light gray

    ax2.imshow(corr_grid, interpolation="nearest", aspect="equal")
    for i in range(distance):
        for j in range(distance):
            idx = i * distance + j
            label = ""
            if x_corrections[idx] and z_corrections[idx]:
                label = "Y"
            elif x_corrections[idx]:
                label = "X"
            elif z_corrections[idx]:
                label = "Z"
            if label:
                ax2.text(j, i, label, ha="center", va="center",
                         fontsize=12, fontweight="bold", color="white")

    result_str = "SUCCESS" if success else "FAILURE"
    result_color = "green" if success else "red"
    ax2.set_title(f"Corrections ({result_str})", color=result_color)
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")

    fig.suptitle(f"Decoder Behavior (d={distance})", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_generalization_matrix(
    matrix_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Figure 9: Generalization matrix — train on noise A, eval on noise B.

    Args:
        matrix_df: DataFrame with columns: train_noise, eval_noise,
            decoder, logical_error_rate.
    """
    _apply_style()
    decoders = sorted(matrix_df["decoder"].unique())
    n_dec = len(decoders)
    fig, axes = plt.subplots(1, max(n_dec, 1), figsize=(5 * max(n_dec, 1), 4), squeeze=False)

    for idx, dec in enumerate(decoders):
        ax = axes[0, idx]
        subset = matrix_df[matrix_df["decoder"] == dec]
        pivot = subset.pivot_table(
            values="logical_error_rate",
            index="train_noise",
            columns="eval_noise",
        )
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="YlOrRd",
            ax=ax, vmin=0, cbar_kws={"label": "LER"},
        )
        ax.set_title(dec.upper())
        ax.set_xlabel("Eval Noise")
        ax.set_ylabel("Train Noise")

    fig.suptitle("Generalization Matrix", y=1.05)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_scaling_table(
    scaling_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Supplementary: Circuit complexity scaling table."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * len(scaling_df)))
    ax.axis("off")

    cols = ["distance", "data_qubits", "total_qubits", "detectors",
            "action_space", "dem_errors", "total_gates"]
    headers = ["d", "Data\nQubits", "Total\nQubits", "Detectors",
               "|A|", "DEM\nErrors", "Total\nGates"]

    table_data = scaling_df[cols].values.tolist()
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title("Circuit Complexity Scaling", pad=20)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
