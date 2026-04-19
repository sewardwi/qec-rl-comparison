"""
Evaluation metrics for decoder comparison.

Provides logical error rate computation with Clopper-Pearson confidence
intervals, threshold estimation, and decoder ratio calculations.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist


def clopper_pearson(
    k: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """
    Clopper-Pearson exact binomial confidence interval.

    k: Number of errors.
    n: Number of shots.
    alpha: Significance level (default 0.05 for 95% CI).

    (lower, upper) confidence interval bounds.
    """
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


def logical_error_rate(
    n_errors: int, n_shots: int, alpha: float = 0.05
) -> dict:
    """
    Compute logical error rate with confidence interval.

    n_errors: Number of logical errors.
    n_shots: Total number of shots.
    alpha: Significance level for CI.

    Dict with rate, ci_lower, ci_upper, n_errors, n_shots.
    """
    rate = n_errors / n_shots if n_shots > 0 else 0.0
    ci_lo, ci_hi = clopper_pearson(n_errors, n_shots, alpha)
    return {
        "logical_error_rate": rate,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "num_errors": n_errors,
        "num_shots": n_shots,
    }


def decoder_ratio(rl_ler: float, mwpm_ler: float) -> float:
    """
    Compute RL-to-MWPM error rate ratio. <1 means RL wins.

    rl_ler: RL decoder logical error rate.
    mwpm_ler: MWPM decoder logical error rate.

    Ratio rl_ler / mwpm_ler, or inf if mwpm_ler is 0.
    """
    if mwpm_ler <= 0:
        return float("inf")
    return rl_ler / mwpm_ler


def estimate_threshold(
    error_rates: np.ndarray,
    logical_rates_by_distance: dict[int, np.ndarray],
) -> float | None:
    """
    Estimate the error correction threshold from crossing curves.

    Finds the physical error rate where log-log curves for different
    distances intersect. Uses linear interpolation between grid points.

    error_rates: Array of physical error rates (sorted ascending).
    logical_rates_by_distance: Dict mapping distance -> array of
        logical error rates (same length as error_rates).

    Estimated threshold error rate, or None if no crossing found.
    """
    distances = sorted(logical_rates_by_distance.keys())
    if len(distances) < 2:
        return None

    # use the two largest distances for threshold estimation
    d_small = distances[-2]
    d_large = distances[-1]
    ler_small = np.array(logical_rates_by_distance[d_small])
    ler_large = np.array(logical_rates_by_distance[d_large])

    # work in log space for better interpolation
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log10(error_rates)
        log_ler_small = np.log10(ler_small)
        log_ler_large = np.log10(ler_large)

    # find crossing: where larger distance transitions from better to worse
    diff = log_ler_large - log_ler_small
    valid = np.isfinite(diff)
    diff = diff[valid]
    log_p_valid = log_p[valid]

    if len(diff) < 2:
        return None

    # look for sign change in diff
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            # linear interpolation to find crossing
            t = diff[i] / (diff[i] - diff[i + 1])
            log_threshold = log_p_valid[i] + t * (
                log_p_valid[i + 1] - log_p_valid[i]
            )
            return float(10**log_threshold)

    return None


def aggregate_seed_results(
    results_by_seed: list[dict],
) -> dict:
    """
    Aggregate evaluation results across multiple random seeds.

    results_by_seed: List of result dicts, each with
        'logical_error_rate', 'num_errors', 'num_shots'.

    Dict with mean, std, and pooled CI of logical error rate.
    """
    lers = [r["logical_error_rate"] for r in results_by_seed]
    total_errors = sum(r["num_errors"] for r in results_by_seed)
    total_shots = sum(r["num_shots"] for r in results_by_seed)

    pooled_ler = total_errors / total_shots if total_shots > 0 else 0.0
    ci_lo, ci_hi = clopper_pearson(total_errors, total_shots)

    return {
        "logical_error_rate_mean": float(np.mean(lers)),
        "logical_error_rate_std": float(np.std(lers)),
        "logical_error_rate_pooled": pooled_ler,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "total_errors": total_errors,
        "total_shots": total_shots,
        "num_seeds": len(results_by_seed),
    }
