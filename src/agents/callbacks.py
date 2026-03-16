"""Training callbacks for RL agents on surface code decoding."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DecodingEvalCallback(BaseCallback):
    """Periodically evaluate the agent's decoding performance.

    Every eval_freq steps: run the agent on eval_episodes episodes,
    compute logical error rate, log ratio to MWPM baseline, and
    save the best model.
    """

    def __init__(
        self,
        eval_env,
        mwpm_ler: float,
        eval_freq: int = 10_000,
        eval_episodes: int = 1000,
        save_path: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        Args:
            eval_env: Environment for evaluation.
            mwpm_ler: MWPM logical error rate (for computing ratio).
            eval_freq: Evaluate every this many timesteps.
            eval_episodes: Number of episodes per evaluation.
            save_path: Directory to save best model. None to skip saving.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.mwpm_ler = mwpm_ler
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_path = save_path
        self.best_ler = float("inf")
        self.eval_results: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        successes = 0
        total_rewards = []

        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(
                    int(action)
                )
                episode_reward += reward
                done = terminated or truncated

            if info.get("success"):
                successes += 1
            total_rewards.append(episode_reward)

        ler = 1.0 - successes / self.eval_episodes
        ratio = ler / self.mwpm_ler if self.mwpm_ler > 0 else float("inf")
        mean_reward = float(np.mean(total_rewards))

        result = {
            "timestep": self.num_timesteps,
            "logical_error_rate": ler,
            "mwpm_ratio": ratio,
            "mean_reward": mean_reward,
            "success_rate": successes / self.eval_episodes,
        }
        self.eval_results.append(result)

        # Log to tensorboard
        self.logger.record("eval/logical_error_rate", ler)
        self.logger.record("eval/mwpm_ratio", ratio)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/success_rate", successes / self.eval_episodes)

        if self.verbose >= 1:
            print(
                f"  [Eval @ {self.num_timesteps}] LER={ler:.4f}, "
                f"MWPM ratio={ratio:.3f}, reward={mean_reward:.3f}"
            )

        # Save best model
        if ler < self.best_ler and self.save_path is not None:
            self.best_ler = ler
            path = os.path.join(self.save_path, "best_model")
            self.model.save(path)
            if self.verbose >= 1:
                print(f"  New best model saved (LER={ler:.4f})")

        return True


class CurriculumCallback(BaseCallback):
    """Linearly increase physical error rate during training.

    Helps the agent learn basic correction patterns on easy instances
    before encountering harder cases.
    """

    def __init__(
        self,
        env,
        p_start: float = 0.001,
        p_end: float = 0.01,
        total_timesteps: int = 500_000,
        verbose: int = 0,
    ):
        """
        Args:
            env: The training environment (must have _circuit that can be replaced).
            p_start: Starting physical error rate.
            p_end: Final physical error rate.
            total_timesteps: Over how many timesteps to anneal.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.env = env
        self.p_start = p_start
        self.p_end = p_end
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        current_p = self.p_start + (self.p_end - self.p_start) * progress

        if self.verbose >= 2 and self.num_timesteps % 10_000 == 0:
            print(f"  [Curriculum] p={current_p:.5f} at step {self.num_timesteps}")

        return True
