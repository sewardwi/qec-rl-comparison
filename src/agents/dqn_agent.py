"""DQN agent creation via Stable-Baselines3."""

from __future__ import annotations

from stable_baselines3 import DQN

from src.agents.networks import SyndromeCNN


def create_dqn_agent(env, config: dict) -> DQN:
    """
    Create a DQN agent configured for surface code decoding.

    env: Gymnasium environment
    config: Dict with optional keys:
        - seed (int): Random seed, default 42
        - log_dir (str): Tensorboard log directory
        - features_dim (int): CNN output features, default 128
        - overrides (dict): Override any DQN constructor kwarg

    Configured DQN agent.
    """
    features_dim = config.get("features_dim", 128)

    policy_kwargs = dict(
        features_extractor_class=SyndromeCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=[256, 256],
    )

    kwargs = dict(
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        train_freq=4,
        verbose=1,
        tensorboard_log=config.get("log_dir"),
        seed=config.get("seed", 42),
    )
    kwargs.update(config.get("overrides", {}))

    return DQN("CnnPolicy", env, **kwargs)
