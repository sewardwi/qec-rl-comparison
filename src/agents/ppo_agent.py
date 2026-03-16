"""PPO agent creation via Stable-Baselines3."""

from __future__ import annotations

from stable_baselines3 import PPO

from src.agents.networks import SyndromeCNN


def create_ppo_agent(env, config: dict) -> PPO:
    """Create a PPO agent configured for surface code decoding.

    Args:
        env: Gymnasium environment.
        config: Dict with optional keys:
            - seed (int): Random seed. Default 42.
            - log_dir (str): Tensorboard log directory.
            - features_dim (int): CNN output features. Default 128.
            - overrides (dict): Override any PPO constructor kwarg.

    Returns:
        Configured PPO agent.
    """
    features_dim = config.get("features_dim", 128)

    policy_kwargs = dict(
        features_extractor_class=SyndromeCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        share_features_extractor=True,
    )

    kwargs = dict(
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=config.get("log_dir"),
        seed=config.get("seed", 42),
    )
    kwargs.update(config.get("overrides", {}))

    return PPO("CnnPolicy", env, **kwargs)
