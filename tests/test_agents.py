"""Tests for RL agent creation, CNN feature extractors, and wrappers."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.agents.callbacks import DecodingEvalCallback
from src.agents.dqn_agent import create_dqn_agent
from src.agents.networks import SyndromeCNN
from src.agents.ppo_agent import create_ppo_agent
from src.envs.surface_code_env import SurfaceCodeEnv
from src.envs.wrappers import FlattenSyndromeWrapper, PadToSquareWrapper


# --- CNN Feature Extractor ---


class TestSyndromeCNN:
    @pytest.mark.parametrize(
        "shape",
        [(6, 4, 4), (8, 6, 6), (10, 8, 8)],
        ids=["d3", "d5", "d7"],
    )
    def test_forward_pass(self, shape):
        """CNN should accept observation shapes for d=3, 5, 7."""
        obs_space = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
        cnn = SyndromeCNN(obs_space, features_dim=128)
        batch = torch.rand(4, *shape)
        out = cnn(batch)
        assert out.shape == (4, 128)

    def test_custom_features_dim(self):
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(6, 4, 4), dtype=np.float32)
        cnn = SyndromeCNN(obs_space, features_dim=64)
        batch = torch.rand(2, 6, 4, 4)
        out = cnn(batch)
        assert out.shape == (2, 64)

    def test_single_sample(self):
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(6, 4, 4), dtype=np.float32)
        cnn = SyndromeCNN(obs_space, features_dim=128)
        single = torch.rand(1, 6, 4, 4)
        out = cnn(single)
        assert out.shape == (1, 128)


# --- Wrappers ---


class TestFlattenSyndromeWrapper:
    def test_flattened_shape(self):
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        wrapped = FlattenSyndromeWrapper(env)
        obs, _ = wrapped.reset()
        expected_dim = 6 * 4 * 4  # (syndrome_rounds+2) * grid_h * grid_w
        assert obs.shape == (expected_dim,)
        assert wrapped.observation_space.shape == (expected_dim,)

    def test_values_preserved(self):
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        wrapped = FlattenSyndromeWrapper(env)
        obs, _ = wrapped.reset()
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)


class TestPadToSquareWrapper:
    def test_already_square(self):
        """d=3 grids are 4x4 (already square), padding should be no-op."""
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        wrapped = PadToSquareWrapper(env)
        obs, _ = wrapped.reset()
        assert obs.shape[1] == obs.shape[2]  # Square

    def test_pad_preserves_content(self):
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        wrapped = PadToSquareWrapper(env)
        obs, _ = wrapped.reset()
        c, side, side2 = obs.shape
        assert side == side2  # Square
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)


# --- DQN Agent ---


class TestDQNAgent:
    def test_creation(self):
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        agent = create_dqn_agent(env, {"seed": 42})
        assert agent is not None

    def test_predict(self):
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        agent = create_dqn_agent(env, {"seed": 42})
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        assert 0 <= int(action) < env.action_space.n

    def test_short_training(self):
        """Agent should train for a few steps without errors."""
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        agent = create_dqn_agent(
            env,
            {"seed": 42, "overrides": {"learning_starts": 32, "verbose": 0}},
        )
        agent.learn(total_timesteps=100)


# --- PPO Agent ---


class TestPPOAgent:
    def test_creation(self):
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        agent = create_ppo_agent(env, {"seed": 42})
        assert agent is not None

    def test_predict(self):
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        agent = create_ppo_agent(env, {"seed": 42})
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        assert 0 <= int(action) < env.action_space.n

    def test_short_training(self):
        """Agent should train for a few steps without errors."""
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        agent = create_ppo_agent(
            env,
            {"seed": 42, "overrides": {"n_steps": 64, "verbose": 0}},
        )
        agent.learn(total_timesteps=128)


# --- Eval Callback ---


class TestDecodingEvalCallback:
    def test_callback_runs(self):
        """Eval callback should run during short training."""
        env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)
        eval_env = SurfaceCodeEnv.from_config(distance=3, physical_error_rate=0.01)

        callback = DecodingEvalCallback(
            eval_env=eval_env,
            mwpm_ler=0.05,
            eval_freq=64,
            eval_episodes=10,
            verbose=0,
        )

        agent = create_dqn_agent(
            env,
            {"seed": 42, "overrides": {"learning_starts": 32, "verbose": 0}},
        )
        agent.learn(total_timesteps=128, callback=callback)

        assert len(callback.eval_results) >= 1
        result = callback.eval_results[0]
        assert "logical_error_rate" in result
        assert "mwpm_ratio" in result
        assert 0.0 <= result["logical_error_rate"] <= 1.0
