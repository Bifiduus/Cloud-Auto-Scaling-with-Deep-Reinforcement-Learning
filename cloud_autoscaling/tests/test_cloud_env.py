"""Tests for the shared simulator; execute with `python -m pytest -q`."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from cloud_scaling import CloudAutoScalingEnv, CloudConfig


def test_gymnasium_api():
    env = CloudAutoScalingEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_reproducible_workload():
    a, b = CloudAutoScalingEnv(), CloudAutoScalingEnv()
    oa, _ = a.reset(seed=12)
    ob, _ = b.reset(seed=12)
    np.testing.assert_allclose(oa, ob)
    for _ in range(12):
        xa = a.step(1)
        xb = b.step(1)
        np.testing.assert_allclose(xa[0], xb[0])
        assert xa[1:] == xb[1:]


def test_startup_takes_two_intervals():
    env = CloudAutoScalingEnv(CloudConfig(initial_servers=2, startup_steps=2, horizon=8))
    env.reset(seed=3)
    _, _, _, _, info = env.step(2)
    assert info["active_servers"] == 2
    assert info["pending_delays"] == (2,)
    _, _, _, _, info = env.step(1)
    assert info["active_servers"] == 2
    assert info["pending_delays"] == (1,)
    _, _, _, _, info = env.step(1)
    assert info["active_servers"] == 3
    assert info["pending_delays"] == ()


def test_queue_conservation_per_step():
    env = CloudAutoScalingEnv(CloudConfig(horizon=20))
    env.reset(seed=9)
    for _ in range(20):
        old_queue = env.queue
        arrivals = env.arrivals
        _, _, _, truncated, info = env.step(1)
        assert old_queue + arrivals == info["served"] + info["queue"] + info["dropped"]
        assert info["queue"] <= env.cfg.max_queue
        if truncated:
            break


def test_limits():
    env = CloudAutoScalingEnv(CloudConfig(initial_servers=1, max_servers=2, horizon=12))
    env.reset(seed=2)
    _, _, _, _, info = env.step(0)
    assert info["active_servers"] == 1 and info["invalid_action"]
    for _ in range(8):
        _, _, _, _, info = env.step(2)
        assert info["active_servers"] + info["pending_servers"] <= 2


def test_observations_and_episode_end():
    env = CloudAutoScalingEnv(CloudConfig(horizon=3))
    obs, _ = env.reset(seed=10)
    assert env.observation_space.contains(obs)
    for step in range(3):
        obs, reward, terminated, truncated, info = env.step(1)
        assert not terminated
        assert truncated == (step == 2)
        assert env.observation_space.contains(obs)
    with pytest.raises(RuntimeError):
        env.step(1)
