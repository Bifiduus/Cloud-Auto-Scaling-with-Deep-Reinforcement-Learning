"""Reference non-learning controllers. Students should design a stronger baseline."""
from __future__ import annotations
import numpy as np


class FixedCapacityPolicy:
    def __init__(self, env, servers: int = 5):
        self.env = env
        self.servers = servers

    def act(self, observation: np.ndarray, info: dict) -> int:
        active = info["active_servers"]
        pending = info["pending_servers"]
        if active + pending < self.servers:
            return 2
        if active > self.servers:
            return 0
        return 1


class ThresholdPolicy:
    """Simple reactive policy with hysteresis; not an optimized controller."""
    def __init__(self, env, upper: float = 0.8, lower: float = 0.35):
        self.env = env
        self.upper = upper
        self.lower = lower

    def act(self, observation: np.ndarray, info: dict) -> int:
        cfg = self.env.cfg
        active, pending = info["active_servers"], info["pending_servers"]
        demand = info["arrivals"] + info["queue"]
        if demand > self.upper * active * cfg.capacity_per_server and active + pending < cfg.max_servers:
            return 2
        if demand < self.lower * active * cfg.capacity_per_server and active > cfg.min_servers:
            return 0
        return 1


class RandomPolicy:
    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, info: dict) -> int:
        return int(self.rng.integers(0, 3))
