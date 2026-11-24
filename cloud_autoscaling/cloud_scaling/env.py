"""Small, reproducible Gymnasium environment for cloud auto-scaling.

One step is a five-minute interval. Arrivals and processing capacities are
counts of requests per interval (not rates per second). The simulator is a
teaching abstraction, not a production cloud performance model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .workload import SCENARIOS, WorkloadGenerator


@dataclass(frozen=True)
class CloudConfig:
    horizon: int = 288
    min_servers: int = 1
    max_servers: int = 8
    initial_servers: int = 3
    capacity_per_server: int = 100
    startup_steps: int = 2
    max_queue: int = 350
    running_cost: float = 0.06
    pending_cost: float = 0.02
    queue_penalty: float = 0.12  # per 100 requests waiting at end of interval
    drop_penalty: float = 2.5   # per 100 requests dropped
    switch_cost: float = 0.04
    scenario: str = "daily"

    def __post_init__(self) -> None:
        if not (1 <= self.min_servers <= self.initial_servers <= self.max_servers):
            raise ValueError("Server count constraints are inconsistent")
        if any(x <= 0 for x in (self.horizon, self.capacity_per_server, self.startup_steps, self.max_queue)):
            raise ValueError("Horizon, capacities and startup steps must be positive")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        if any(x < 0 for x in (self.running_cost, self.pending_cost, self.queue_penalty, self.drop_penalty, self.switch_cost)):
            raise ValueError("Cost/penalty coefficients must be nonnegative")


class CloudAutoScalingEnv(gym.Env):
    """Actions: 0 = remove one active server; 1 = hold; 2 = start one server.

    Observation is normalized:
      active/max_servers, queue/max_queue, arrivals/(max_servers*capacity),
      previous_arrivals/(max_servers*capacity), sin(time), cos(time),
      pending counts with 1..startup_steps steps left / max_servers.

    Pending servers become active when their remaining delay reaches zero at
    the *start* of a later step. A server ordered at t becomes usable at t+L.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, config: CloudConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or CloudConfig()
        if render_mode not in (None, "ansi"):
            raise ValueError("Only render_mode=None or 'ansi' is supported")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        size = 6 + self.cfg.startup_steps
        high = np.full(size, np.finfo(np.float32).max, dtype=np.float32)
        low = np.zeros(size, dtype=np.float32)
        low[4:6] = -1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._initialized = False

    def _observation(self) -> np.ndarray:
        denom = self.cfg.max_servers * self.cfg.capacity_per_server
        theta = 2.0 * np.pi * (self.t % self.cfg.horizon) / self.cfg.horizon
        counts = [self.pending.count(delay) / self.cfg.max_servers
                  for delay in range(1, self.cfg.startup_steps + 1)]
        return np.asarray([
            self.active / self.cfg.max_servers,
            self.queue / self.cfg.max_queue,
            self.arrivals / denom,
            self.previous_arrivals / denom,
            np.sin(theta), np.cos(theta), *counts,
        ], dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "step": self.t,
            "active_servers": self.active,
            "pending_servers": len(self.pending),
            "pending_delays": tuple(self.pending),
            "queue": self.queue,
            "arrivals": self.arrivals,
            "cumulative_cost": self.total_cost,
            "cumulative_dropped": self.total_dropped,
            "cumulative_arrivals": self.total_arrivals,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.active = self.cfg.initial_servers
        self.pending: list[int] = []
        self.queue = 0
        self.previous_arrivals = 0
        self.total_cost = 0.0
        self.total_dropped = 0
        self.total_arrivals = 0
        self.workload = WorkloadGenerator(self.np_random, self.cfg.scenario, self.cfg.horizon)
        self.arrivals = self.workload.next(0)
        self._initialized = True
        return self._observation(), self._info()

    def step(self, action: int):
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        if self.t >= self.cfg.horizon:
            raise RuntimeError("Episode finished: call reset()")
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        action = int(action)

        # Advance orders from previous steps, before the current control action.
        new_pending = []
        for remaining in self.pending:
            remaining -= 1
            if remaining == 0:
                self.active += 1
            else:
                new_pending.append(remaining)
        self.pending = new_pending

        switched = False
        invalid_action = False
        if action == 2:
            if self.active + len(self.pending) < self.cfg.max_servers:
                self.pending.append(self.cfg.startup_steps)
                switched = True
            else:
                invalid_action = True
        elif action == 0:
            if self.active > self.cfg.min_servers:
                self.active -= 1
                switched = True
            else:
                invalid_action = True

        capacity = self.active * self.cfg.capacity_per_server
        available = self.queue + self.arrivals
        served = min(available, capacity)
        overflow = available - served
        dropped = max(0, overflow - self.cfg.max_queue)
        self.queue = min(overflow, self.cfg.max_queue)

        cost = self.cfg.running_cost * self.active + self.cfg.pending_cost * len(self.pending)
        congestion_penalty = self.cfg.queue_penalty * self.queue / 100.0
        drop_penalty = self.cfg.drop_penalty * dropped / 100.0
        change_penalty = self.cfg.switch_cost if switched else 0.0
        reward = -(cost + congestion_penalty + drop_penalty + change_penalty)

        self.total_cost += cost + change_penalty
        self.total_dropped += dropped
        self.total_arrivals += self.arrivals
        step_info = {
            "served": served, "dropped": dropped, "queue": self.queue,
            "capacity": capacity, "cost": cost + change_penalty,
            "invalid_action": invalid_action,
            "queue_penalty": congestion_penalty,
            "drop_penalty": drop_penalty,
            "reward": reward,
        }
        self.previous_arrivals = self.arrivals
        self.t += 1
        truncated = self.t >= self.cfg.horizon
        if not truncated:
            self.arrivals = self.workload.next(self.t)
        else:
            self.arrivals = 0
        return self._observation(), float(reward), False, truncated, {**self._info(), **step_info}

    def render(self):
        if not self._initialized:
            return "Not initialized"
        return (f"t={self.t:03d} | arrivals={self.arrivals:4d} | "
                f"active={self.active} | pending={self.pending} | queue={self.queue}")

    def close(self):
        pass
