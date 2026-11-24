"""Synthetic exogenous workloads. Values are requests per five-minute slot."""
from __future__ import annotations
import numpy as np

SCENARIOS = ("daily", "volatile", "shifted_peak")


def mean_workload(step: int, steps_per_day: int = 288, scenario: str = "daily") -> float:
    """Expected arrivals. The scenario is fixed for an episode."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown workload scenario: {scenario}")
    hour = 24.0 * (step % steps_per_day) / steps_per_day
    morning = 120.0 * np.exp(-0.5 * ((hour - 9.0) / 1.6) ** 2)
    midday = 170.0 * np.exp(-0.5 * ((hour - 13.0) / 2.1) ** 2)
    peak_hour = 20.0 if scenario == "shifted_peak" else 18.0
    evening = 545.0 * np.exp(-0.5 * ((hour - peak_hour) / 0.48) ** 2)
    return float(125.0 + morning + midday + evening)


class WorkloadGenerator:
    """AR(1) residual around the known scenario's daily expectation.

    The current arrival rate and time are observations. For a fixed scenario,
    their combination is sufficient to predict the residual distribution.
    """
    def __init__(self, rng: np.random.Generator, scenario: str = "daily", steps_per_day: int = 288):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown workload scenario: {scenario}")
        self.rng = rng
        self.scenario = scenario
        self.steps_per_day = steps_per_day
        self.residual = 0.0

    def next(self, step: int) -> int:
        sigma = 42.0 if self.scenario == "volatile" else 17.0
        self.residual = 0.78 * self.residual + float(self.rng.normal(0.0, sigma))
        value = mean_workload(step, self.steps_per_day, self.scenario) + self.residual
        return max(0, int(round(value)))
