"""Evaluate provided non-learning policies with matched workload seeds."""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

# Allow running via `python scripts/evaluate.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud_scaling import CloudAutoScalingEnv, CloudConfig
from cloud_scaling.baselines import FixedCapacityPolicy, ThresholdPolicy, RandomPolicy


def rollout(env, policy, seed: int):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    switches = 0
    while True:
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        switches += int(action != 1 and not info["invalid_action"])
        if terminated or truncated:
            return {
                "seed": seed, "reward": round(total_reward, 3),
                "cost": round(info["cumulative_cost"], 3),
                "drop_rate": round(info["cumulative_dropped"] / max(info["cumulative_arrivals"], 1), 5),
                "dropped": info["cumulative_dropped"],
                "arrivals": info["cumulative_arrivals"], "switches": switches,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["daily", "volatile", "shifted_peak"], default="daily")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="baseline_results.csv")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    env = CloudAutoScalingEnv(CloudConfig(scenario=args.scenario))
    rows = []
    for name, policy in [
        ("fixed5", FixedCapacityPolicy(env, servers=5)),
        ("threshold", ThresholdPolicy(env)),
        ("random", RandomPolicy(seed=2026)),
    ]:
        for episode in range(args.episodes):
            result = rollout(env, policy, seed=1000 + episode)
            rows.append({"policy": name, "scenario": args.scenario, **result})
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
