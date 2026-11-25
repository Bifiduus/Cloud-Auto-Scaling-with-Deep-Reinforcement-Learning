from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud_scaling import CloudAutoScalingEnv, CloudConfig

def main():
    actions = [0, 0, 0, 1, 2, 2, 1, 1, 2, 0, 1, 1, 1, 2, 1, 1]

    env = CloudAutoScalingEnv(CloudConfig(scenario="daily"), render_mode="ansi")
    obs, info = env.reset(seed=42)
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(env.render(), reward)
        if terminated or truncated:
            break
    env.close()

if __name__ == "__main__":
    main()