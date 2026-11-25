# Cloud Auto-Scaling with Deep RL — Student starter kit

**M2 Artificial Intelligence · 2026–2027 · Project in pairs**

This is a **synthetic simulator**, not a Kubernetes cluster. It provides the environment and simple reference controllers. You implement and evaluate the RL agents.

## Setup (macOS/Linux)

Use Python 3.11 or 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/evaluate.py --scenario daily --episodes 5
```

Windows activation: `.venv\Scripts\activate`.

## Files

- `cloud_scaling/env.py`: Gymnasium simulator, complete.
- `cloud_scaling/workload.py`: three synthetic demand profiles, complete.
- `cloud_scaling/baselines.py`: fixed, reactive threshold and random controllers, complete.
- `scripts/evaluate.py`: reproducible baseline evaluation and CSV export.
- `student_agent.py`: **incomplete** DQN API that you must implement.
- `tests/test_cloud_env.py`: simulator invariants (the RL agent is not tested here).

## API example

```python
from cloud_scaling import CloudAutoScalingEnv, CloudConfig

env = CloudAutoScalingEnv(CloudConfig(scenario="daily"), render_mode="ansi")
obs, info = env.reset(seed=42)
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(1)  # hold
    print(env.render(), reward)
    if terminated or truncated:
        break
env.close()
```

**Actions:** `0` = remove one active server, `1` = hold, `2` = order a new server. Invalid scale actions are ignored and reported in `info`. Pending servers take two simulation steps before becoming active. One step is **5 minutes**; an episode represents 24 hours by default (288 steps). A server can handle 100 requests per step.

**Observation (normalized float32):** `[active/max, queue/max_queue, arrivals/max_capacity, previous_arrivals/max_capacity, sin(time), cos(time), pending_with_1_step_left/max, pending_with_2_steps_left/max]`. The last two fields generalize to the configured startup delay. `info` is for interpretation and metrics, **not extra policy inputs**. The scenario is set when constructing an environment and is not automatically made available to the policy.

**Workload scenarios:** `daily`, `volatile`, `shifted_peak`. All are synthetic. Fix seeds for reproducibility and use evaluation seeds that were not used for training. You may use different scenarios for train/test experiments, but report the split.

**Simulator interpretation:** arrivals join a FIFO-like aggregate backlog; a server can process a limited number of requests per step; excess requests are queued up to a fixed limit, then dropped. Only aggregate counts are simulated, not individual latencies. Do **not** report actual latency percentiles or SLA guarantees based on this model. The penalty weights are artificial and should be discussed critically.

**Episode termination:** the environment returns `terminated=False` and `truncated=True` at the time limit. Document how your learning algorithm treats truncation when calculating TD targets.

## What is not supplied

No working DQN, no replay buffer, no training script, no pretrained weights, no privileged future demand forecasts. Those are project tasks.

## Useful commands

```bash
python scripts/evaluate.py --scenario daily --episodes 10 --output results_daily.csv
python scripts/evaluate.py --scenario volatile --episodes 10 --output results_volatile.csv
python scripts/evaluate.py --scenario shifted_peak --episodes 10 --output results_shifted.csv
```

When csv files are built with the previous commands, we can do the following command from 'cloud_autoscaling/' :

```bash
python scripts/evaluate_baselines.py --input-dir .
```

This will compute the mean of the metrics for each (scenario, baseline)