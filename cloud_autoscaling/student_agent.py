"""Student work: implement DQN + replay + target network here.

Do not import or use the simulator's internal hidden variables during action
selection. `observation` is the agent's input; `info` is for diagnostics only.
"""


class DQNAgent:
    def __init__(self, observation_dim: int, action_dim: int, **kwargs):
        # TODO: Q network, target network, optimizer, replay buffer.
        raise NotImplementedError("Implement the DQN agent")

    def act(self, observation, epsilon: float = 0.0) -> int:
        # TODO: epsilon-greedy action selection.
        raise NotImplementedError

    def update(self, state, action, reward, next_state, terminated, truncated):
        # TODO: store transition, sample minibatch, compute TD target,
        # detach target, optimize, periodically synchronize target network.
        # Document handling of time-limit truncation in the report.
        raise NotImplementedError
