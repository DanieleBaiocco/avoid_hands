import numpy as np

from src.agent.main.environment.state import State
from src.agent.main.model.model import SingletonModel


class Policy:
    def __init__(self, actions: dict, epsilon_start=0.9, epsilon_end=0.15, epsilon_decay_steps=1000):
        self.q_estimator = SingletonModel()
        self.n_actions = len(actions)
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.policy = self._build_epsilon_greedy_policy()
        self.uniform_action_probs = [1 / self.n_actions for _ in range(self.n_actions)]

    def _build_epsilon_greedy_policy(self):
        def epsilon_greedy_policy(state: State, current_time_step):
            epsilon = self.epsilons[min(current_time_step, self.epsilon_decay_steps)]
            print(epsilon)
            A = np.ones(self.n_actions, dtype=float) * epsilon / self.n_actions
            q_values = self.q_estimator.model(state.state)
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            return A

        return epsilon_greedy_policy

    def epsilon_greedy_action_selection(self, state, current_time_step) -> int:
        action = np.random.choice(np.arange(self.n_actions), p=self.policy(state, current_time_step))
        return action

    def uniform_action_selection(self) -> int:
        action = np.random.choice(np.arange(self.n_actions), p=self.uniform_action_probs)
        return action
