from src.agent.main.environment.policy import Policy
from src.agent.main.environment.state import State
from src.game.game_env import GameEnv


class GameEnvWrapper(GameEnv):
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay_steps):
        super().__init__()
        self.policy = Policy(self.actions, epsilon_start, epsilon_end, epsilon_decay_steps)

    def reset(self, seed=None, options=None):
        obs, _ = super(GameEnvWrapper, self).reset()
        return State(obs)

    def step(self, state: State, current_time_step=None, uniform=False):
        action = 0
        if uniform:
            action = self.policy.uniform_action_selection()
        else:
            assert current_time_step is not None
            action = self.policy.epsilon_greedy_action_selection(state, current_time_step)
        obs, reward, terminated, _, info = super(GameEnvWrapper, self).step(action)
        next_state = State(obs, previous_state=state)
        return (state, action, reward, next_state), terminated, info
