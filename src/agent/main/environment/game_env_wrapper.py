from src.agent.main.environment.policy import Policy
from src.agent.main.environment.state import State
from src.game.game_env import GameEnv


class GameEnvWrapper:
    def __init__(self, env: GameEnv):
        self.env = env
        self.policy = Policy(env.actions)

    def reset(self):
        obs, _ = self.env.reset()
        return State(obs)

    def step(self, state: State = None, uniform=False, current_time_step=None):
        action = 0
        if uniform:
            action = self.policy.uniform_action_selection()
        else:
            assert state is not None and current_time_step is not None
            action = self.policy.epsilon_greedy_action_selection(state, current_time_step)
        obs, reward, terminated, _, info = self.env.step(action)
        next_state = State(obs, previous_state=state)
        return (state, action, reward, next_state), terminated, info
