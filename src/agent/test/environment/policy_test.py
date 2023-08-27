import unittest

from src.agent.main.environment.game_env_wrapper import GameEnvWrapper
from src.agent.main.environment.policy import Policy
from src.game.game_env import GameEnv
from pygame.locals import K_LEFT, K_UP, K_DOWN, K_RIGHT


class PolicyTest(unittest.TestCase):
    def test_epsilon_greedy_action_selection(self):
        game_env_wrapper = GameEnvWrapper(epsilon_start=0.9,
                                          epsilon_end=0.15,
                                          epsilon_decay_steps=10000)
        policy = Policy(game_env_wrapper.actions, epsilon_start=0.9,
                        epsilon_end=0.15,
                        epsilon_decay_steps=10000)
        time_step = 0
        state = game_env_wrapper.reset()
        action_to_take = policy.epsilon_greedy_action_selection(state, time_step)
        self.assertTrue(action_to_take in [0, 1, 2, 3, 4])

    def test_epsilon_greedy_policy(self):
        game_env = GameEnv()
        policy = Policy(game_env.actions, epsilon_start=0.9,
                        epsilon_end=0.15,
                        epsilon_decay_steps=10000)
        action_to_take = policy.uniform_action_selection()
        self.assertTrue(action_to_take in [0, 1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
