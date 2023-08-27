import unittest

from src.agent.main.environment.game_env_wrapper import GameEnvWrapper


class GameEnvWrapperTest(unittest.TestCase):
    def test_step(self):
        game_env_wrapper = GameEnvWrapper(epsilon_start=0.9,
                                          epsilon_end=0.15,
                                          epsilon_decay_steps=10000)
        init_state = game_env_wrapper.reset()
        time_step = 0
        sample, terminated, info = game_env_wrapper.step(state=init_state, current_time_step=time_step)
        state, action, reward, next_state = sample
        self.assertEqual(init_state, state)


if __name__ == '__main__':
    unittest.main()
