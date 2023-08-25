import unittest

from src.agent.main.environment.game_env_wrapper import GameEnvWrapper
from src.game.game_env import GameEnv


class GameEnvWrapperTest(unittest.TestCase):

    # IL RIFERIMENTO DI STATE MAGARI PRESENTE COME NEXT STATE NELLO STEP PRECEDENTE E' LO STESSO PER LO STATE DI
    # PARTENZA ALLO STEP SUCCESSIVO
    def test_step(self):
        game_env = GameEnv()
        game_env_wrapper = GameEnvWrapper(game_env)
        init_state = game_env_wrapper.reset()
        time_step = 0
        sample, terminated, info = game_env_wrapper.step(init_state, time_step)
        state, action, reward, next_state = sample
        self.assertEqual(init_state, state)


if __name__ == '__main__':
    unittest.main()
