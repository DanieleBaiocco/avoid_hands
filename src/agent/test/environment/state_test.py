import unittest

from src.agent.main.environment.state import State
from src.game.game_env import GameEnv


class StateTest(unittest.TestCase):

    def test_equals(self):
        game_env = GameEnv()
        obs1, _ = game_env.reset()
        state1 = State(obs1)
        _, _, _, stacks = state1.state.shape
        for i in range(1, stacks):
            self.assertFalse((state1.state[:, :, :, 0].numpy() - state1.state[:, :, :, i].numpy()).any())

        state2 = state1
        self.assertEqual(state1, state2)

    def test_preprocess_observation(self):
        game_env = GameEnv()
        obs1, _ = game_env.reset()
        state1 = State(obs1)
        obs2, _, _, _, _ = game_env.step(1)
        state2 = State(obs2, state1)

        for i in range(3):
            self.assertFalse((state1.state[:, :, :, i + 1].numpy() - state2.state[:, :, :, i].numpy()).any())

        for i in range(3):
            self.assertTrue((state2.state[:, :, :, 3].numpy() - state2.state[:, :, :, i].numpy()).any())


if __name__ == '__main__':
    unittest.main()
