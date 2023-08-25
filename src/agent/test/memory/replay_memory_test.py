import copy
import itertools
import unittest

from src.agent.main.environment.game_env_wrapper import GameEnvWrapper
from src.agent.main.environment.state import State
from src.agent.main.memory.replay_memory import ReplayMemory
from src.game.game_env import GameEnv


class ReplayMemoryTest(unittest.TestCase):
    def test_populate_memory(self):
        game_env = GameEnv()
        game_env_wrapper = GameEnvWrapper(game_env)
        replay_memory = ReplayMemory(game_env_wrapper, n_init_episodes=2, n_max_episodes=7)
        replay_memory.populate_memory()
        for k, episode in enumerate(replay_memory.memory):
            for i, (sample, terminated) in enumerate(episode):
                if i == len(episode) - 1:
                    self.assertTrue(terminated)
                    break
                else:
                    self.assertFalse(terminated)
                    state1 = sample[3]
                    sample_next, _ = replay_memory.memory[k][i + 1]
                    state2 = sample_next[0]
                    self.assertTrue(state1 == state2)
        self.assertTrue(replay_memory.n_init_episodes == len(replay_memory.memory) - 1)

    def test_take_a_step(self):
        game_env = GameEnv()
        n_max_episodes = 2
        n_init_episodes = 2
        game_env_wrapper = GameEnvWrapper(game_env)
        replay_memory = ReplayMemory(game_env_wrapper, n_init_episodes=n_init_episodes, n_max_episodes=n_max_episodes)
        replay_memory.populate_memory()
        init_state: State = game_env_wrapper.reset()

        for _ in itertools.count():
            next_state = replay_memory.take_a_step(init_state, 1)
            if not replay_memory.memory[-1]:
                break
            init_state = copy.deepcopy(next_state)

        self.assertTrue(len(replay_memory.memory) == n_max_episodes + 1)
        self.assertTrue(replay_memory.memory[-1] == [])


if __name__ == '__main__':
    unittest.main()
