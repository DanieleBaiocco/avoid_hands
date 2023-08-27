import unittest

from src.agent.main.environment.game_env_wrapper import GameEnvWrapper
from src.agent.main.memory.cache import Cache
from src.agent.main.memory.replay_memory import ReplayMemory
from src.game.game_env import GameEnv


class CacheTest(unittest.TestCase):
    def test_build_cache(self):
        game_env_wrapper = GameEnvWrapper(epsilon_start=0.9,
                                          epsilon_end=0.15,
                                          epsilon_decay_steps=10000)
        replay_memory = ReplayMemory(game_env_wrapper, n_init_episodes=2, n_max_episodes=2)
        cache = Cache(block_size=6, cache_size=60, lmbda=5,
                      discount_factor=0.99,
                      refresh_frequency=50)
        replay_memory.populate_memory()
        cache.build_cache(replay_memory)
        self.assertTrue(len(cache.cache) == cache.S)


if __name__ == '__main__':
    unittest.main()
