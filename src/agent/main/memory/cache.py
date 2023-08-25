import random
import numpy as np
from src.agent.main.memory.replay_memory import ReplayMemory
from src.agent.main.model.model import SingletonModel


class Cache:
    def __init__(self, cache_size=80000, block_size=100, lmbda=5, discount_factor = 0.99,
                 refresh_frequency=10000):

        self.q_estimator = SingletonModel()
        self.S = cache_size
        self.B = block_size
        self.lmbda = lmbda
        self.discount_factor = discount_factor
        self.cache = None
        self.refresh_frequency = refresh_frequency

    def build_cache(self, replay_memory: ReplayMemory):
        self.cache = []
        experiences = np.random.choice(replay_memory.memory, size=self.S // self.B, replace=True)
        # non so se ci va il -1
        blocks = [experience[random.randint(0, len(experience) - self.B - 1)::] for experience in experiences]
        for block in blocks:
            (state, action, reward, next_state), terminated = block[-1]
            R_lambda = 0
            if terminated:
                R_lambda = reward
            else:
                q_values_next = self.q_estimator.model(next_state)
                Rnext_lambda = np.amax(q_values_next, axis=1)
                R_lambda = reward + self.lmbda * (self.discount_factor * Rnext_lambda + (1 - self.lmbda) * Rnext_lambda)
            # non so se serve deep copy per state
            self.cache.append((state, action, R_lambda))
            for sample in block[::-2].reverse():
                (state, action, reward, next_state), _ = sample
                q_values_next = self.q_estimator.model(next_state)
                best_action = np.amax(q_values_next, axis=1)
                R_lambda = reward + self.lmbda * (self.discount_factor * R_lambda + (1 - self.lmbda) * best_action)
                self.cache.append((state, action, R_lambda))

    def is_cache_refreshable(self, timestep: int) -> bool:
        return timestep % self.refresh_frequency == 0
