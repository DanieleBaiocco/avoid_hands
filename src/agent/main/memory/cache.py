import random
import numpy as np

from src.agent.main.environment.state import State
from src.agent.main.memory.replay_memory import ReplayMemory
from src.agent.main.model.model import SingletonModel


class Cache:
    def __init__(self, cache_size, block_size, lmbda, discount_factor,
                 refresh_frequency=10000):
        assert cache_size % block_size == 0
        self.q_estimator = SingletonModel()
        self.S = cache_size
        self.B = block_size
        self.lmbda = lmbda
        self.discount_factor = discount_factor
        self.cache = None
        self.refresh_frequency = refresh_frequency

    def _build_blocks(self, experiences):
        blocks = []
        for experience in experiences:
            start_idx = random.randint(0, len(experience) - self.B)
            block = experience[start_idx:start_idx + self.B]
            blocks.append(block)
        return blocks

    def _compute_r_lambda(self, reward, next_state: State, R_lambda_previous=None):
        q_values_next = self.q_estimator.model(next_state.state)
        best_action = np.amax(q_values_next, axis=1)[0]
        if R_lambda_previous is not None:
            R_lambda = reward + self.lmbda * (self.discount_factor * R_lambda_previous + (1 - self.lmbda) * best_action)
        else:
            R_lambda = reward + self.lmbda * (self.discount_factor * best_action + (1 - self.lmbda) * best_action)
        return R_lambda

    def build_cache(self, replay_memory: ReplayMemory):
        self.cache = []
        curr_mem = replay_memory.memory
        if len(curr_mem[-1]) < self.B:
            curr_mem = curr_mem[:-1]
        experiences = random.choices(curr_mem, k=self.S // self.B)
        blocks = self._build_blocks(experiences)
        for block in blocks:
            reversed_block = block[::-1]
            (state, action, reward, next_state), terminated = reversed_block[0]
            if terminated:
                R_lambda = reward
            else:
                R_lambda = self._compute_r_lambda(reward, next_state)
            self.cache.append((state, action, R_lambda))
            for sample in reversed_block[1:]:
                (state, action, reward, next_state), _ = sample
                R_lambda = self._compute_r_lambda(reward, next_state, R_lambda)
                self.cache.append((state, action, R_lambda))
        random.shuffle(self.cache)

    def is_cache_refreshable(self, timestep: int) -> bool:
        return timestep % self.refresh_frequency == 0
