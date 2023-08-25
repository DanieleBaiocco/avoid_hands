import random

import numpy as np

from src.agent.main.environment.game_env_wrapper import GameEnvWrapper
from src.agent.main.memory.cache import Cache
from src.agent.main.memory.replay_memory import ReplayMemory
from src.agent.main.model.learner import Learner


class Algorithm:
    def __init__(self, game_env: GameEnvWrapper,
                 replay_memory: ReplayMemory,
                 cache: Cache,
                 learner: Learner,
                 n_episodes=1000,
                 minibatch_size=32,
                 training_steps=1000000,
                 ):
        self.game_env = game_env
        self.replay_memory = replay_memory
        self.cache = cache
        self.learner = learner
        self.n_episodes = n_episodes
        self.minibatch_size = minibatch_size
        self.training_steps = training_steps

    def run(self):
        self.replay_memory.populate_memory()
        initial_state = self.game_env.reset()
        for training_step in range(self.training_steps):
            if self.cache.refresh_frequency(training_step):
                self.cache.build_cache(self.replay_memory)
                for i in range(self.cache.S // self.minibatch_size):
                    samples = random.sample(self.cache.cache, self.minibatch_size)
                    states_batch, action_batch, targets_batch = samples
                    loss = self.learner.train_step(states_batch, targets_batch, action_batch)
            initial_state = self.replay_memory.take_a_step(initial_state, self.learner.get_time_step())
