import copy
import itertools

import numpy as np

from src.agent.main.environment.game_env_wrapper import GameEnvWrapper
from src.agent.main.environment.state import State


class ReplayMemory:
    def __init__(self,
                 game_env: GameEnvWrapper,
                 n_max_episodes=1000000,
                 n_init_episodes=50000):
        self.n_max_episodes = n_max_episodes
        self.n_init_episodes = n_init_episodes
        self.memory = []
        self.game_env = game_env
    def populate_memory(self):
        current_episode_steps = []
        current_state = self.game_env.reset()
        for _ in itertools.count():
            sample, terminated, info = self.game_env.step(current_state, uniform=True)
            current_episode_steps.append((sample, terminated))
            if terminated:
                self.memory.append(copy.deepcopy(current_episode_steps))
                if len(self.memory) == self.n_init_episodes:
                    break
                current_episode_steps.clear()
                current_state = self.game_env.reset()
            else:
                _, _, _, next_state = sample
                current_state = copy.deepcopy(next_state)
        self.memory.append([])


    def take_a_step(self, state: State, current_timestep):
        sample, terminated, info = self.game_env.step(state, current_timestep)
        self.memory[-1].append((sample, terminated))
        if terminated:
            if len(self.memory) > self.n_max_episodes:
                self.memory.pop(0)
            self.memory.append([])
            return self.game_env.reset()
        _, _, _, next_state = sample
        return copy.deepcopy(next_state)
