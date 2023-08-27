import itertools
import os
import random

from src.agent.main.environment.game_env_wrapper import GameEnvWrapper
from src.agent.main.memory.cache import Cache
from src.agent.main.memory.replay_memory import ReplayMemory
from src.agent.main.model.learner import Learner
import numpy as np
from gym.wrappers.record_video import RecordVideo


class Algorithm:
    def __init__(self, game_env: GameEnvWrapper,
                 replay_memory: ReplayMemory,
                 cache: Cache,
                 learner: Learner,
                 n_episodes,
                 minibatch_size,
                 monitor_path,
                 record_video_every=5
                 ):
        game_env = RecordVideo(game_env, video_folder=os.path.join(monitor_path, "video"),
                               episode_trigger=lambda episode_number: True)
        self.game_env = game_env
        self.replay_memory = replay_memory
        self.cache = cache
        self.learner = learner
        self.n_episodes = n_episodes
        self.minibatch_size = minibatch_size

    def build_batches(self, samples):
        state_batch, action_batch, target_batch = [], [], []
        for el in samples:
            state, action, target = el
            state_batch.append(state.state)
            action_batch.append(action)
            target_batch.append(target)
        state_batch = np.stack(state_batch, axis=1)[0]
        return state_batch, np.array(action_batch), np.array(target_batch)

    def run(self):
        self.replay_memory.populate_memory()
        for i_episode in range(self.n_episodes):
            loss = None
            initial_state = self.game_env.reset()
            for step in itertools.count():
                if self.cache.is_cache_refreshable(step):
                    self.cache.build_cache(self.replay_memory)
                    for i in range(self.cache.S // self.minibatch_size):
                        samples = random.sample(self.cache.cache, self.minibatch_size)
                        states_batch, action_batch, targets_batch = self.build_batches(samples)
                        loss = self.learner.train_step(data=states_batch, target=targets_batch,
                                                       batch_size=self.minibatch_size, actions_taken=action_batch)
                print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    step, self.learner.get_time_step(), i_episode + 1, self.n_episodes, loss), end="")
                initial_state, terminated = self.replay_memory.take_a_step(initial_state, self.learner.get_time_step())
                if terminated:
                    break
        self.game_env.close()