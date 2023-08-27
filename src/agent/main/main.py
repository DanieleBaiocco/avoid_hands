import os

from src.agent.main.environment.game_env_wrapper import GameEnvWrapper
from src.agent.main.memory.cache import Cache
from src.agent.main.memory.replay_memory import ReplayMemory
from src.agent.main.model.algorithm import Algorithm
from src.agent.main.model.learner import Learner
from src.game.game_env import GameEnv
import tensorflow as tf

if __name__ == '__main__':
    game_env_wrapper = GameEnvWrapper(epsilon_start=0.9,
                                      epsilon_end=0.15,
                                      epsilon_decay_steps=10000)
    replay_memory = ReplayMemory(game_env_wrapper,
                                 n_max_episodes=10,
                                 n_init_episodes=5)
    cache = Cache(cache_size=20,
                  block_size=10,
                  lmbda=5,
                  discount_factor=0.99,
                  refresh_frequency=50)
    learner = Learner(optimizer=tf.keras.optimizers.RMSprop(0.0003, 0.99, 0.0, 1e-6),
                      loss_fn=tf.keras.losses.MeanSquaredError())
    monitor_path = os.path.abspath("./records/")
    print(monitor_path)
    deeoqnetworklambda = Algorithm(game_env_wrapper,
                                   replay_memory,
                                   cache,
                                   learner,
                                   n_episodes=30,
                                   minibatch_size=4,
                                   monitor_path=monitor_path,
                                   record_video_every=1)
    deeoqnetworklambda.run()
