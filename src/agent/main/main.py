from src.agent.main.environment.game_env_wrapper import GameEnvWrapper
from src.agent.main.memory.cache import Cache
from src.agent.main.memory.replay_memory import ReplayMemory
from src.agent.main.model.algorithm import Algorithm
from src.agent.main.model.learner import Learner
from src.game.game_env import GameEnv
import tensorflow as tf

if __name__ == '__main__':
    game_env = GameEnv()
    game_env_wrapper = GameEnvWrapper(game_env)
    replay_memory = ReplayMemory(game_env_wrapper)
    cache = Cache()
    learner = Learner(optimizer=tf.keras.optimizers.RMSprop(0.0003, 0.99, 0.0, 1e-6),
                      loss_fn=tf.keras.losses.MeanSquaredError())
    deeoqnetworklambda = Algorithm(game_env_wrapper, replay_memory, cache, learner)
    deeoqnetworklambda.run()


