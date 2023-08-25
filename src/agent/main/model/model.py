import keras

from src.agent.main.model.singleton_meta import SingletonMeta
import tensorflow as tf
from overrides import override


class SingletonModel(metaclass=SingletonMeta):
    def __init__(self, input_dim=None, n_actions=5):
        if input_dim is None:
            input_dim = [220, 110, 4]
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_dim)
        conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(conv1)
        conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(conv2)
        flatten = tf.keras.layers.Flatten()(conv3)
        dense = tf.keras.layers.Dense(512)(flatten)
        outputs = tf.keras.layers.Dense(self.n_actions)(dense)
        return keras.Model(inputs=inputs, outputs=outputs)
