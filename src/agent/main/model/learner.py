from src.agent.main.model.model import SingletonModel
import tensorflow as tf


class Learner:
    def __init__(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.q_estimator = SingletonModel()

    @tf.function
    def train_step(self, data, target, batch_size, actions_taken):
        with tf.GradientTape() as tape:
            predictions = self.q_estimator.model(data, training=True)
            general_indxs = tf.cast(tf.range(batch_size) * tf.shape(predictions)[1], np.int64)
            gather_indices = general_indxs + actions_taken
            action_predictions = tf.gather(tf.reshape(predictions, [-1]), gather_indices)
            loss_value = self.loss_fn(target, action_predictions)
        grads = tape.gradient(loss_value, self.q_estimator.model.trainable_weights)
        print("Gradients to apply ", grads)
        self.optimizer.apply_gradients(zip(grads, self.q_estimator.model.trainable_weights))
        print(loss_value)
        return loss_value

    def get_time_step(self):
        return self.optimizer.iterations.numpy()
