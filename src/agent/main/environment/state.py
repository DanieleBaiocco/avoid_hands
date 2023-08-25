import tensorflow as tf


class State:

    def __init__(self, observation, previous_state=None):
        self.state = _preprocess_observation(observation, previous_state)

    def __eq__(self, obj):
        if not isinstance(obj, State):
            return False
        return not (self.state.numpy() - obj.state.numpy()).any()


def _preprocess_observation(observation, previous_state: State, output_dim=None):
    observation = tf.expand_dims(observation, axis=0)
    if output_dim is None:
        output_dim = [220, 110]
    output = tf.image.rgb_to_grayscale(observation)
    output = tf.image.resize(
        output, output_dim, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output = tf.squeeze(output, axis=3)
    if previous_state is None:
        output = tf.stack([output] * 4, axis=3)
    else:
        output = tf.expand_dims(output, axis=3)
        output = tf.concat((previous_state.state[:, :, :, 1:], output[:, :, :, :]), axis=3)
    return output
