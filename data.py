import numpy as np
import tensorflow as tf


def get_input_fn(data_set, labels, num_epochs=None, shuffle=False, batch_size=100):
    return tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(data_set)},
        y=np.array(labels),
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle=shuffle)
