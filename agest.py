import model
import data
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir='./model', params={'learning_rate': 1e-6})

estimator.train(input_fn=lambda: data.input_fn("data", True, 8))

