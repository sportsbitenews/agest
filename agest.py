import model
import data
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)

estimator = tf.estimator.Estimator(
    model_fn=model.model_fn,
    model_dir='./model',
    params={'learning_rate': 1e-4})

#estimator.train(input_fn=lambda: data.input_fn("wiki_crop", True, 10))

ev = estimator.evaluate(input_fn=lambda: data.input_fn("eval", False, 1))

print("Accuracy: ", ev["accuracy"])

res = estimator.predict(input_fn=lambda: data.input_fn("test", False, 1))

for r in res:
    print(r["age"])