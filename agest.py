import model1
import data
import scipy
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras


tf.logging.set_verbosity(tf.logging.INFO)

# estimator = tf.estimator.Estimator(
#     model_fn=model1.model_fn,
#     model_dir="./model",
#     params={"learning_rate": 1e-4,
#             "class_weights": [1.0 for i in range(100)]})
#
# for i in range(15):
#     estimator.train(input_fn=lambda: data.input_fn("wiki_crop", True, 1))
#     ev = estimator.evaluate(input_fn=lambda: data.input_fn("eval", False, 1))
#     print("Accuracy on test set: ", ev["accuracy"])
#
# res = estimator.predict(input_fn=lambda: data.input_fn("test", False, 1))
# for r in res:
#     print(r["age"])

train = True
if train:
    features, labels = data.input_fn("wiki_crop", True, None)
else:
    features, labels = data.input_fn("test", False, 1)

input = keras.layers.Input(tensor=features)
base_model = keras.applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input)
for layer in base_model.layers:
    layer.trainable = False
pooling = keras.layers.GlobalAveragePooling2D()(base_model.output)
fully_connected = keras.layers.Dense(1024, activation='relu')(pooling)
output = keras.layers.Dense(100, activation='softmax')(fully_connected)
model = keras.models.Model(input, output)
model.compile(
    optimizer=keras.optimizers.SGD(1e-4),
    loss='categorical_crossentropy',
    target_tensors=[labels],
    metrics=['accuracy']
)

model.summary()

if train:
    model.load_weights("inceptionv_weights.h5")
    model.fit(epochs=20, steps_per_epoch=100)
    model.save_weights("inceptionv_weights.h5")
else:
    model.load_weights("inceptionv_weights.h5")
    print(np.argmax(model.predict(x=features, steps=1), 1))




