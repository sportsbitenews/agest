import tensorflow as tf


def model_fn(features, labels, mode, params):
    x = features['x']

    # 224x224x3->112x112x64
    x = tf.layers.conv2d(x, 64, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 64, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.5)

    # 112x112x64->56x56x128
    x = tf.layers.conv2d(x, 128, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 128, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.5)

    # 56x56->128->28x28x256
    x = tf.layers.conv2d(x, 256, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 256, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.5)

    # 28x28x256->14x14x512
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.5)

    # 14x14x512->7x7x512
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.5)

    x = tf.reshape(x, shape=[-1, 2 * 2 * 512])

    # dense layer w/ 4096 elements
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)

    # dense layer w/ 4096 elements
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)

    # regression layer
    prediction = tf.layers.dense(x, units=1, activation=None)

    prediction = tf.Print(prediction, [tf.shape(prediction)])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'age': prediction})

    # loss function & optimizer
    loss = tf.losses.mean_squared_error(predictions=prediction, labels=labels)
    train_op = tf.train.AdamOptimizer(params['learning_rate'])\
        .minimize(loss, global_step=tf.train.get_global_step())

    # prediction and accuracy
    # correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    # accuracy, accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={'age': prediction},
        loss=loss,
        train_op=train_op)


