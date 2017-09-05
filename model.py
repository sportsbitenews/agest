import tensorflow as tf


def model_fn(features, labels, mode, params):
    x = features['x']

    # 224x224x3->112x112x64
    x = tf.layers.conv2d(x, 64, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 64, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.4)

    # 112x112x64->56x56x128
    x = tf.layers.conv2d(x, 128, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 128, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.4)

    # 56x56->128->28x28x256
    x = tf.layers.conv2d(x, 256, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 256, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.4)

    # 28x28x256->14x14x512
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.2)

    # 14x14x512->7x7x512
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.dropout(x, rate=0.2)

    x = tf.reshape(x, shape=[-1, 2 * 2 * 512])

    # dense layer w/ 4096 elements
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)

    # dense layer w/ 4096 elements
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)

    # readout layer
    logits = tf.layers.dense(x, units=100)

    # predictions
    predictions = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'ages': predictions})

    # loss function & optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(params['learning_rate'])\
        .minimize(loss, global_step=tf.train.get_global_step())

    # prediction and accuracy
    # correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    # accuracy, accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={'ages': predictions},
        loss=loss,
        train_op=train_op,
        eval_metric_ops={'accuracy': tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(predictions, 1))})


