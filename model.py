import tensorflow as tf


def model_fn(features, labels, mode, params):
    x = features

    # 256x256x3->128x128x64
    x = tf.layers.conv2d(x, 64, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 64, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    #x = tf.layers.dropout(x, rate=0.5)

    # 128x128x64->64x64x128
    x = tf.layers.conv2d(x, 128, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 128, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    # x = tf.layers.dropout(x, rate=0.5)

    # 64x64x128->32x32x256
    x = tf.layers.conv2d(x, 256, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 256, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    #x = tf.layers.dropout(x, rate=0.5)

    # 32x32x256->16x16x512
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    #x = tf.layers.dropout(x, rate=0.5)

    # 16x16x512->8x8x512
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.conv2d(x, 512, kernel_size=[3, 3], padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    #x = tf.layers.dropout(x, rate=0.5)

    x = tf.reshape(x, shape=[-1, 2 * 2 * 512])

    # dense layer w/ 2048 elements
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)

    # dense layer w/ 2048 elements
    x = tf.layers.dense(x, units=4096, activation=tf.nn.relu)

    # regression layer
    output = tf.layers.dense(x, units=100, activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'age': tf.argmax(output, axis=1),
                'probability': tf.nn.softmax(output)
            })

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(predictions=tf.argmax(output, axis=1), labels=tf.argmax(labels, axis=1))
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            loss=loss,
            mode=mode,
            predictions={
                'age': tf.argmax(output, axis=1),
                'probability': tf.nn.softmax(output)
            },
            eval_metric_ops=eval_metric_ops
        )

    # loss function & optimizer
    train_op = tf.train.AdamOptimizer(params['learning_rate'])\
        .minimize(loss, global_step=tf.train.get_global_step())

    # prediction and accuracy
    batch_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(labels, axis=1)), dtype=tf.float32)
    )

    logging_hook = tf.train.LoggingTensorHook({"batch_accuracy": batch_accuracy},
                                              every_n_iter=100)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'age': tf.argmax(output, axis=1),
            'probability': tf.nn.softmax(output)
        },
        loss=loss,
        train_op=train_op,
        training_hooks=[logging_hook]
    )


