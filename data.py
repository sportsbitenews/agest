import os
import numpy as np
import tensorflow as tf


def input_fn(data_folder, shuffle=False, repeat_count=1):
    # contains tuples of the form (filename, age)
    files = []
    labels = []

    for subdir in os.listdir(data_folder):
        full_subdir_path = os.path.join(data_folder, subdir)
        if not os.path.isdir(full_subdir_path):
            continue
        for file in os.listdir(full_subdir_path):
            full_file_path = os.path.join(full_subdir_path, file)
            if not os.path.isfile(full_file_path):
                continue

            _, birth_date, year_taken = os.path.splitext(file)[0].split("_")
            year_birth = birth_date.split("-")[0]
            age = int(year_taken) - int(year_birth)

            if 0 < age <= 100:
                logit = [0.0] * 100
                logit[age-1] = 1.0
                files.append(full_file_path)
                labels.append(logit)

    tf_files = tf.constant(files)
    tf_labels = tf.constant(labels)

    dataset = tf.contrib.data.Dataset.from_tensor_slices((tf_files, tf_labels))

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [256, 256])
        return image_resized, label

    def filter_function(features, label):
        image_mean, image_stddev = tf.nn.moments(features, [0, 1, 2])
        return tf.not_equal(image_stddev, tf.constant(0.0))

    def adjust_function(features, label):
        image_mean, image_stddev = tf.nn.moments(features, [0, 1])
        image_resized = tf.divide(tf.subtract(features, tf.constant(127.0)), tf.constant(255.0))
        return image_resized, label

    dataset = dataset.map(_parse_function).\
        filter(filter_function).map(adjust_function)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count).batch(32)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()





