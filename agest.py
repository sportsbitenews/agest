import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from six.moves import cPickle as pickle


image_size = 64
image_maxval = 255.0

def pickle_classes(folder, pickle_folder, force=False):
    subfolders = os.listdir(folder)
    num_per_class = []

    for cls in range(1, 101):
        pickle_name = os.path.join(pickle_folder, str(cls) + '.pickle')
        if os.path.exists(pickle_name) and not force:
            print(pickle_name, 'file exists, skipping')
            continue

        my_files = []
        for s in subfolders:
            current_path = os.path.join(folder, s)
            if not os.path.isdir(current_path):
                continue

            image_files = os.listdir(current_path)

            for i in image_files:
                image_name = os.path.splitext(i)[0]
                dates = image_name.split('_')
                birth_date, year_taken = dates[1], dates[-1]
                year_birth, _, _ = birth_date.split('-')
                age = int(year_taken) - int(year_birth)
                if 1 <= age <= 101 and age == cls:
                    my_files.append(os.path.join(current_path, i))

        dataset = np.ndarray(shape=(len(my_files), 64, 64, 3), dtype=np.float32)

        num_images_read = 0
        for i in my_files:
            try:
                img = ndimage.imread(i, mode='RGB').astype(float)

                if img.shape[0] <= 1 or img.shape[1] <= 1:
                    continue

                img = (scipy.misc.imresize(img, (image_size, image_size, 3), interp='lanczos'))
                dataset[num_images_read, :, :, :] = (img - 127.0) / 255.0
                num_images_read = num_images_read + 1
            except ImportError as e:
                print('Cannot read file:', i, ', skipping')
                continue

        dataset = dataset[0:num_images_read, :, :, :]
        num_per_class.append(num_images_read)

        with open(pickle_name, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        print('Class pickled', cls, 'num files', dataset.shape[0])

    return num_per_class


def check_pickled_data(folder):
    ages = [10, 20, 40, 80]
    plt.figure()
    for i in range(4):
        pickle_name = os.path.join(folder, str(ages[i]) + '.pickle')

        if not os.path.exists(pickle_name):
            print('Pickle check failed: ', pickle_name, 'does not exist')
            return False

        with open(pickle_name, 'rb') as f:
            dataset = pickle.load(f)

        for j in range(4):
            idx = np.random.randint(dataset.shape[0])
            plt.subplot(4, 4, i * 4 + j + 1)
            plt.imshow(dataset[idx] + 0.5)

    plt.show()
    return True


def show_hist(folder):
    hist = []
    for i in range(1, 101):
        pickle_name = os.path.join(folder, str(i) + '.pickle')
        with open(pickle_name, 'rb') as f:
            dataset = pickle.load(f)
        hist.append(dataset.shape[0])

    plt.bar(range(1, 101), hist)
    plt.show()


def merge_datasets(folder, num_per_class):

    dataset = np.ndarray(shape=(num_per_class * 100, image_size, image_size, 3), dtype=np.float32)
    labels = np.ndarray(shape=(num_per_class * 100, 100), dtype=np.float32)

    for cls in range(100):
        print('Merging class', cls)
        pickle_name = str(cls + 1) + '.pickle'

        with open(os.path.join(folder, pickle_name), 'rb') as f:
            data = pickle.load(f)

        start_idx = cls * num_per_class
        end_idx = start_idx + num_per_class

        if data.shape[0] >= num_per_class:
            dataset[start_idx:end_idx, :, :, :] = data[0:num_per_class, :, :, :]
        else:
            elem_per_step = data.shape[0]
            steps = num_per_class // data.shape[0]
            rest = num_per_class % data.shape[0]

            print('Not enough elements in the class, repeating', steps, 'times')

            for j in range(steps):
                b = (start_idx + j * elem_per_step)
                e = b + elem_per_step
                dataset[b:e, :, :, :] = data[0:elem_per_step, :, :, :]

            b = (start_idx + steps * elem_per_step)
            e = b + rest
            dataset[b:e, :, :, :] = data[0:rest, :, :, :]

        label = np.zeros(100, dtype=np.float32)
        label[cls] = 1.0
        labels[start_idx:end_idx, :] = label

    print('Permuting dataset...')
    permutation = np.random.permutation(dataset.shape[0])
    dataset = dataset[permutation, :, :, :]
    labels = labels[permutation, :]

    v = int(0.8 * dataset.shape[0])
    t = int(0.9 * dataset.shape[0])

    return dataset[:v, :, :, :], labels[:v, :],\
        dataset[v:t, :, :, :], labels[v:t, :],\
        dataset[t:, :, :, :], labels[t:, :]


def check_dataset(sets, labels):
    plt.figure()
    for i in range(4):
        for j in range(4):
            idx = np.random.randint(sets[i].shape[0])
            a = plt.subplot(4, 4, i * 4 + j + 1)
            label = np.argmax(labels[i][idx]) + 1
            a.set_title(str(label))
            plt.imshow(sets[i][idx] + 0.5)
    plt.tight_layout()
    plt.show()


def save_dataset(dataset, labels, name, max_records_per_file=20000):
    num_files = int(math.ceil(dataset.shape[0] / max_records_per_file))
    for i in range(num_files):
        start = i * max_records_per_file
        end = min(dataset.shape[0], start + max_records_per_file)
        with open(name + '_' + str(i) + '.pickle', 'wb') as f:
            pickle.dump((dataset[start:end, :, :, :], labels[start:end, :]), f, pickle.HIGHEST_PROTOCOL)


def conv2d(x, filter_shape):
    weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    return tf.nn.conv2d(input=x, filter=weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_graph(num_labels):
    # input
    x0 = tf.placeholder(shape=[None, image_size, image_size, 3], dtype=tf.float32)
    y_ = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)

    # 64x64x3->32x32x8
    b0 = tf.Variable(tf.constant(0.1, shape=[8]))
    x1 = max_pool2x2(tf.nn.relu(conv2d(x0, [5, 5, 3, 8]) + b0))

    # 32x32x8->16x16x32
    b1 = tf.Variable(tf.constant(0.1, shape=[32]))
    x2 = max_pool2x2(tf.nn.relu(conv2d(x1, [5, 5, 8, 32]) + b1))

    # 16x16x32->8x8x64
    b2 = tf.Variable(tf.constant(0.1, shape=[64]))
    x3 = max_pool2x2(tf.nn.relu(conv2d(x2, [5, 5, 32, 64]) + b2))

    # reshaped input for dense layer
    x3_ = tf.reshape(x3, shape=[-1, 8 * 8 * 64])

    # dense layer w/ 1024 elements
    w3 = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[8 * 8 * 64, 1024]))
    b3 = tf.Variable(tf.constant(0.1, shape=[1024]))
    x4 = tf.nn.relu(tf.matmul(x3_, w3) + b3)

    # dense layer w/ 2048 elements
    w4 = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[1024, 2048]))
    b4 = tf.Variable(tf.constant(0.1, shape=[2048]))
    x5 = tf.nn.relu(tf.matmul(x4, w4) + b4)

    # readout layer
    w5 = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[2048, num_labels]))
    b5 = tf.Variable(tf.constant(0.1, shape=[num_labels]))
    y = tf.matmul(x5, w5) + b5

    return tf.nn.softmax(y), y, x0, y_


check_pickled = False
check_sets = True
regenerate = False
pickle_folder = 'pickle'
dataset_folder = 'wiki_crop'

if not os.path.exists('data_train_0.pickle') or regenerate:
    hist = pickle_classes(dataset_folder, pickle_folder)

    if check_pickled:
        check_pickled_data(pickle_folder)
        show_hist(pickle_folder)

    train_set, train_labels,\
        valid_set, valid_labels,\
        test_set, test_labels = merge_datasets(pickle_folder, 1000)

    print('Merge completed')
    save_dataset(train_set, train_labels, "data_train")
    save_dataset(valid_set, valid_labels, "data_valid")
    save_dataset(test_set, test_labels, "data_test")

else:
    with open('data_train_1.pickle', 'rb') as f:
        train_set, train_labels = pickle.load(f)
    with open('data_valid_0.pickle', 'rb') as f:
        valid_set, valid_labels = pickle.load(f)
    with open('data_test_0.pickle', 'rb') as f:
        test_set, test_labels = pickle.load(f)

print('Train set: ', train_set.shape)
print('Train labels: ', train_labels.shape)
print('Validation set: ', valid_set.shape)
print('Validation labels: ', valid_labels.shape)
print('Test set: ', test_set.shape)
print('Test labels: ', test_labels.shape)

if check_sets:
    check_dataset([train_set, train_set, valid_set, test_set],
                  [train_labels, train_labels, valid_labels, test_labels])


# Prepare the model
pred, y, x0, y_ = create_graph(100)

# loss function & optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# prediction and accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_epochs = 100
    for epoch in range(num_epochs):
        print('Epoch(', epoch, '/', num_epochs,')')
        # train_dataset, train_labels = permute_dataset(train_dataset, train_labels)
        for i in range(train_set.shape[0] // batch_size):
            batch_offset = i * batch_size
            batch_x = train_set[batch_offset:batch_offset + batch_size, :, :, :]
            batch_y = train_labels[batch_offset:batch_offset + batch_size, :]
            sess.run(train_step, feed_dict={x0: batch_x, y_: batch_y})

            if i % 500 == 0:
                print('Batch loss: ', sess.run(loss, feed_dict={x0: batch_x, y_: batch_y}))

                print('Accuracy on validation set: ', 100 *
                      sess.run(accuracy, feed_dict={x0: valid_set, y_: valid_labels}), '%')

        print('Accuracy on test set: ', 100 * sess.run(accuracy, feed_dict={x0: test_set, y_: test_labels}), '%')

