
''''
after alot of trial and error with this project I've decided to dump it so I can actually get something to work :)

problem: I can't get the input data type right because I'm too dumb and the tutorial I followed was outdated by 6 years

help.
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

# don't get close to this
print()
print()
print()
#######


mnist_training , training_info = tfds.load('mnist', split= 'train', shuffle_files=True, download=False, with_info=True)
mnist_test = tfds.load('mnist', split= 'test', shuffle_files=True, download=False)


mnist_training_df = tfds.as_dataframe(mnist_training)
# mnist_test_df = tfds.as_dataframe(mnist_test)

# tf.convert_to_tensor(mnist_training_df)

# print(mnist_training)
# print()

# mnist_training_numpy = tfds.as_numpy(mnist_training)
# mnist_test_numpy = tfds.as_numpy(mnist_test)

# print(mnist_test_df.head())
# print(training_info.)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = training_info.features["label"].num_classes
batch_size = 100

def make_normal_ds(PrefetchDataset_object):
    return tf.data.Dataset.from_tensor_slices(list(PrefetchDataset_object))


# mnist_training = make_normal_ds(mnist_training)

# print(mnist_training_numpy)
print(list(mnist_training))

# def normalize_img(image):
#     return tf.cast(image, tf.float32) / 255.


# #change the image values to float32 data type
# mnist_training = mnist_training.map(normalize_img, num_parallel_calls = tf.data.AUTOTUNE)


tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder('float')
y = tf.compat.v1.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.compat.v1.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.compat.v1.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.compat.v1.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)  #this is the part that does the training u dumbo
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= prediction,labels=y))

    optimizer = tf.keras.optimizers.Adam().minimize(cost) #learning rate default 0.001

    hm_epochs = 10
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(mnist_training_df) / batch_size)):
                _, c = sess.run([optimizer, cost], feed_dict = {x: x, y: y})
                epoch_loss += c
            print('Epoch', epoch,'completed out of', hm_epochs,'loss:', epoch_loss)




        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy is :', accuracy.eval(mnist_test))



# train_neural_network(mnist_training)

