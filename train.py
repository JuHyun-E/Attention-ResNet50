from keras.datasets import cifar10
import numpy as np
from ResNet50 import *
import matplotlib.pyplot as plt
import os


# CIFAR 10
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

classes = 10
learning_rate = 1e-3
epochs = 10000
batch_size = 128
num_batch = int(X_train.shape[0] / batch_size)
print('number of batch: ', num_batch)

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(tf.float32, shape=[None, classes])
keep_prob = tf.placeholder(tf.float32)

Y_train_one_hot = tf.squeeze(tf.one_hot(y_train, classes), axis=1)
Y_test_one_hot = tf.squeeze(tf.one_hot(y_test, classes), axis=1)

y_pred, _ = ResNet50(X, classes)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

# Model save path
MODEL_SAVE_FOLDER_PATH = './model_/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

plt_costs = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    # checkpoint
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_FOLDER_PATH)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model Load")

    for epoch in range(epochs):
        costs = 0
        for b in range(num_batch):
            batch = next_batch(batch_size, X_train, Y_train_one_hot.eval())

            if b % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
                loss_print = loss.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
                print("Epoch: {}/{}, training accuracy: {}, mini batch loss: {}".format(epoch + 1, epochs, train_accuracy, loss_print))

            _, step_loss = sess.run([train, loss], feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.8})
            costs += step_loss / num_batch

        print("Epoch: {}/{}, loss: {}".format(epoch+1, epochs, costs))

        if epoch % 1 == 0:
            plt_costs.append(costs)
            save_path = saver.save(sess, MODEL_SAVE_FOLDER_PATH + 'model%s.ckpt' % (epoch+1))

    plt.plot(plt_costs)
    plt.title('Model loss')
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.show()
