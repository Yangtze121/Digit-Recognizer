import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.simplefilter(action='ignore')


def add_layer(inputs, input_size, output_size, keep_prob=1.0, activation_function=None):
    Weights = tf.Variable(tf.random_normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]) + 1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def pred_accuracy(data_x, data_y):
    global pred
    pred_y = sess.run(pred, feed_dict={xs: data_x, keep_prob: 1})
    correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(data_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), axis=0)
    result = sess.run(accuracy)
    return result


df_train = pd.read_csv('../data/train.csv').astype(np.float32).as_matrix()
df_test = pd.read_csv('../data/test.csv').astype(np.float32).as_matrix()
train_x = (df_train[:, 1:] / 255).reshape(-1, 784)                              # very important to normalize!!
test = (df_test.reshape(-1, 784)) / 255
t_y = df_train[:, :1].reshape(-1, 1)

# one-hot
train_y = np.zeros([len(t_y), 10])
for i in range(len(t_y)):
    train_y[i, int(t_y[i])] = 1.0

# split train and validation
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=7)

# define placeholder
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# add layer
h1 = add_layer(xs, 784, 1600, keep_prob, activation_function=tf.nn.softmax)
pred = add_layer(h1, 1600, 10, keep_prob, activation_function=tf.nn.softmax)

# loss and train_step
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(pred), axis=1))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# train_step = tf.train.AdagradOptimizer(0.35).minimize(cross_entropy)
train_step = tf.train.MomentumOptimizer(0.4, 0.4).minimize(cross_entropy)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

index = np.zeros(len(test))
for i in range(0, len(test)):
    index[i] = int(i) + 1
# train
t1 = time.time()
batch_size = 100
n_batch = int(len(train_x) / batch_size)
for epoch in range(100):
    print('epoch', epoch + 1)
    for batch in range(n_batch):
        batch_x = train_x[batch * batch_size: (batch+1) * batch_size]
        batch_y = train_y[batch * batch_size: (batch+1) * batch_size]
        sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
        if batch % 20 == 0:
            now_accuracy = pred_accuracy(val_x, val_y)
            print('batch: [%d]   train accuracy is: %f  val accuracy is: %f' % (batch, pred_accuracy(train_x, train_y), now_accuracy))
        if batch % 100 == 0:
            # make submission
            t2 = time.time()
            print('use time: %d s' % (t2 - t1))
            test_y = sess.run(pred, feed_dict={xs: test, keep_prob: 1.0})
            res = sess.run(tf.argmax(test_y, 1)).reshape(-1)
            submission = pd.DataFrame({'ImageId': index, 'Label': res})
            submission['ImageId'] = submission['ImageId'].astype(int)
            save_dir = '../submission/nn/submission_nn_epoch_%d_batch_%d.csv' % (epoch + 1, batch)
            submission.to_csv(save_dir, index=False, sep=',')
            print(save_dir, 'DONE!')
