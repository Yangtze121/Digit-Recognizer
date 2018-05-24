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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # padding: 'SAME' or 'VALID'


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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
x_image = tf.reshape(xs, [-1, 28, 28, 1])   # 最后1位为颜色channel， 黑白为1

# conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32])   # patch 5x5, in size 1(黑白), out size 32(卷积后的图片厚度)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 14*14*32

# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])   # patch 5x5, in size 32, out size 64(卷积后的图片厚度)
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                          # output size 7*7*64

# func1 layer
W_f1 = weight_variable([7*7*64, 1024])
b_f1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_f1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_f1) + b_f1)
h_f1_drop = tf.nn.dropout(h_f1, keep_prob)

# func2 layer
W_f2 = weight_variable([1024, 10])
b_f2 = bias_variable([10])
pred = tf.nn.softmax(tf.matmul(h_f1_drop, W_f2) + b_f2)

# loss and train_step
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(pred), axis=1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

index = np.zeros(len(test))
for i in range(0, len(test)):
    index[i] = int(i) + 1
# train
early_accuracy = -1
epslon = -1e20
batch_size = 100
n_batch = int(len(train_x) / batch_size)
t1 = time.time()
for epoch in range(30):
    print('epoch', epoch + 1)
    for batch in range(n_batch):
        batch_x = train_x[batch * batch_size: (batch+1) * batch_size]
        batch_y = train_y[batch * batch_size: (batch+1) * batch_size]
        sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
        if batch % 20 == 0:
            now_accuracy = pred_accuracy(val_x, val_y)
            print('batch: [%d]   train accuracy is: %f  val accuracy is: %f' % (batch, pred_accuracy(train_x, train_y), now_accuracy))
        if batch % 60 == 0:
            # make submission
            t2 = time.time()
            print('use time: %d s' % (t2 - t1))
            test_y = sess.run(pred, feed_dict={xs: test, keep_prob: 1.0})
            res = sess.run(tf.argmax(test_y, 1)).reshape(-1)
            submission = pd.DataFrame({'ImageId': index, 'Label': res})
            submission['ImageId'] = submission['ImageId'].astype(int)
            save_dir = '../submission/cnn/submission_cnn_epoch_%d_batch_%d.csv' % (epoch + 1, batch)
            submission.to_csv(save_dir, index=False, sep=',')
            print(save_dir, 'DONE!')



