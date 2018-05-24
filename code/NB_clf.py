import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn import naive_bayes
import time
import warnings
warnings.simplefilter(action='ignore')


# load data and normalizing data
# df_train = pd.read_csv('../data/train.csv').as_matrix()
# df_test = pd.read_csv('../data/test.csv').as_matrix()
# train_x = df_train[:, 1:].reshape(-1, 784)
# test = df_test.reshape(-1, 784)
# train_y = df_train[:, :1].reshape(-1, 1)
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
train_y = df_train['label'].as_matrix().reshape(-1, 1)
del df_train['label']
len_train = len(df_train)
merge = pd.concat([df_train, df_test], axis=0).as_matrix().reshape(-1, 784)
merge = preprocessing.MinMaxScaler().fit_transform(merge)            # must use minmaxscaler input must non-negative
train_x = merge[:len_train, :]
test = merge[len_train:, :]
index = np.zeros(len(test))
for i in range(0, len(test)):
    index[i] = int(i)+1

# train and predict data
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=7)
t1 = time.time()
clf = naive_bayes.BernoulliNB()
clf.fit(train_x, train_y)
print('val accuracy is:%f' % (clf.score(val_x, val_y)))
print('time usage: %d' % (time.time() - t1))
res = clf.predict(test)

# make submission
submission = pd.DataFrame({'ImageId': index, 'Label': res})
submission['ImageId'] = submission['ImageId'].astype(int)
submission.to_csv('../data/submission_NB.csv', index=False, sep=',')
print('submission_NB.csv DONE!')
