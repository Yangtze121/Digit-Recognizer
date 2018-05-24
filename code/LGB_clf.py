import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import warnings
warnings.simplefilter(action='ignore')
import tensorflow as tf


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
merge = preprocessing.scale(merge)
train_x = merge[:len_train, :]
test = merge[len_train:, :]
index = np.zeros(len(test))
for i in range(0, len(test)):
    index[i] = int(i)+1

# train and predict data
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=7)
clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        max_depth=-1, n_estimators=20000, objective='multi',
        learning_rate=0.02, min_child_weight=0.5, random_state=7, n_jobs=-1)
clf.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric='multi_error',early_stopping_rounds=50, verbose=50,)
res = clf.predict(test)

# make submission
submission = pd.DataFrame({'ImageId': index, 'Label': res})
submission['ImageId'] = submission['ImageId'].astype(int)
submission.to_csv('../data/submission_LGB.csv', index=False, sep=',')
print('submission_LGB.csv DONE!')

# plot images
# pix1 = train_x[4 ,:, :].reshape(28, 28)
# fig = plt.figure()
# ax1 = fig.add_subplot(221)
# ax1.imshow(pix1, cmap=plt.cm.gray_r)
# plt.show()