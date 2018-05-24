import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.simplefilter(action='ignore')
import tensorflow as tf


# load data and normalizing data
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
clf = RandomForestClassifier(max_depth=32, random_state=2018, n_estimators=1500,
                             min_samples_leaf=20, min_samples_split=10, n_jobs=32,
                             oob_score=True)
clf.fit(train_x, train_y)
res = clf.predict(test)

# make submission
print('oob_score is: %f' % (clf.oob_score_))
submission = pd.DataFrame({'ImageId': index, 'Label': res})
submission['ImageId'] = submission['ImageId'].astype(int)
submission.to_csv('../data/submission_RF.csv', index=False, sep=',')
print('submission_RF.csv DONE!')
