import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn import preprocessing
import warnings
warnings.simplefilter(action='ignore')
import time


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
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=7)

# train and predict data
t1 = time.time()
clf = svm.SVC(C=1.0, kernel='linear', verbose=50, decision_function_shape='ovr')
clf.fit(train_x, train_y)
res = clf.predict(test)

# make submission
t2 = time.time()
print('time usage:%d' %(t2 - t1))
submission = pd.DataFrame({'ImageId': index, 'Label': res})
submission['ImageId'] = submission['ImageId'].astype(int)
submission.to_csv('../data/submission_SVM.csv', index=False, sep=',')
print('submission_SVM.csv DONE!')