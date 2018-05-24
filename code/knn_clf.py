import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.simplefilter(action='ignore')


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
clf = KNeighborsClassifier(n_neighbors=5, n_jobs=30)
clf.fit(train_x, train_y)
score = clf.score(val_x, val_y)
print('val accuracy is: %f' % score)
res = clf.predict(test)

# make submission
submission = pd.DataFrame({'ImageId': index, 'Label': res})
submission['ImageId'] = submission['ImageId'].astype(int)
submission.to_csv('../data/submission_knn.csv', index=False, sep=',')
print('submission_knn.csv DONE!')