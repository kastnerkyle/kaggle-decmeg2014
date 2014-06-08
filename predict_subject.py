from scipy.io import loadmat
import numpy as np
from path import path

data_dir = path('data')

train_subject_ids = range(1, 11)
train_subject_names = ["train_subject%02d.mat" % sid
                       for sid in train_subject_ids]

all_data = []
all_targets = []
all_sids = []

for sid, subject in zip(train_subject_ids, train_subject_names):
    print subject
    f = loadmat(data_dir / subject)
    X = f['X'][:, 160:, 125:250]
    y = f['y'].ravel() * 2 - 1

    all_data.append(X)
    all_targets.append(y)
    all_sids.append([sid] * len(y))

all_data = np.concatenate(all_data)
all_targets = np.concatenate(all_targets)
all_sids = np.concatenate(all_sids)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1e-5)

pipeline = Pipeline([('scaler', StandardScaler()),
                     ('clf', lr)])

from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
cv = StratifiedShuffleSplit(all_sids, n_iter=10, test_size=.1)

scores = cross_val_score(pipeline,
                         all_data.reshape(len(all_data), -1),
                         all_sids,
                         cv=cv,
                         verbose=1000)
