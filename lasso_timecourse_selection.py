import numpy as np
from scipy.io import loadmat
from path import path
from sklearn.externals.joblib import Memory


train_subject_ids = range(1, 17)

train_subject_names = ["train_subject%02d.mat" % sid
                       for sid in train_subject_ids]

data_dir = path("data")

from sklearn.linear_model import LassoLarsCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score

from sktensor import dtensor, cp_als

mem = Memory(cachedir='cache')
cp_als = mem.cache(cp_als)

all_decomps = []
all_weights = []
all_selected_timecourses = []
all_scores = []


for subject in train_subject_names:
    print subject
    f = loadmat(data_dir / subject)
    X = f['X'][:, 160:, 125:250]
    y = f['y'].ravel() * 2 - 1

    cv = StratifiedShuffleSplit(y, n_iter=50, test_size=.1)

    pipeline = Pipeline([('scaler', StandardScaler()),
                     ('estimator', LassoLarsCV(cv=cv))])

    T = dtensor(X)
    r = cp_als(T, rank=10)

    sample_axes = r[0].U[0]

    pipeline.fit(sample_axes, y)
    weights = pipeline.steps[1][1].coef_
    all_weights.append(weights)

    selected = np.where(weights != 0)[0]
    all_selected_timecourses.append(r[0].U[2][:, selected])

    pipeline_global_eval = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', LogisticRegression(C=1., penalty="l1"))])
    global_scores = cross_val_score(pipeline_global_eval,
                                    sample_axes, y, cv=cv,
                                    verbose=1000)
    all_scores.append(global_scores)

