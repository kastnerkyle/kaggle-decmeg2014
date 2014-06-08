import numpy as np
from path import path
from scipy.io import loadmat


subject_ids = [4]

subject_names = ["train_subject%02d.mat" % sid for sid in subject_ids]
data_dir = path("data")

timecourses = np.load('timecourses.npz')["A"]

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1e-1, penalty="l2"))
        ])

from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit


for subject in subject_names:
    f = loadmat(data_dir / subject)

    X = f['X'][:, 160:, 125:250]
    y = f['y'].ravel() * 2 - 1

    cv = StratifiedShuffleSplit(y, n_iter=20, test_size=.1)

    scores_raw = cross_val_score(pipeline, X.reshape(len(X), -1), y, cv=cv,
                                 verbose=100)

    projected = X.dot(timecourses)

    scores_projected = cross_val_score(
        pipeline, projected.reshape(len(X), -1), y, cv=cv, verbose=100)


