"""
DecMeg2014 example code.

Simple prediction of the class labels of the test set by:
- pooling all the training trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Projecting with RandomProjection
- Using a classifier.

Copyright Emanuele Olivetti 2014, BSD license, 3 clauses.

"""

import numpy as np
from sklearn.linear_model import LassoLarsCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from scipy.io import loadmat
from scipy import signal as sig
from matplotlib import pyplot as plt
from sktensor import dtensor, cp_als
from sklearn.cross_validation import StratifiedShuffleSplit

_LOCAL_MEM = Memory(cachedir='cache')
cp_als = _LOCAL_MEM.cache(cp_als)


def view_filter(b, a):
    w, h = sig.freqz(b, a)
    plt.plot(w / abs(w), np.abs(h))


def notch(Wn, bandwidth):
    """
    Notch filter to kill line-noise.
    """
    f = Wn / 2.0
    R = 1.0 - 3.0 * (bandwidth / 2.0)
    num = 1.0 - 2.0 * R * np.cos(2 * np.pi * f) + R ** 2.
    denom = 2.0 - 2.0 * np.cos(2 * np.pi * f)
    K = num / denom
    b = np.zeros(3)
    a = np.zeros(3)
    a[0] = 1.0
    a[1] = -2.0 * R * np.cos(2 * np.pi * f)
    a[2] = R ** 2.
    b[0] = K
    b[1] = -2.0 * K * np.cos(2 * np.pi * f)
    b[2] = K
    return b, a


def window(XX, lower_limit=160, tmin=0.0, tmax=0.5, sfreq=250, tmin_original=-.5):
    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    print("Restricting MEG data to the interval [%s, %s] sec." % (tmin, tmax))
    XX = XX[:, lower_limit:, :]
    # instead of post-stimulus centering
    print("Apply desired time window and drop sensors 0 to %i." % lower_limit)

    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()
    return XX


def notch_filter(XX):
    # Assuming 250Hz == fs, 125Hz == fs/2, 50Hz = 50/125 = .4
    # 5 Hz bw = 5/125 = .04
    print("Applying notch filter for powerline.")
    bw = .04
    freq = .4
    b, a = notch(freq, bw)
    XX = sig.lfilter(b, a, XX)

    # Assuming 250Hz == fs, 125Hz == fs/2, 50Hz = 10/125 = .08
    # 5 Hz bw = 5/125 = .04
    print("Applying filter for alpha wave.")
    bw = .04
    freq = .08
    b, a = notch(freq, bw)
    XX = sig.lfilter(b, a, XX)
    return XX


def window_filter_baseline(XX, lower_limit=160):
    baseline = XX[:, lower_limit:, :125].mean(-1)
    XX = window(XX)
    XX = notch_filter(XX)
    XX -= baseline[..., np.newaxis]
    return XX


def get_outlier_mask(XX):
    print("Getting outlier mask.")
    mask = (XX ** 2).sum(axis=-1).sum(axis=-1)
    mask = mask.argsort()[10:-10]
    return mask


def load_data():
    all_subjects = range(1, 24)

    X = []
    y = []
    ids_test = []
    label_count = []

    n_basis = 10
    all_U0 = []
    all_U2 = []
    for n, subject in enumerate(all_subjects):
        if subject < 17:
            filename = 'data/train_subject%02d.mat' % subject
        else:
            filename = 'data/test_subject%02d.mat' % subject
        print("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = window_filter_baseline(data['X'])
        mask = get_outlier_mask(XX)
        T = dtensor(XX)
        r = cp_als(T, rank=n_basis)
        U0 = r[0].U[0]
        U1 = r[0].U[1]
        U2 = r[0].U[2]
        X.append(XX)
        all_U0.append(U0)
        all_U2.append(U2)
        if subject < 17:
            yy = data['y'].ravel()
            y.append(yy)
        else:
            ids = data['Id']
            ids_test.append(ids)
        label_count += [subject] * len(XX)

    all_U0 = np.vstack(all_U0)
    all_U2 = np.vstack(all_U2)
    X = np.vstack(X)
    y = np.concatenate(y)

    cv = StratifiedShuffleSplit(yy, n_iter=50, test_size=.1)
    selection_pipe = Pipeline([('scaler', StandardScaler()),
                               ('estimator', LassoLarsCV(cv=cv))])
    selection_pipe.fit(all_U0[:y.shape[0]], y * 2 - 1)
    weights = selection_pipe.steps[1][1].coef_
    selected = np.where(weights != 0)[0]
    proj = all_U2[:, selected].T
    ids_test = np.concatenate(ids_test)
    from IPython import embed; embed()
    return np.dot(X, proj), y, ids_test, label_count

if __name__ == '__main__':
    print("DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain")

    X, y, ids_test, label_count = load_data()

    X_train = X[:len(y)]
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X[len(y):]
    X_test = X_test.reshape(X_test.shape[0], -1)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('estimator', LogisticRegression(C=.01))])
    print("Fitting predictor.")
    pipe.fit(X_train, y)
    y_pred = pipe.predict(X_test)

    filename_submission = "submission.csv"
    print("Creating submission file", filename_submission)
    with open(filename_submission, "w") as f:
        f.write("Id,Prediction\n")
        for i in range(len(y_pred)):
            f.write(str(ids_test[i]) + "," + str(y_pred[i]) + "\n")
    print("Done.")
