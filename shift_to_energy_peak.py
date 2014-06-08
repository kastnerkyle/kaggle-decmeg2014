import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
from path import path
from sklearn.decomposition import PCA
import scipy.signal as sig


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


def apply_notch(X):

    bw = .04
    freq = .4
    b, a = notch(freq, bw)
    return sig.lfilter(b, a, X)


def normalize(X):
    norms = np.sqrt((X ** 2).sum(-1))
    X /= norms[:, :, np.newaxis]


def energy_peak(X, sigma=10):
    energy = (X ** 2).mean(0).mean(0)
    smoothed = gaussian_filter(energy, sigma=sigma)
    return smoothed.argmax()


def shift_to_energy_peak(X, before=20, after=40, sigma=10):

    peak = energy_peak(X, sigma=sigma)
    start = peak - before
    end = peak + after

    return X[:, start:end]


def dim_reduce_sensors(X, n_components=30):

    XX = X.transpose(1, 0, 2).reshape(X.shape[1], -1).T
    pca = PCA(n_components=n_components)
    return pca.inverse_transform(pca.fit_transform(XX
           )).reshape(len(X), X.shape[-1], X.shape[1]).transpose(0, 2, 1)


def dim_reduce_sensors_svd(X, n_components=10):

    XX = X.transpose(1, 0, 2).reshape(X.shape[1], -1)

    U, S, VT = np.linalg.svd(XX, full_matrices=False)
    S[n_components:] = 0.

    XX = U.dot(S[:, np.newaxis] * VT)

    return XX.reshape(X.shape[1], X.shape[0], X.shape[2]).transpose(1, 0, 2)


def project_to_nice_timecourses(X):

    timecourses = np.load("timecourses.npz")["A"]

    timecourses /= np.sqrt((timecourses ** 2).sum(0))
    return X.dot(timecourses)


def remove_worst_trials(X, y, n_remove=10):

    keep_indices = sorted((X ** 2).sum(1).sum(1).argsort()[::-1][n_remove:])

    return X[keep_indices], y[keep_indices]

data_dir = path('data')

train_subject_ids = range(1, 17)
test_subject_ids = range(17, 24)
train_subject_names = ["train_subject%02d.mat" % sid
                       for sid in train_subject_ids]
test_subject_names = ["test_subject%02d.mat" % sid
                      for sid in test_subject_ids]

all_train_data = []
all_train_targets = []
labels = []

for i, subject in enumerate(train_subject_names):

    f = loadmat(data_dir / subject)
    X = f['X'][:, 160:]
    y = f['y'].ravel() * 2 - 1

    X, y = remove_worst_trials(X, y)

    X = apply_notch(X)
    normalize(X)
    # X = dim_reduce_sensors(X, n_components=2)[:, :, 125:250]
    X = X[:, :, 125:250]
    X_cropped = X[:, :, 20:80]  # shift_to_energy_peak(X, before=20, after=40)
    # X_cropped = project_to_nice_timecourses(X)
    all_train_data.append(X_cropped)
    all_train_targets.append(y)
    labels.append([i] * len(X_cropped))

all_train_data = np.concatenate(all_train_data)
all_train_targets = np.concatenate(all_train_targets)
labels = np.concatenate(labels)

from sklearn.cross_validation import cross_val_score, LeaveOneLabelOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

scaler = StandardScaler()
clf = LogisticRegression(C=1e-1, penalty="l2")
# clf = ExtraTreesClassifier(n_estimators=100)

pipeline = Pipeline([('scaler', scaler), ('estimator', clf)])

cv = LeaveOneLabelOut(labels)

# all_train_data = dim_reduce_sensors_svd(all_train_data, n_components=10)

all_scores = []

for i in xrange(all_train_data.shape[-1]):
    scores = cross_val_score(pipeline,
                         all_train_data[:, :, i],
                         all_train_targets,
                         cv=cv,
                         verbose=100)
    all_scores.append(scores)


# scores = cross_val_score(pipeline,
    #                      all_train_data.reshape(len(all_train_data), -1),
    #                      all_train_targets,
    #                      cv=cv,
    #                      verbose=100)
