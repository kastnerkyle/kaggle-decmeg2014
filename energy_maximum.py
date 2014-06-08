import numpy as np
from scipy.io import loadmat
import mne
import scipy

from path import path

train_subject_ids = range(1, 17)
train_subject_names = ["train_subject%02d.mat" % sid
                       for sid in train_subject_ids]
test_subject_ids = range(17, 24)
test_subject_names = ["test_subject%02d.mat" % sid
                      for sid in test_subject_ids]

subject_names = train_subject_names + test_subject_names

data_dir = path('data')

import matplotlib.pyplot as plt

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

for i, subject in enumerate(subject_names):
    print i

    f = loadmat(data_dir / subject)
    X = f['X']
    # y = f['y'].ravel() * 2 - 1

    X = X[:, 160:, 125:250]

    X /= np.sqrt((X ** 2).sum(-1))[..., np.newaxis]

    energy = (X ** 2).mean(0)

    plt.figure(fig1.number)
    plt.subplot(3, 8, i + 1)
    plt.imshow(energy)
    plt.xlabel("sub%02d" % (i + 1))

    plt.figure(fig2.number)
    plt.subplot(3, 8, i + 1)
    plt.plot(np.arange(0, 0.5, 0.004), np.linalg.svd(energy)[2][:3].T)
    plt.xlabel("sub%02d" % (i + 1))

    plt.figure(fig3.number)
    plt.subplot(3, 8, i + 1)
    plt.plot(np.arange(0, 0.5, 0.004), energy.mean(0), 'k', lw=2)
    plt.plot(np.arange(0, 0.5, 0.004),
             scipy.ndimage.gaussian_filter(energy.mean(0), sigma=10), 'b')
    plt.xlabel("sub%02d" % (i + 1))


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=5e-1, penalty="l2")
# from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit


fig4 = plt.figure()

for i, subject in enumerate(train_subject_names):
    print i

    f = loadmat(data_dir / subject)
    X = f['X']
    y = f['y'].ravel() * 2 - 1

    X = X[:, 160:, 125:250]

    X /= np.sqrt((X ** 2).sum(-1))[..., np.newaxis]

    weights = lr.fit(X.reshape(len(X), -1), y).coef_.reshape(X.shape[1:])
    plt.subplot(3, 8, i + 1)
    plt.plot((weights ** 2).sum(0))

plt.show()

