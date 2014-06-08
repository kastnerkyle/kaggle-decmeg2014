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
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
import scipy.signal as sig
from sktensor import dtensor, cp_als
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
from IPython import embed

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

def create_features(XX, tmin, tmax,
                    sfreq, tmin_original=-0.5,
                    perform_baseline_correction=True,
                    plot_name=""):
    """
    Creation of the feature space.

    - restricting the time window of MEG data to [tmin, tmax]sec.
    - Concatenating the 306 timeseries of each trial in one long
      vector.
    - Normalizing each feature independently (z-scoring).

    - optional: "baseline correction", a data centering concept often
                used in M/EEG, will calculate a mean value per sensor
                from pre-stimulus measurements, and subtract this from
                the relevant measurement. Replaces centering based on
                post-stimulus data

    Returns a feature vector XX,

    """
    print("Applying the desired time window and dropping sensors.")
    lower_limit = 240
    XX = XX[:, lower_limit:, :]
    # instead of post-stimulus centering
    baseline = XX[..., :125].mean(-1)

    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()
    XX /= np.linalg.norm(XX, axis=2)[..., np.newaxis]

    #Assuming 250Hz == fs, 125Hz == fs/2, 50Hz = 50/125 = .4
    #5 Hz bw = 5/125 = .04
    print("Applying notch filter for powerline.")
    bw = .04
    freq = .4
    b, a = notch(freq, bw)
    XX = sig.lfilter(b, a, XX)

    #Assuming 250Hz == fs, 125Hz == fs/2, 50Hz = 10/125 = .08
    #5 Hz bw = 5/125 = .04
    print("Applying filter for alpha wave.")
    bw = .04
    freq = .08
    b, a = notch(freq, bw)
    XX = sig.lfilter(b, a, XX)

    XX -= baseline[..., np.newaxis]

    print("CP-ALS Decomposition.")
    T = dtensor(XX)
    P, fit, itr, exectimes = cp_als(T, 2, init='nvecs')
    #P, fit, itr, exectimes = cp_als(T, 8, init='random')
    proj = P.U[2]
    fproj = np.abs(np.fft.fft(proj, axis=0))[:XX.shape[-1] // 2, :]

    plt.figure()
    plt.plot(proj)
    plt.title(plot_name)

    print("Projecting.")
    XX = np.dot(XX, proj)

    print("New shape is %sx%sx%s" % XX.shape)

    print("2D Reshaping: concatenating all 306 timeseries.")
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    print("Features Normalization.")
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))
    return XX


if __name__ == '__main__':
    print("DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain")
    subjects_train = range(1, 17)
    print("Training on subjects", subjects_train)

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0.0
    tmax = 0.500
    print("Restricting MEG data to the interval [%s, %s] sec." % (tmin, tmax))

    X_train = []
    y_train = []
    X_test = []
    ids_test = []
    label_count = []

    print("Creating the trainset.")
    for n, subject in enumerate(subjects_train):
        filename = 'data/train_subject%02d.mat' % subject
        print("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        print("Dataset summary:")
        print("XX:", XX.shape)
        print("yy:", yy.shape)
        print("sfreq:", sfreq)

        XX = create_features(XX, tmin, tmax, sfreq, plot_name=filename)

        X_train.append(XX)
        y_train.append(yy)
        label_count += [subject] * len(XX)

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    print("Trainset:", X_train.shape)

    print("Creating the testset.")
    subjects_test = range(17, 24)
    for n, subject in enumerate(subjects_test):
        filename = 'data/test_subject%02d.mat' % subject
        print("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        ids = data['Id']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        print("Dataset summary:")
        print("XX:", XX.shape)
        print("ids:", ids.shape)
        print("sfreq:", sfreq)

        XX = create_features(XX, tmin, tmax, sfreq, plot_name=filename)

        X_test.append(XX)
        ids_test.append(ids)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)
    print("Testset:", X_test.shape)

    clf = LogisticRegression(C=.1)
    in_subject = []
    out_subject = []
    lol = LeaveOneLabelOut(label_count)
    embed()

    itr = 0
    for train_index, test_index in lol:
        print("Patient %s" % itr)
        clf.fit(X_train[train_index], y_train[train_index])
        y_pred = clf.predict(X_train[test_index])
        osub = accuracy_score(y_train[test_index], y_pred)
        print("Accuracy on unknown: %0.2f" % osub)

        clf.fit(X_train[train_index], y_train[train_index])
        y_pred = clf.predict(X_train[train_index])
        isub = accuracy_score(y_train[train_index], y_pred)
        print("Accuracy on known: %0.2f" % isub)
        itr += 1

    print("LeaveOneSubjectOut scores.")
    in_scores = np.array(in_subject)
    out_scores = np.array(out_subject)
    print(in_scores)
    print(out_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (in_scores.mean(),
                                           in_scores.std() * 2))
    print("Accuracy: %0.2f (+/- %0.2f)" % (out_scores.mean(),
                                           out_scores.std() * 2))

    print("Training.")
    print(X_train.shape)
    clf.fit(X_train, y_train)

    print("Predicting.")
    y_pred = clf.predict(X_test)

    filename_submission = "submission.csv"
    print("Creating submission file", filename_submission)
    with open(filename_submission, "w") as f:
        f.write("Id,Prediction\n")
        for i in range(len(y_pred)):
            f.write(str(ids_test[i]) + "," + str(y_pred[i]) + "\n")
    print("Done.")
