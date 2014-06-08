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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.io import loadmat
import scipy.signal as sig
from sktensor import dtensor, cp_als
from sklearn.cross_validation import LeavePLabelOut
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os


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


def window_baseline(XX, lower_limit=160):
    baseline = XX[:, lower_limit:, :125].mean(-1)
    XX = window(XX)
    print("Baseline.")
    XX -= baseline[..., np.newaxis]
    return XX


def window_filter(XX):
    XX = window(XX)
    XX = notch_filter(XX)
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


def get_tensor_decomposition(XX, n=2):
    print("CP-ALS Decomposition.")
    T = dtensor(XX)
    P, fit, itr, exectimes = cp_als(T, n, init='nvecs')
    proj = P.U
    return proj


def load_train_data(exclude_subject=16):
    subjects_train = [i for i in range(1, 17) if i != exclude_subject]
    print("Loading subjects", subjects_train)
    X_train = []
    y_train = []
    label_count = []

    print("Creating the trainset.")
    for n, subject in enumerate(subjects_train):
        filename = 'data/train_subject%02d.mat' % subject
        print("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        XX = window_filter_baseline(XX)
        X_train.append(XX)
        y_train.append(yy)
        label_count += [subject] * len(XX)

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    print("Trainset:", X_train.shape)
    return X_train, y_train, label_count


def load_val_data(subject=16):
    subjects_val = [subject]
    print("Loading subjects", subjects_val)
    X_val = []
    y_val = []
    label_count = []

    print("Creating the validation set.")
    for n, subject in enumerate(subjects_val):
        filename = 'data/train_subject%02d.mat' % subject
        print("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        XX = window_filter_baseline(XX)
        X_val.append(XX)
        y_val.append(yy)
        label_count += [subject] * len(XX)

    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    print("Validation set:", X_val.shape)
    return X_val, y_val, label_count


def load_test_data():
    subjects_test = range(17, 24)
    print("Loading subjects", subjects_test)
    X_test = []
    ids_test = []
    label_count = []

    print("Creating the testset.")
    for n, subject in enumerate(subjects_test):
        filename = 'data/test_subject%02d.mat' % subject
        print("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        ids = data['Id']
        XX = window_filter_baseline(XX)
        X_test.append(XX)
        ids_test.append(ids)
        label_count += [subject] * len(XX)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)
    print("Testset:", X_test.shape)
    return X_test, ids_test, label_count


def get_data(val_index=16):
    if val_index > 16:
        raise ValueError("There are only 16 training patients!")

    saved_data_path = "saved_data_val_%i.npz" % val_index
    if not os.path.exists(saved_data_path):
        print("Saved, preprocessed data not found in %s" % saved_data_path)
        X_train, y_train, label_count_train = load_train_data()
        X_test, ids_test, label_count_test = load_test_data()
        X_val, y_val, label_count_val = load_val_data()
        np.savez(saved_data_path,
                 X_train=X_train,
                 y_train=y_train,
                 X_val=X_val,
                 y_val=y_val,
                 X_test=X_test,
                 ids_test=ids_test,
                 label_count_train=label_count_train,
                 label_count_test=label_count_test,
                 label_count_val=label_count_val)
    else:
        print("Saved, preprocessed data found in %s" % saved_data_path)
        npzfile = np.load(saved_data_path)
        X_train = npzfile['X_train']
        y_train = npzfile['y_train']
        X_val = npzfile['X_val']
        y_val = npzfile['y_val']
        X_test = npzfile['X_test']
        ids_test = npzfile['ids_test']
        label_count_train = npzfile['label_count_train']
        label_count_test = npzfile['label_count_test']
        label_count_val = npzfile['label_count_val']

    return (X_train, y_train, label_count_train, X_test, ids_test,
            label_count_test, X_val, y_val, label_count_val)


def project_against_timeseries_tensors(X_train, X_test, X_val, label_count_train,
                                       label_count_test, label_count_val):
    n_basis = 75
    saved_proj = "saved_time_projs_%s.npz" % n_basis

    if not os.path.exists(saved_proj):
        X_full = np.vstack((X_train, X_test, X_val))
        label_count_full = np.concatenate((label_count_train, label_count_test,
                                           label_count_val))
        print("Saved time projection file not found in %s" % saved_proj)
        print("Creating projections")
        lol = LeavePLabelOut(label_count_full, p=1)
        proj = []
        for n, (train_index, test_index) in enumerate(lol):
            print("Getting dictionary for patient %s" % n)
            trial_proj, sensor_proj, time_proj = get_tensor_decomposition(
                X_full[test_index], n_basis)
            proj.append(time_proj)
        proj = np.array(proj)
        np.savez(saved_proj, proj=proj)
    else:
        print("Saved projection files found in %s" % saved_proj)
        npzfile = np.load(saved_proj)
        proj = npzfile['proj']

    proj = np.max(proj, axis=-1)
    X_train = np.dot(X_train, proj.T)
    X_test = np.dot(X_test, proj.T)
    X_val = np.dot(X_val, proj.T)
    print("Shape of reduced train data %i x %i x %i" % X_train.shape)
    print("Shape of reduced test data %i x %i x %i" % X_test.shape)
    print("Shape of reduced val data %i x %i x %i" % X_val.shape)
    return X_train, X_test, X_val


if __name__ == '__main__':
    print("DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain")

    validation_subject = 16
    (X_train, y_train, label_count_train,
     X_test, ids_test, label_count_test,
     X_val, y_val, label_count_val) = get_data(val_index=validation_subject)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=.1, penalty='l2'))])

    X_train, X_test, X_val = project_against_timeseries_tensors(
        X_train, X_test, X_val, label_count_train, label_count_test,
        label_count_val)
    print("Projection complete.")
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    print("Training.")
    pipe.fit(X_train, y_train)

    print("Predicting validation subject.")
    y_val_pred = pipe.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    print("Accuracy on validation subject %s" % acc)

    print("Predicting test set.")
    y_pred = pipe.predict(X_test)

    filename_submission = "submission.csv"
    print("Creating submission file", filename_submission)
    with open(filename_submission, "w") as f:
        f.write("Id,Prediction\n")
        for i in range(len(y_pred)):
            f.write(str(ids_test[i]) + "," + str(y_pred[i]) + "\n")
    print("Done.")
