"""Utilities for data preprocessing"""

from path import path
from scipy.io import loadmat
import numpy as np
from math import sqrt
from IPython import embed

TRIAL_NORM_REJECTION_THRESHOLD = sqrt(1e-21)
TRIAL_REJECTION_THRESHOLD_AFTER_NORMALIZING = 2.5

# subject_index, channel, channel_type, watch out subject_index off by 1
# the actual channel is channel * 3 + channel_type
REJECTED_CHANNELS = ([(16, 85, 0),
                     (18, 51, 0),
                     (18, 51, 1),
                     (18, 51, 2)]  +  # all by hand, the rest by thresholdin

                     [(5, 0, 2),
                      (5, 1, 2),
                      (8, 64, 2),
                      (12, 28, 2),
                      (12, 29, 2),
                      (14, 28, 2),
                      (15, 78, 2),
                      (15, 79, 2),
                      (15, 80, 2),
                      (18, 49, 0),
                      (18, 60, 1),
                      (19, 28, 2),
                      (19, 29, 2),
                      (20, 6, 0),
                      (20, 49, 0),
                      (20, 76, 0),
                      (22, 42, 2),
                      (22, 84, 2)] +

                     [(2, 82, 2),
                      (2, 83, 2),
                      (3, 77, 1),
                      (18, 99, 1),
                      (20, 76, 1),
                      (20, 78, 0),
                      (21, 76, 2),
                      (22, 88, 2)]
                     )


def reject_channels(X, rejected_channels=REJECTED_CHANNELS):
    """We reject channels by setting their time courses to 0.
    This way we can preserve shape.
    Kicks out channel for any array you pass it"""

    Y = X.view()
    Y.shape = (len(X), 102, 3, -1)

    for _, c, t in rejected_channels:
        Y[:, c, t, :] = 0.

    return X


def crop_channels(X, crop_slice=slice(159, None)):
    return X[:, crop_slice, :]


def crop_time(X, time_slice=slice(125, 250)):
    return X[:, :, time_slice]


def reject_trials_norm(X, threshold=TRIAL_NORM_REJECTION_THRESHOLD):
    """Reject trials using norm. Make sure to apply this before cropping
    and other preprocessing"""

    squared_norms = (X ** 2).mean(-1).mean(-1)

    norms = np.sqrt(squared_norms)
    keep_trials = norms < threshold

    return X[keep_trials], keep_trials


def impute_trials_norm(X, threshold=TRIAL_NORM_REJECTION_THRESHOLD,
                       also_impute=None,
                       impute_random=42):
    """Replaces trials that pass threshold with the mean of the others.
    Used for the test data.
    impute_random will impute randomly chose trials, otherwise the mean
    """

    squared_norms = (X ** 2).mean(-1).mean(-1)

    norms = np.sqrt(squared_norms)
    keep_trials = norms < threshold
    if also_impute is not None:
        keep_trials = keep_trials & ~also_impute

    kept_trial_mean = X[keep_trials].mean(0)

    if not impute_random:
        X[~keep_trials] = kept_trial_mean[np.newaxis]
    else:
        rng = np.random.RandomState(impute_random)
        num_trials_to_impute = (~keep_trials).sum()
        random_indices = rng.randint(0, keep_trials.sum(),
                                     num_trials_to_impute)
        X[~keep_trials] = X[keep_trials][random_indices]

    return X, keep_trials


def subtract_sensor_specific_subject_mean(X, ):

    Y = X.view()
    Y.shape = (len(X), -1, 3, X.shape[-1])

    grad_mean = Y[:, :, :2, :].mean()
    mag_mean = Y[:, :, 2, :].mean()

    Y[:, :, :2, :] -= grad_mean
    Y[:, :, 2, :] -= mag_mean

    return X


def divide_by_sensor_specific_subject_std(X):
    Y = X.view()
    Y.shape = (len(X), -1, 3, X.shape[-1])

    grad_std = Y[:, :, :2, :].std()
    mag_std = Y[:, :, 2, :].std()

    Y[:, :, :2, :] /= grad_std
    Y[:, :, 2, :] /= mag_std

    return X


def _load_subjects(train_test, ids=None, preproc=None, concatenated=True):
    """load train subjects corresponding to given ids, or all.
    Apply preproc if provided."""

    if preproc is None:
        if train_test == 'train':
            preproc = lambda x, y: (x, y)
        else:
            preproc = lambda x: (x, None)

    subject_names = ["%s_subject%02d.mat" % (train_test, sid) for sid in ids]
    data_dir = path('data')

    all_data = []
    if train_test == 'train':
        all_targets = []
    all_labels = []

    for sid, subject in zip(ids, subject_names):
        print(subject)
        f = loadmat(data_dir / subject)
        if train_test == 'train':
            y = f['y'].ravel()
            X, y = preproc(f['X'], y)
        else:
            X, unimputed = preproc(f['X'])
        labels = [sid] * len(X)

        all_data.append(X)
        if train_test == 'train':
            all_targets.append(y)
        all_labels.append(labels)

    if concatenated:
        all_data = np.concatenate(all_data)
        if train_test == 'train':
            all_targets = np.concatenate(all_targets)
        all_labels = np.concatenate(all_labels)

    if train_test == 'train':
        return all_data, all_targets, all_labels
    else:
        return all_data, all_labels


def preprocessing_train(X, y, normalize_trials=False):
    X = reject_channels(X)
    X, keep = reject_trials_norm(X)
    y = y[keep]
    keep_indices = np.where(keep)[0]
    X = crop_channels(X)
    X = crop_time(X)
    X = subtract_sensor_specific_subject_mean(X)
    X = divide_by_sensor_specific_subject_std(X)
    X, keep2 = reject_trials_norm(
        X, TRIAL_REJECTION_THRESHOLD_AFTER_NORMALIZING)
    keep_indices = keep_indices[keep2]
    y = y[keep2]
    X = subtract_sensor_specific_subject_mean(X)
    X = divide_by_sensor_specific_subject_std(X)

    return X, y


def preprocessing_test(X, normalize_trials=False):
    X = reject_channels(X)
    X, keep = impute_trials_norm(X)
    X = crop_channels(X)
    X = crop_time(X)
    X = subtract_sensor_specific_subject_mean(X)
    X = divide_by_sensor_specific_subject_std(X)
    X, keep2 = impute_trials_norm(
        X, TRIAL_REJECTION_THRESHOLD_AFTER_NORMALIZING,
        also_impute=~keep)
    keep_indices = np.where(keep2)[0]
    X = subtract_sensor_specific_subject_mean(X)
    X = divide_by_sensor_specific_subject_std(X)

    return X, keep_indices


def load_train_subjects(ids=None, preproc=preprocessing_train,
                        concatenated=True):
    if ids is None:
        ids = range(1, 17)

    return _load_subjects("train", ids, preproc, concatenated)


def load_test_subjects(ids=None, preproc=preprocessing_test,
                       concatenated=True):
    if ids is None:
        ids = range(17, 24)

    return _load_subjects("test", ids, preproc, concatenated)



def _calibrate_reject_trials():
    all_norms = []
    all_channel_norms = []
    for train_id in range(1, 17):
        print(train_id)
        X, _, _ = load_train_subjects(ids=[train_id])
        X = reject_channels(X)
        X, keep = reject_trials_norm(X)
        X = crop_channels(X)
        X = crop_time(X)
        X = subtract_sensor_specific_subject_mean(X)
        X = divide_by_sensor_specific_subject_std(X)
        X, keep2 = reject_trials_norm(X,
                               TRIAL_REJECTION_THRESHOLD_AFTER_NORMALIZING)
        X = subtract_sensor_specific_subject_mean(X)
        X = divide_by_sensor_specific_subject_std(X)
        norms_squared = (X.squeeze() ** 2).mean(-1).mean(-1)
        norms = np.sqrt(norms_squared)
        channel_norms = (X ** 2).mean(0).mean(-1)
        all_channel_norms.append(channel_norms)
        all_norms.append(norms)

    for test_id in range(17, 24):
        print(test_id)
        X, _ = load_test_subjects(ids=[test_id])
        X = reject_channels(X)
        X, kept = impute_trials_norm(X)
        X = crop_channels(X)
        X = crop_time(X)
        X = subtract_sensor_specific_subject_mean(X)
        X = divide_by_sensor_specific_subject_std(X)
        X, kept_again = impute_trials_norm(X,
                               TRIAL_REJECTION_THRESHOLD_AFTER_NORMALIZING,
                               also_impute=~kept)
        X = subtract_sensor_specific_subject_mean(X)
        X = divide_by_sensor_specific_subject_std(X)
        norms_squared = (X.squeeze() ** 2).mean(-1).mean(-1)
        norms = np.sqrt(norms_squared)
        all_norms.append(norms)
        channel_norms = (X ** 2).mean(0).mean(-1)
        all_channel_norms.append(channel_norms)
        all_norms.append(norms)

    import pylab as pl
    pl.figure()
    pl.plot(np.concatenate(all_norms))
    pl.yscale('log')
    pl.figure()
    pl.plot(np.array(all_channel_norms).reshape(-1, 3))
    return all_norms, all_channel_norms


if __name__ == "__main__":
    # X_train, y_train, train_labels = load_train_subjects()
    X_test, test_labels = load_test_subjects()
