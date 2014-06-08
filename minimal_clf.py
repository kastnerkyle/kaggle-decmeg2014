import numpy as np
from sktensor import dtensor, cp_als
from sklearn.externals.joblib import Memory
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.stats import kurtosis
from sklearn.ensemble import BaggingClassifier
from utils import load_train_subjects, load_test_subjects

import matplotlib.pyplot as plt


def separate_grad_mag(X):
    itr = np.arange(X.shape[1])
    grad1 = np.where(itr % 3 == 0)[0]
    grad2 = np.where((itr - 1) % 3 == 0)[0]
    mag1 = np.where((itr - 2) % 3 == 0)[0]
    return grad1, grad2, mag1


def drop_fifty_and_ten_hz(X):
    idx = np.where(kurtosis(X) > 2)[0]
    return idx


def get_basis(labels, data, n_components):
    # Generate decompositions
    print("Performing tensor decomposition of training data.")
    all_basis = []
    for n in np.unique(labels):
        idx = np.where(labels == n)[0]
        X = data[idx]
        grad1, grad2, mag1 = separate_grad_mag(X)
        grad = np.concatenate((grad1, grad2))
        # Magnetometers look real rough
        for idx in [grad, mag1]:
            Xi = X[:, idx, :]
            r = cp_als(dtensor(Xi), n_components, init="nvecs")
            r_good_idx = drop_fifty_and_ten_hz(r[0].U[2])
            basis = r[0].U[2][:, r_good_idx]
            all_basis.append(basis)

    basis = np.hstack(all_basis)
    del all_basis
    return basis


mem = Memory(cachedir="cache", verbose=10)
cp_als = mem.cache(cp_als)
get_basis = mem.cache(get_basis)
load_train_subjects = mem.cache(load_train_subjects)
load_test_subjects = mem.cache(load_test_subjects)


def subject_splitter(X, y, sids):
    # Total hack but interesting idea if it can be generalized
    all_X = []
    all_y = []
    for n in np.unique(sids):
        idx = np.where(sids != n)[0]
        all_X.append(X[idx])
        all_y.append(y[idx])
        # all_y.append(2 * n + y[idx])
        #all_y.append((2 * y[idx] - 1) * n)
    return zip(all_X, all_y)


class TensorTransform(BaseEstimator, TransformerMixin):
    def __init__(self, subject_ids, n_components):
        self.subject_ids = subject_ids
        self.n_components = n_components
        self.locked = False

    def fit(self, X, y=None):
        if not self.locked:
            components = get_basis(self.subject_ids, X, self.n_components)
            self.components_ = components
            self.locked = True
        return self

    def transform(self, X, y=None):
        return np.dot(X, self.components_).reshape(len(X), -1)


class FeatureUnionWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.clf.predict_proba(X).reshape(len(X), -1)


class StackedGeneralizer(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        # Filthy hack
        sids = X[:, -1]
        all_pipelines = [make_pipeline(LogisticRegressionCV()).fit(X_s, y_s) for
                         X_s, y_s in subject_splitter(X[:, :-1], y, sids)]
        f_union = make_union(*[FeatureUnionWrapper(p) for p in all_pipelines])
        self.clf_ = make_pipeline(f_union, LogisticRegressionCV()).fit(X[:, :-1], y)
        return self

    def predict(self, X, y=None):
        # Dirty hack
        return self.clf_.predict(X[:, :-1])

all_train_data, all_train_targets, all_train_labels = load_train_subjects()
all_test_data, all_test_labels = load_test_subjects()

tf = TensorTransform(np.concatenate((all_train_labels, all_test_labels)),
                     n_components=10)
tf.fit(np.vstack((all_train_data, all_test_data)))

val_idx = np.where(all_train_labels == 16)[0]
all_val_data = all_train_data[val_idx]
all_val_targets = all_train_targets[val_idx]
all_val_labels = all_train_labels[val_idx]

train_idx = np.where(all_train_labels < 15)[0]
all_train_data = all_train_data[train_idx]
all_train_targets = all_train_targets[train_idx]
all_train_labels = all_train_labels[train_idx]

X_train = all_train_data
y_train = all_train_targets
X_val = all_val_data
y_val = all_val_targets

X_train = tf.transform(X_train)
X_val = tf.transform(X_val)

X_train = np.hstack((X_train, all_train_labels[:, np.newaxis]))
X_val = np.hstack((X_val, all_val_labels[:, np.newaxis]))

# Train pipeline
bc = BaggingClassifier(base_estimator=StackedGeneralizer(),
                       n_estimators=10,
                       max_samples=.8,
                       max_features=1.0,
                       random_state=1999)

bg = make_pipeline(StandardScaler(), bc)
print("Training...")
bg.fit(X_train, y_train)

y_pred = bg.predict(X_train)
print("Accuracy on training data")
print(accuracy_score(y_train, y_pred))

y_pred = bg.predict(X_val)
print("Accuracy on validation data")
print(accuracy_score(y_val, y_pred))

# Try some cleanup...
del X_train
del all_train_data
del X_val
del all_val_data

# Load test data
print("Load testing data.")
all_test_data, all_test_labels = load_test_subjects()

X_test = all_test_data

X_test = tf.transform(X_test)
X_test = np.hstack((X_test, all_test_labels[:, np.newaxis]))
y_pred = bg.predict(X_test)

predictions = y_pred
submission_file = "submission.csv"
print("Creating submission file", submission_file)
indices = np.zeros_like(predictions)
for l in np.unique(all_test_labels):
    indices[all_test_labels == l] = 1000 * l + np.arange(
        (all_test_labels == l).sum())

with open(submission_file, "w") as f:
    f.write("Id,Prediction\n")
    for ind, pred in zip(indices, predictions):
        f.write("%d,%d\n" % (ind, pred))
print("Done.")
