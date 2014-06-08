import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from utils import load_train_subjects, load_test_subjects

import matplotlib.pyplot as plt



mem = Memory(cachedir="cache", verbose=10)
load_train_subjects = mem.cache(load_train_subjects)
load_test_subjects = mem.cache(load_test_subjects)

all_train_data, all_train_targets, all_train_labels = load_train_subjects()
all_test_data, all_test_labels = load_test_subjects()

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

X_train = X_train.reshape(len(X_train), -1)
X_val = X_val.reshape(len(X_val), -1)

# Train pipeline
bg = make_pipeline(StandardScaler(), SVC())
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

X_test = X_test.reshape(len(X_test), -1)
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
