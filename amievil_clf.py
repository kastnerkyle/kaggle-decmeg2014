from scipy.io import loadmat
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

def load_training_data():
    print("Creating the trainset.")
    labels = []
    X_train = []
    y_train = []
    for n, subject in enumerate(range(1,17)):
        filename = 'data/train_subject%02d.mat' % subject
        print("Loading", filename)
        data = loadmat(filename, squeeze_me=True)
        XX = data['X'][:, 159:, 125:250]
        yy = data['y']
        X_train.append(XX)
        y_train.append(yy)
        labels.extend([subject] * len(XX))

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    labels = np.array(labels)
    return X_train, y_train, labels

all_train_data, all_train_targets, all_train_labels = load_training_data()

val_idx = np.where(all_train_labels == 16)[0]
all_val_data = all_train_data[val_idx]
all_val_targets = all_train_targets[val_idx]
all_val_labels = all_train_labels[val_idx]

train_idx = np.where(all_train_labels != 16)[0]
all_train_data = all_train_data[train_idx]
all_train_targets = all_train_targets[train_idx]
all_train_labels = all_train_labels[train_idx]

X_train = all_train_data
y_train = all_train_targets
X_val = all_val_data
y_val = all_val_targets

X_train = X_train.reshape(len(X_train), -1)
X_val = X_val.reshape(len(X_val), -1)
bg = Pipeline([('scaler', StandardScaler()),
               ('lr',LogisticRegression(C=1e-5))])
print("Training...")
bg.fit(X_train, y_train)

y_pred = bg.predict(X_train)
print("Accuracy on training data")
print(accuracy_score(y_train, y_pred))

y_pred = bg.predict(X_val)
print("Accuracy on validation data")
print(accuracy_score(y_val, y_pred))
