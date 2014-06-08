import numpy as np
from scipy.io import loadmat
from path import path

train_subject_ids = range(1, 17)
train_subject_names = ["train_subject%02d.mat" % sid
                       for sid in train_subject_ids]

data_dir = path("data")

all_data = []
all_targets = []
all_labels = []

for subject, sid in zip(train_subject_names, train_subject_ids):
    print subject
    f = loadmat(data_dir / subject)
    X = f['X'][:, 160:, 125:250]
    y = f['y'].ravel() * 2 - 1

    all_data.append(X)
    all_targets.append(y)

    all_labels.append([sid] * len(X))


from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pipeline = Pipeline([('scaler', StandardScaler()),
                     ('lr', LogisticRegression(C=1e-2))])

all_scores = []

# out of subject predictions
for sid, X, y in zip(train_subject_ids, all_data, all_targets):

    print "Fitting on %d" % sid
    cv = StratifiedShuffleSplit(y, test_size=.5, n_iter=10)
    for train, test in cv:
        pipeline.fit(X.reshape(len(X), -1)[train], y[train])
        fold_scores = []
        all_scores.append(fold_scores)

        for sid2, X2, y2 in zip(train_subject_ids, all_data, all_targets):
            if sid2 == sid:
                predictions = pipeline.predict(X.reshape(len(X), -1)[test])
                fold_scores.append(accuracy_score(y[test], predictions))
            else:
                predictions = pipeline.predict(X2.reshape(len(X2), -1))
                fold_scores.append(accuracy_score(y2, predictions))
            print "%d %1.3f" % (sid2, fold_scores[-1])


# predicting subject label of left out subject
all_data = np.concatenate(all_data)
all_targets = np.concatenate(all_targets)
all_labels = np.concatenate(all_labels)

from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.multiclass import OneVsRestClassifier

pipeline2 = Pipeline([('scaler', StandardScaler()),
                     #('lr', OneVsRestClassifier(LogisticRegression(C=1e-2)))
                      ('lr', LogisticRegression(C=1e-2))
])


lolo = LeaveOneLabelOut(all_labels)

left_out_labels = []
all_label_probabilities = []
for train, test in lolo:
    cv = StratifiedShuffleSplit(all_labels[train], test_size=.875, n_iter=5)
    left_out_labels.append(np.unique(all_labels[test]))
    print left_out_labels[-1]
    label_probabilities = []
    all_label_probabilities.append(label_probabilities)

    for train2, test2 in cv:
        pipeline2.fit(all_data.reshape(len(all_data), -1)[train[train2]],
                     all_labels[train[train2]])
        prediction_probas = pipeline2.predict_proba(
            all_data.reshape(len(all_data), -1)[test])

        label_probabilities.append(prediction_probas)
        print prediction_probas

