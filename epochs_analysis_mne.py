import numpy as np
from scipy.io import loadmat
import mne

from path import path

subject_ids = [1]  # range(1, 17)
subject_names = ["train_subject%02d.mat" % sid for sid in subject_ids]
data_dir = path('data')


reject = dict(mag=5e-12, grad=400e-13)
layout = mne.layouts.read_layout('Vectorview-all')
info = mne.create_info(
    layout.names, 250, ['grad', 'grad', 'mag'] * 102)

broken_epochs = {
    1 : [535],  # Subject 1 has weird spikes in almost every epoch, 
                # at different times. Need to check if specific channels
    4 : [105, 360, 388, 439]}


for subject in subject_names:

    f = loadmat(data_dir / subject)
    X = f['X']
    y = f['y'].ravel() * 2 - 1

    events = np.zeros([len(X), 3], dtype=np.int64)
    events[:, 0] = np.arange(len(events))
    events[:, 2] = y

    epochs = mne.epochs.EpochsArray(
        X, info, events, event_id=dict(face=1, scramble=-1), tmin=-.5)
    epochs.reject = reject

    epochs.plot()
