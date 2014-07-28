import numpy as np
import theano
import theano.tensor as T
theano.config.floatX = 'float32'

from utils import load_train_subjects, load_test_subjects

train_data, train_targets, train_labels = load_train_subjects([4])


rank = 2

all_subject_labels = train_labels

unique_subject_labels, label_associations = np.unique(
    all_subject_labels, return_inverse=True)

rng = np.random.RandomState(42)
init_time_components = (
    rng.rand(rank, train_data.shape[-1]) - 0.5
    ).astype(np.float32)[np.newaxis] * \
    np.ones([len(unique_subject_labels), 1, 1])

init_sensor_components = (
    rng.rand(rank, train_data.shape[1]) - 0.5
    ).astype(np.float32)[np.newaxis] * \
    np.ones([len(unique_subject_labels), 1, 1])

init_offsets = np.zeros(len(unique_subject_labels))

time_components = theano.shared(init_time_components,
                                name='time_components',
                                borrow=False)

sensor_components = theano.shared(init_sensor_components,
                                  name='sensor_components',
                                  borrow=False)

offsets = theano.shared(init_offsets, name='offsets', borrow=False)

input_data = T.tensor3(name='input_data')
input_labels = T.lvector(name='input_labels')

with_offset = input_data - offsets[input_labels].dimshuffle(0, 'x', 'x')
sensor_projections = (
    sensor_components[input_labels].dimshuffle(0, 1, 2, 'x') *
    with_offset.dimshuffle(0, 'x', 1, 2)).sum(axis=2)

time_projections = (sensor_projections *
                    time_components[input_labels]).sum(axis=-1)

blowup = (time_projections.dimshuffle(0, 1, 'x', 'x') *
          sensor_components[input_labels].dimshuffle(0, 1, 2, 'x') *
          time_components[input_labels].dimshuffle(0, 1, 'x', 2)).sum(1)

blowup_with_offset = blowup + offsets[input_labels].dimshuffle(0, 'x', 'x')

autoencoder_mse = T.mean((input_data - blowup_with_offset) ** 2)

smoothness_penalty = T.mean((time_components[:, :, 1:] -
                             time_components[:, :, :-1]) ** 2)
penalty_factor = 10.

grad_ae_mse = T.grad(cost=autoencoder_mse +
                     penalty_factor * smoothness_penalty,
                     wrt=[sensor_components,
                          time_components,
                          offsets])

learning_rate = .1
batchsize = 50
updates = [(sensor_components,
            sensor_components - learning_rate * grad_ae_mse[0]),
           (time_components,
            time_components - learning_rate * grad_ae_mse[1])]

i = T.lscalar('index')

s_train_data = theano.shared(train_data, borrow=True)
s_train_labels = theano.shared(label_associations, borrow=True)

if __name__ == "__main__":
    f_time_projections = theano.function([input_data, input_labels],
                                         time_projections)
    f_blowup = theano.function([input_data, input_labels], blowup)

    from sklearn.cross_validation import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(label_associations, n_iter=1, test_size=.1)

    train, val = iter(cv).next()
    s_train, s_val = theano.shared(train), theano.shared(val)

    givens_train = {input_data:
                        s_train_data[s_train[i * batchsize:
                                                 (i + 1) * batchsize]],
                    input_labels:
                        s_train_labels[s_train[i * batchsize:
                                                   (i + 1) * batchsize]]}

    train_model = theano.function(inputs=[i],
                                  outputs=autoencoder_mse,
                                  updates=updates,
                                  givens=givens_train)

    num_batches = len(train) / batchsize

    num_epochs = 1000
    all_train_mses = []
    for e in range(num_epochs):
        train_mses = []
        all_train_mses.append(train_mses)
        for i in range(num_batches):
            train_mses.append(train_model(i))
        print np.mean(train_mses)

