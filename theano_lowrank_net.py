import numpy as np
# from scipy.io import loadmat

# train_subject_ids = range(1, 17)
# test_subject_ids = range(17, 24)

# train_subject_names = ["train_subject%02d.mat" % sid
#                        for sid in train_subject_ids]
# test_subject_names = ["test_subject%02d.mat" % sid
#                       for sid in test_subject_ids]

# from path import path
# data_dir = path("data")

# train_data = []
# train_targets = []
# train_labels = []

# for sid, subject in zip(train_subject_ids, train_subject_names):
#     print subject
#     f = loadmat(data_dir / subject)
#     X = f['X'][:, 159:, 125:250]
#     y = f['y'].ravel()
#     labels = [sid] * len(y)

#     # try standard scaling here
#     X -= X.mean(0)
#     X /= X.std(0)

#     train_data.append(X)
#     train_targets.append(y)
#     train_labels.append(labels)


# train_data = np.concatenate(train_data)
# train_targets = np.concatenate(train_targets)
# train_labels = np.concatenate(train_labels)

from utils import load_train_subjects
train_data, train_targets, train_labels = load_train_subjects()

joint_labels = np.array(["%s_%s" % (str(l), str(t))
                for t, l in zip(train_labels, train_targets)])
unique_labels, label_indices = np.unique(joint_labels, return_inverse=True)


train_gradiometers = train_data.reshape(
    len(train_data), -1, 3,
    train_data.shape[-1])[:, :, :2, :].reshape(
    len(train_data), -1,
    train_data.shape[-1])


import theano
import theano.tensor as T

theano.config.floatX = 'float32'

rank_s, rank_t = 4, 4
n_joint_outputs = 32

# initialize shared variables
rng = np.random.RandomState(42)
# matrices attacking each side of a trial matrix
W_sensors = (rng.rand(rank_s, train_data.shape[1]) - 0.5).astype('float32')
W_time = (rng.rand(train_data.shape[2], rank_t) - 0.5).astype('float32')
# the matrix from the low rank representation to joint output and offset
W_joint_outputs = (rng.rand(rank_s * rank_t, n_joint_outputs) - 0.5
                   ).astype('float32')
b_joint_outputs = np.zeros(n_joint_outputs, dtype=np.float32)

input_data = T.tensor3(name='input_data', dtype='float32')
SW_sensors = theano.shared(W_sensors, name='SW_sensors')
SW_time = theano.shared(W_time, name='SW_time')
SW_joint_outputs = theano.shared(W_joint_outputs, name='SW_joint_outputs')
Sb_joint_outputs = theano.shared(b_joint_outputs, name='Sb_joint_outputs')

sensor_projections = SW_sensors.dot(input_data).dimshuffle(1, 0, 2)
time_projections = sensor_projections.dot(SW_time)

joint_outputs_raw = time_projections.reshape(
    (-1, rank_s * rank_t)).dot(SW_joint_outputs)

joint_outputs_raw_b = joint_outputs_raw + Sb_joint_outputs.dimshuffle('x', 0)

joint_outputs_softmax = T.nnet.softmax(joint_outputs_raw_b)
y = T.lvector('y')

negative_log_likelihood = -T.mean(
    T.log(joint_outputs_softmax)[T.arange(y.shape[0]), y])

classify = T.argmax(joint_outputs_softmax, axis=1)
accuracy = T.cast(T.eq(classify, y), 'float32').mean()


grad_log_lik = T.grad(cost=negative_log_likelihood,
                      wrt=[SW_sensors,
                           SW_time,
                           SW_joint_outputs,
                           Sb_joint_outputs])
# write this gradient separately because the other won't output float32 ...
gll_SWjo = T.grad(cost=negative_log_likelihood, wrt=SW_joint_outputs)
gll_SWsen = T.grad(cost=negative_log_likelihood, wrt=SW_sensors)

learning_rate = .01
batch_size = 50
updates = [(SW_sensors, SW_sensors - learning_rate * gll_SWsen),
           (SW_time, SW_time - learning_rate * grad_log_lik[1]),
           (SW_joint_outputs, SW_joint_outputs -
            learning_rate * gll_SWjo),
           (Sb_joint_outputs, Sb_joint_outputs -
            learning_rate * grad_log_lik[3])]

index = T.lscalar('index')


from sklearn.cross_validation import StratifiedShuffleSplit
train, val_test = iter(StratifiedShuffleSplit(label_indices,
                                              n_iter=1,
                                              test_size=.5)).next()

val = val_test[:len(val_test) / 2]
test = val_test[len(val_test) / 2:]

reduced_label_indices = np.unique(label_indices,
                                  return_inverse=True)[1]

s_label_indices = theano.shared(reduced_label_indices)
s_train_data = theano.shared(train_data)
s_train = theano.shared(train)

givens = {y: s_label_indices[s_train[index * batch_size:
                                     (index + 1) * batch_size]],
         input_data: s_train_data[s_train[index * batch_size:
                                           (index + 1) * batch_size]]}

train_model = theano.function(inputs=[index],
                              outputs=negative_log_likelihood,
                              updates=updates,
                              givens=givens)
val_model = theano.function(inputs=[],
                            outputs=negative_log_likelihood,
                            givens={y: s_label_indices[val],
                                    input_data: s_train_data[val]})
val_accuracy = theano.function(inputs=[],
                               outputs=accuracy,
                               givens={y: s_label_indices[val],
                                       input_data: s_train_data[val]})

len_train = len(train)
n_batches = len_train / batch_size

W_l2_targets = rng.rand(n_joint_outputs, 2).astype('float32')
b_l2_targets = np.zeros(2, dtype=np.float32)

SW_l2_targets = theano.shared(W_l2_targets)
Sb_l2_targets = theano.shared(b_l2_targets)

l2_target_raw = joint_outputs_softmax.dot(SW_l2_targets) + \
    Sb_l2_targets.dimshuffle('x', 0)
l2_target_softmax = T.nnet.softmax(l2_target_raw)
y2 = T.lvector()

l2_neg_log_lik = -T.mean(T.log(l2_target_softmax)[T.arange(y2.shape[0]), y2])

l2_classify = T.argmax(l2_target_softmax, axis=1)
l2_accuracy = T.cast(T.eq(l2_classify, y2), 'float32').mean()


s_targets = theano.shared(train_targets)
grad_l2_neg_log_lik = T.grad(cost=l2_neg_log_lik,
                             wrt=[SW_l2_targets,
                                  Sb_l2_targets])

learning_rate2 = 0.01
l2_updates = [(SW_l2_targets,
               SW_l2_targets - learning_rate2 * grad_l2_neg_log_lik[0]),
              (Sb_l2_targets,
               Sb_l2_targets - learning_rate2 * grad_l2_neg_log_lik[1])]

l2_givens = {y2: s_targets[s_train[index * batch_size:
                                       (index + 1) * batch_size]],
         input_data: s_train_data[s_train[index * batch_size:
                                           (index + 1) * batch_size]]}

train_l2 = theano.function(inputs=[index],
                           outputs=l2_neg_log_lik,
                           updates=l2_updates,
                           givens=l2_givens)

val_l2 = theano.function(inputs=[],
                         outputs=l2_neg_log_lik,
                         givens={y2: s_targets[val],
                                 input_data: s_train_data[val]})

val_l2_accuracy = theano.function(inputs=[],
                               outputs=l2_accuracy,
                               givens={y2: s_targets[val],
                                       input_data: s_train_data[val]})

grad_l2_everybody = T.grad(cost=l2_neg_log_lik,
                           wrt=[SW_sensors,
                                SW_time,
                                SW_joint_outputs,
                                Sb_joint_outputs,
                                SW_l2_targets,
                                Sb_l2_targets])
updates_everybody = [(SW_sensors,
                      SW_sensors - learning_rate * grad_l2_everybody[0]),
                     (SW_time,
                      SW_time - learning_rate * grad_l2_everybody[1]),
                     (SW_joint_outputs,
                      SW_joint_outputs -
                      learning_rate * grad_l2_everybody[2]),
                     (Sb_joint_outputs,
                      Sb_joint_outputs -
                      learning_rate * grad_l2_everybody[3]),
                     (SW_l2_targets,
                      SW_l2_targets - learning_rate2 * grad_l2_everybody[4]),
                     (Sb_l2_targets,
                      Sb_l2_targets - learning_rate2 * grad_l2_everybody[5])]

train_l2_everybody = theano.function(inputs=[index],
                                     outputs=l2_neg_log_lik,
                                     updates=updates_everybody,
                                     givens=l2_givens)


if __name__ == "__main__":
    # f_sensor_projections = theano.function([input_data],
    #                                        sensor_projections)
    # f_time_projections = theano.function([input_data],
    #                                      time_projections)
    # f_joint_outputs_raw = theano.function([input_data],
    #                                       joint_outputs_raw)
    # f_joint_outputs_raw_b = theano.function([input_data],
    #                                         joint_outputs_raw_b)
    f_joint_outputs_softmax = theano.function([input_data],
                                         joint_outputs_softmax)

    # f_neg_log_lik = theano.function([input_data, y],
    #                                 negative_log_likelihood)

    # f_grad_neg_log_lik = theano.function([input_data, y],
    #                                      grad_log_lik)

    # f_gll_SWjo = theano.function([input_data, y], gll_SWjo)
    # f_gll_SWsen = theano.function([input_data, y], gll_SWsen)

    f_accuracy = theano.function([input_data, y], accuracy)


    train_energy = []
    val_energy = []
    val_acc = []
    n_epochs = 200
    for e in range(n_epochs):
        train_epoch = []
        train_energy.append(train_epoch)
        for i in range(n_batches):
            train_epoch.append(train_model(i))

        val_energy.append(val_model())
        val_acc.append(val_accuracy())
        print "Epoch %d: mean train %1.3f, val %1.3f, acc %1.3f" % (
            e, np.mean(train_epoch), val_energy[-1], val_acc[-1])



    n_l2_epochs = 200
    train_energy_l2 = []
    val_energy_l2 = []
    val_acc_l2 = []
    for e in range(n_l2_epochs):
        train_epoch = []
        train_energy_l2.append(train_epoch)
        for i in range(n_batches):
            train_epoch.append(train_l2(i))

        val_energy_l2.append(val_l2())
        val_acc_l2.append(val_l2_accuracy())
        print "Epoch %d: mean train %1.3f, val %1.3f, acc %1.3f" % (
            e, np.mean(train_epoch), val_energy_l2[-1],
            val_acc_l2[-1])

    n_everybody_epochs = 200
    train_energy_everybody = []
    val_energy_everybody = []
    val_acc_everybody = []
    for e in range(n_everybody_epochs):
        train_epoch = []
        train_energy_everybody.append(train_epoch)
        for i in range(n_batches):
            train_epoch.append(train_l2_everybody(i))

        val_energy_everybody.append(val_l2())
        val_acc_everybody.append(val_l2_accuracy())
        print "Epoch %d: mean train %1.3f, val %1.3f, acc %1.3f" % (
            e, np.mean(train_epoch), val_energy_everybody[-1],
            val_acc_everybody[-1])



    # softmax_on_val_set = f_joint_outputs_softmax(train_data[val])
    # softmax_on_test_set = f_joint_outputs_softmax(train_data[test])

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.multiclass import OneVsRestClassifier
    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics import accuracy_score

    # pipeline = Pipeline([('scaler', StandardScaler()),
    #                  ('clf', 
    #                   (LogisticRegression(
    #                 C=1e-1, penalty="l2")))])

    # pipeline.fit(softmax_on_val_set, train_targets[val])
    # predictions = pipeline.predict(softmax_on_test_set)

    # acc = accuracy_score(train_targets[test], predictions)














# predictions = pipeline.fit(
#     train_data.reshape(len(train_data), -1)[train],
#     label_indices[train]).predict(
#     train_data.reshape(len(train_data), -1)[val])

# acc = accuracy_score(label_indices[val], predictions)
