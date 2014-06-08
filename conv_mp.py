import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.utils import check_random_state


def fft_convolution_product(a, b, axes=((0,), (0,)), mode="valid"):
    """Does an fft convolution product along given axes.
    If sizes along axes correspond and `valid` is chosen, then this reduces
    to the common matrix product."""

    a_axes, b_axes = axes
    if len(a_axes) > 1 or len(b_axes) > 1:
        raise NotImplementedError("Only implemented for 1D conv right now")

    # Make smaller template be in first array
    if a.shape[a_axes[0]] > b.shape[b_axes[0]]:
        raise ValueError("Please specify smaller conv kernel in first place")
    # embed smaller array in larger one
    size = list(a.shape)
    for aa, ab in zip(a_axes, b_axes):
        size[aa] = b.shape[ab]

    a_ = np.zeros(size)
    slices = [slice(None, s) for s in a.shape]
    a_[slices] = a

    a_ = np.fft.fftn(a_, axes=a_axes)
    b_ = np.fft.fftn(b, axes=b_axes)

    # roll convolution axes to the end
    a_ = np.rollaxis(a_, a_axes[0], a_.ndim)
    b_ = np.rollaxis(b_, b_axes[0], b_.ndim)

    remaining_axes_a = tuple(slice(None, s) for s in a_.shape[:-1])
    remaining_axes_b = tuple(slice(None, s) for s in b_.shape[:-1])

    broad_axes_a = remaining_axes_a + (np.newaxis,) * len(remaining_axes_b)
    broad_axes_b = (np.newaxis,) * len(remaining_axes_a) + remaining_axes_b

    convolution = np.fft.ifftn(a_[broad_axes_a] * 
                               b_[broad_axes_b],
                               axes=(-1,))
    if mode == "valid":
        convolution = convolution[..., a.shape[a_axes[0]] - 1:]
    else:
        raise NotImplementedError

    return np.real(convolution)


def mp_code_update(residual, filters, copy_residual=True):
    """Does one MP update. Works on last axis. Filters should be 2D"""

    if filters.ndim > 2:
        raise ValueError("filters should be 2D")
    filter_length = filters.shape[-1]

    filter_norms = np.sqrt((filters.reshape(len(filters), -1) ** 2).sum(-1))
    normalized_filters = filters / filter_norms[..., np.newaxis]

    convolutions = fft_convolution_product(
        normalized_filters[..., ::-1],
        residual,
        axes=((filters.ndim - 1,), (residual.ndim - 1,)))

    abs_convolutions = np.abs(np.rollaxis(convolutions, 0, -1))
    argmaxes = abs_convolutions.reshape(
        residual.shape[:-1] + (-1,)).argmax(-1)

    argmax_fil = argmaxes // convolutions.shape[-1]
    argmax_t = argmaxes % convolutions.shape[-1]

    if copy_residual:
        residual = residual.copy()

    channel_index_slices = [slice(0, p) for p in residual.shape[:-1]]
    channel_indices = np.mgrid[channel_index_slices]

    activation_value = np.rollaxis(convolutions, 0, -1)[
        list(channel_indices) + [argmax_fil, argmax_t]]

    activations = np.zeros_like(np.rollaxis(convolutions, 0, -1))
    activations[list(channel_indices) +
                [argmax_fil, argmax_t]] = activation_value

    for chind in channel_indices.reshape(len(channel_indices), -1).T:

        fil_index = argmax_fil[list(chind[:, np.newaxis])][0]
        t_index = argmax_t[list(chind[:, np.newaxis])][0]
        activation = activation_value[list(chind[:, np.newaxis])][0]
        sl = [slice(c, c + 1) for c in chind] + \
            [slice(t_index, t_index + filter_length)]
            #  [slice(fil_index, fil_index + 1)] + \
        residual[sl] -= activation * normalized_filters[fil_index]

            # Watch out, as of now, activations are wrt normed filters
    return activations, residual


def conv_mp(signal, filters, n_components=1):
    counter = 0
    residual = signal.copy()
    global_activations = 0
    while counter < n_components:
        counter += 1
        activations, residual = mp_code_update(residual, filters)
        global_activations = global_activations + activations
    return global_activations, residual


def kron_id_view(vec, id_length, axis=-1):
    shape = (vec.shape[:axis] +
             (vec.shape[axis] - id_length + 1, id_length) +
             vec.shape[axis % vec.ndim + 1:])
    strides = vec.strides[:axis] + (vec.strides[axis],) + vec.strides[axis:]

    return as_strided(vec, shape=shape, strides=strides)


def update_filters(signal, activations):
    signal_length = signal.shape[-1]
    activation_length = activations.shape[-1]
    filter_length = signal_length - activation_length + 1
    num_filters = activations.shape[-2]

    ata = np.einsum('ijkl, ijml -> km', activations, activations)
    inv_ata = np.linalg.inv(ata)

    v_ = np.zeros([num_filters, filter_length])
    for i in range(filter_length):
        v_[:, i] = np.einsum(
            "ijkl, ijl -> k",
            activations,
            signal[:, :, i:-(filter_length - i - 1) or None])

    return inv_ata.dot(v_)


def conv_dict_learning(signal,
                       n_components=20,
                       n_iter=100,
                       n_templates=3,
                       template_length=20,
                       init_templates=None,
                       random_state=42,):

    rng = check_random_state(random_state)
    if init_templates is not None:
        templates = init_templates.copy()
    else:
        templates = rng.randn(n_templates, template_length)

    for i in xrange(n_iter):
        activations, residual = conv_mp(signal, templates, n_components)
        templates = update_filters(signal, activations)

    return templates, activations, residual


from numpy.testing import assert_array_almost_equal


def test_simple_convolution():

    b = np.arange(20)
    A = np.eye(5)

    for a in A:
        npconv = np.convolve(a, b, mode="valid")
        convprod = fft_convolution_product(a, b)

        assert_array_almost_equal(npconv, convprod)


def test_multiple_convolution():
    b = np.arange(100).reshape(5, 20)
    a = np.eye(4)

    convprod = fft_convolution_product(a, b, axes=((1,), (1,)))

    convolutions = np.array([[np.convolve(aa, bb, mode="valid") for bb in b]
                             for aa in a])

    assert_array_almost_equal(convprod, convolutions)


def generate_conv_sparse_signal():

    filter_size = 20
    filter_1 = np.zeros(filter_size)
    filter_1[0] = 1.

    x = np.arange(filter_size)

    filter_2 = np.maximum(
        0, 1. - ((x - filter_size / 2.) / (filter_size / 4.)) ** 2)

    filter_3 = np.maximum(
        0, 1. - np.abs((x - filter_size / 2.) / (filter_size / 4.)))

    filters = np.array([filter_1, filter_2, filter_3])

    rng = np.random.RandomState(42)
    signal_length = 400
    support_fraction = .02
    support = rng.rand(3, signal_length) < support_fraction
    activation_values = rng.randn(support.sum())
    activations = np.zeros_like(support, dtype=np.float64)
    activations[support] = activation_values

    signals = fft_convolution_product(
        filters,
        activations,
        axes=((1,), (1,)))[[0, 1, 2], [0, 1, 2]]

    return signals, filters, activations


def test_mp_code_update():
    signals, filters, activations = generate_conv_sparse_signal()
    signal1 = signals[0:2].sum(0)
    signal2 = signals[1:].sum(0)
    signal = np.array([
            [signals.sum(0), signals[2]],
            [signals[0], signals[1]],
            [signal1, signal2]])
    act, res = mp_code_update(signal, filters)

    return signal, filters, act, res


def test_conv_mp():
    signals, filters, activations = generate_conv_sparse_signal()
    signal1 = signals[0:2].sum(0)
    signal2 = signals[1:].sum(0)
    signal = np.array([
            [signals.sum(0), signals[2]],
            [signals[0], signals[1]],
            [signal1, signal2]])
    act, res = conv_mp(signal, filters, n_components=10)

    return signal, filters, act, res


def test_filter_update():
    signals, filters, activations = generate_conv_sparse_signal()
    signal1 = signals[0:2].sum(0)
    signal2 = signals[1:].sum(0)
    signal = np.array([
            [signals.sum(0), signals[2]],
            [signals[0], signals[1]],
            [signal1, signal2]])

    rng = np.random.RandomState(42)
    init_filters = rng.randn(*filters.shape)
    act, res = conv_mp(signal, filters, n_components=20)
    new_filters = update_filters(signal, act)

    return init_filters, new_filters


def test_conv_dict_learning():
    signals, filters, activations = generate_conv_sparse_signal()
    signal1 = signals[0:2].sum(0)
    signal2 = signals[1:].sum(0)
    signal = np.array([
            [signals.sum(0), signals[2]],
            [signals[0], signals[1]],
            [signal1, signal2]])


    filters, activations, residual = conv_dict_learning(signal,
                                                        n_components=20,
                                                        n_iter=10,
                                                        n_templates=3,
                                                        template_length=20,
                                                        )
    return filters, activations, residual, signal



