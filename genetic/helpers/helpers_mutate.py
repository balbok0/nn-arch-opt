import random

import numpy as np
from keras.callbacks import LearningRateScheduler
from typing import *

import helpers_other
from network import Network
from program_variables import program_params as const
from deprecated import deprecated


def add_layer(base_net):
    # type: (Network) -> Network
    """
    Creates a copy of given Network, but with added layer, randomly derived from given parameters.

    :param base_net: Network, which copy (with added layer) will be returned.
    :return: Copy of given network, with additional layer inserted in a random position.
    """
    layer_idx = random.randint(0, len(base_net.arch))

    possible_layers = {}

    if len(base_net.arch) + 1 > const.max_depth != 0:
        remove_layer(base_net)

    if layer_idx == 0:
        possible_layers['conv'] = random.choice(const.mutations.fget()['kernel_size']), \
                                  random.choice(const.mutations.fget()['conv_filters'])
    elif layer_idx == len(base_net.arch):
        possible_layers['dense'] = random.choice(const.mutations.fget()['dense_size'])
    else:
        prev_layer = base_net.arch[layer_idx - 1]
        prev_type = helpers_other.arch_type(prev_layer)
        next_layer = base_net.arch[layer_idx]

        if prev_type in ['conv', 'max']:
            possible_layers['conv'] = (random.choice(const.mutations.fget()['kernel_size']),
                                       random.choice(const.mutations.fget()['conv_filters']))
            if not prev_type == 'max' and helpers_other.can_add_max_number(base_net.arch):
                possible_layers['max'] = 'max'

        check_if_flat = lambda x: helpers_other.arch_type(x) in ['dense', 'drop']

        if check_if_flat(next_layer):
            possible_layers['dense'] = random.choice(const.mutations.fget()['dense_size'])
            if check_if_flat(prev_layer) and not prev_type == 'drop':
                possible_layers['drop'] = 'drop' + str(random.choice(const.mutations.fget()['dropout']))

    layer_name = random.choice(possible_layers.values())

    return _add_layer(base_net, layer_name, layer_idx)


def _add_layer(base_net, layer_name, layer_idx):
    # type: (Network, Union[int, str, Tuple[Tuple[int], int]], int) -> Network
    new_arch = base_net.arch[:layer_idx] + [layer_name] + base_net.arch[layer_idx:]

    layer_idx += 1  # difference between net.arch and actual architecture. - First activation layer.
    if helpers_other.arch_type(layer_name) in ['dense', 'drop']:
        layer_idx += 1  # difference between net.arch and actual architecture. - Flatten layer.

    if const.debug:
        print('')
        print('_add_layer')
        print('Old arch: {}'.format(base_net.arch))
        print('New arch: {}'.format(new_arch))
        print('')

    return Network(
        architecture=new_arch,
        copy_model=helpers_other._insert_layer(
            base_net.model,
            helpers_other.arch_to_layer(layer_name, base_net.act),
            layer_idx),
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )


def remove_layer(base_net):
    # type: (Network) -> Network
    """
    Creates a copy of given Network, but with removed layer, at a random index.

    :param base_net: Network, which copy (with removed layer) will be returned.
    :return: Copy of given network, with one less layer.
    """
    if len(base_net.arch) <= const.min_depth != 0:
        return add_layer(base_net)
    layer_idx = random.randint(1, len(base_net.arch) - 2)  # so that, Conv is always first, and Dense is always last.
    layer_name = base_net.arch[layer_idx]
    new_arch = base_net.arch[:layer_idx] + base_net.arch[layer_idx + 1:]

    layer_idx += 1  # difference between net.arch and actual architecture. - First activation layer.
    if helpers_other.arch_type(layer_name) in ['dense', 'drop'] and len(const.input_shape.fget()) > 2:
        layer_idx += 1  # difference between net.arch and actual architecture. - Flatten layer.

    return Network(
        architecture=new_arch,
        copy_model=helpers_other._remove_layer(base_net.model, layer_idx),
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )


def change_opt(base_net):
    # type: (Network) -> Network
    """
    Creates a copy of given Network, but with changed optimizer on which it will be trained.

    :param base_net: Network, which copy will be returned.
    :return: Copy of given network, with changed optimizer.
    """
    return Network(
        architecture=base_net.arch,
        copy_model=base_net.model,
        opt=random.choice(const.mutations.fget()['optimizer']),
        lr=random.choice(const.mutations.fget()['optimizer_lr']),
        activation=base_net.act,
        callbacks=base_net.callbacks
    )


def change_activation(base_net):
    # type: (Network) -> Network
    """
    Creates a copy of given Network, but with changed activation function on each layer specified in architecture.

    :param base_net: Network, which copy will be returned.
    :return: Copy of given network, with changed activation functions.
    """
    return Network(
        architecture=base_net.arch,
        copy_model=base_net.model,
        opt=base_net.opt,
        activation=random.choice(const.mutations.fget()['activation']),
        callbacks=base_net.callbacks
    )


@deprecated
def change_lr_schedule(base_net):
    # type: (Network) -> Network
    """
    Creates a copy of given Network, but with changed callbacks (Learning Rate Scheduler specifically)
    on which it will be trained.

    :param base_net: Network, which copy will be returned.
    :return: Copy of given network, with changed callbacks.
    """
    if random.choice(const.mutations.fget()['learning_decay_type']) == 'linear':
        def schedule(x):
            return base_net.opt.get_config()['lr'] - \
                   float(x * random.choice(const.mutations.fget()['learning_decay_rate']))
    else:
        def schedule(x):
            return base_net.opt.get_config()['lr'] - \
                   float(np.exp(-x * random.choice(const.mutations.fget()['learning_decay_rate'])))

    return Network(
        architecture=base_net.arch,
        copy_model=base_net.model,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=Network.default_callbacks + [LearningRateScheduler(schedule)]
    )


def add_conv_max(base_net, conv_num=const.n_conv_per_seq):
    # type: (Network, int) -> Network
    """
    Adds a sequence of Convolutional layers, followed by MaxPool layer to a copy of a given Network.

    :param base_net: Network, which copy (with added sequence) will be returned.
    :para: Shape of a singular x input to the Network.
    :param conv_num: Number of convolutional layers in a sequence.
    :return: Copy of given network, with additional sequence inserted in a position of maxpool layer,
                or at the beginning of the model.
    """
    if len(const.input_shape.fget()) < 3:
        return add_dense_drop(base_net)

    if not helpers_other.can_add_max_number(base_net.arch):
        if const.debug:
            print('')
            print('add_conv_max - before calling remove_conv_max')
            print('Arch: {}'.format(base_net.arch))
            print('Max l limit: {}'.format(const.max_layers_limit.fget()))
            print('')
        return remove_conv_max(base_net)

    max_idx = [0]
    idx = 1
    for l in base_net.arch:
        if helpers_other.arch_type(l) == 'max':
            max_idx += [idx]
        idx += 1

    if const.deep_debug:
        print('')
        print('add_conv_max')
        print('max_idx: {}'.format(max_idx))
        print('')

    idx_add = random.choice(max_idx)
    conv_params = (random.choice(const.mutations.fget()['kernel_size']),
                   random.choice(const.mutations.fget()['conv_filters']))
    return __add_conv_max(base_net, idx_add, conv_num, conv_params)


def __add_conv_max(base_net, idx, conv_num, conv_params):
    # type: (Network, int, int, Tuple[Tuple[int, int], int]) -> Network
    new_arch = base_net.arch[:idx] + [conv_params] * conv_num + ['max'] + base_net.arch[idx:]
    new_net = Network(
        architecture=new_arch,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )

    if const.deep_debug:
        print('')
        print('__add_dense_drop: after adding dense')
        print('Index of adding sequence: %d' % idx)
        print('Old arch: {}'.format(base_net.arch))
        print('New arch: {}'.format(new_arch))
        print('\n\t BASE MODEL')
        print(base_net.model.summary())
        print('\n\t NEW MODEL')
        print(new_net.model.summary())
        print('')

    new_first_dense, _ = helpers_other.find_first_dense(new_net.model)
    old_first_dense, _ = helpers_other.find_first_dense(base_net.model)

    for i_l, l in enumerate(new_net.model.layers[:idx + 1]):  # + 1 due to Activation
        old_l = base_net.model.get_layer(index=i_l)
        if const.deep_debug:
            print('')
            print('Idx: {}'.format(i_l))
            print('__add_conv_max: inside for-loop till idx')
            print('Old layer type: {}'.format(type(old_l)))
            print('New layer type: {}'.format(type(l)))
            print('Old layer weights len: {}'.format(len(old_l.get_weights())))
            print('New layer weights len: {}'.format(len(l.get_weights())))
            print('')
        l.set_weights(old_l.get_weights())

    for i_l, l in enumerate(new_net.model.layers[idx + 2 + conv_num:new_first_dense - 1], start=idx + 1):
        from keras.layers import MaxPool2D

        assert isinstance(base_net.model.get_layer(index=i_l), type(l))

        if not isinstance(l, MaxPool2D):
            if const.deep_debug:
                print('')
                print('__add_conv_max: inside for-loop since end of seq, to dense start')
                print('Idx old: {}'.format(i_l))
                print('idx new: {}'.format(1 + conv_num + i_l))
                print('Old layer type: {}'.format(type(base_net.model.get_layer(index=i_l))))
                print('New layer type: {}'.format(type(l)))

            old_kernel = base_net.model.get_layer(index=i_l).get_weights()[1]
            rand_kernel = l.get_weights()[1]

            if const.deep_debug:
                print('Old kernel shape: {}'.format(old_kernel.shape))
                print('New (randomly initialized, with correct shape) kernel shape: {}'.format(rand_kernel.shape))
                print('')

            l.set_weights([l.get_weights()[0]] + [old_kernel])

    new_net.model.get_layer(index=new_first_dense).set_weights(
        [new_net.model.get_layer(index=new_first_dense).get_weights()[0]] +
        [base_net.model.get_layer(index=old_first_dense).get_weights()[1]]
    )

    for i_l, l in enumerate(new_net.model.layers[new_first_dense + 1:],
                            start=old_first_dense + 1):
        if const.deep_debug:
            print('')
            print('__add_conv_max: inside for-loop since dense start till end')
            print('Idx old: {}'.format(i_l))
            print('idx new: {}'.format(i_l + new_first_dense - old_first_dense))
            print('Old layer type: {}'.format(type(base_net.model.get_layer(index=i_l))))
            print('New layer type: {}'.format(type(l)))
            print('')
        l.set_weights(base_net.model.get_layer(index=i_l).get_weights())

    return new_net


def add_dense_drop(base_net):
    # type: (Network) -> Network
    """
    Adds a sequence of Dense layer, followed by Dropout layer to a copy of a given Network.

    :param base_net: Network, which copy (with added sequence) will be returned.
    :return: Copy of given network, with additional sequence inserted in a position of a random dropout layer,
                or at the beginning of 1D computations in the model.
    """
    drop_idx = [helpers_other.find_first_drop_dense_arch(base_net.arch)]
    idx = 0
    for l in base_net.arch:
        if helpers_other.arch_type(l) == 'drop':
            drop_idx += [idx + 1]  # Since it can be added after
        idx += 1

    if const.deep_debug:
        print('')
        print('add_drop_dense')
        print('drop_idx: {}'.format(drop_idx))
        print('')

    idx_add = random.choice(drop_idx)
    dense_params = random.choice(const.mutations.fget()['dense_size'])
    drop_params = 'drop%.2f' % random.choice(const.mutations.fget()['dropout'])

    if const.deep_debug:
        print('')
        print('add_drop_dense before private call')
        print('idx_add: {}'.format(idx_add))
        print('dense_params: {}'.format(dense_params))
        print('drop_params: {}'.format(drop_params))
        print('arch: {}'.format(base_net.arch))
        print('')

    return __add_dense_drop(base_net, idx_add, dense_params, drop_params)


def __add_dense_drop(base_net, idx, dense_params, drop_params):
    # type: (Network, int, int, str) -> Network
    if idx < len(base_net.arch) and helpers_other.arch_type(base_net.arch[idx]) == 'drop':
        from warnings import warn
        warn('Invalid index given to __add_dense_drop. idx cannot point at place dropout layer in base_net.arch.\n'
             'Adding 1 to idx, so this check will be satisfied.')
        idx += 1
    new_arch = base_net.arch[:idx] + [dense_params] + [drop_params] + base_net.arch[idx:]

    new_net = Network(
        architecture=new_arch,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )

    if const.debug:
        print('')
        print('__add_dense_drop: after adding dense')
        print('Index of adding sequence: %d' % idx)
        print('Old arch: {}'.format(base_net.arch))
        print('New arch: {}'.format(new_arch))
        print('\n\t BASE MODEL')
        print(base_net.model.summary())
        print('\n\t NEW MODEL')
        print(new_net.model.summary())
        print('')

    dim_offset = 2 if len(const.input_shape.fget()) > 2 else 1
    for i_l, l in enumerate(new_net.model.layers[:idx + dim_offset - 1]):
        if const.deep_debug:
            print('')
            print('__add_dense_drop inside for loop till idx')
            print('\tIdx: {}'.format(i_l))
            print('\tOld layer type: {}'.format(type(base_net.model.get_layer(index=i_l))))
            print('\tNew layer type: {}'.format(type(l)))

            print('')
        l.set_weights(base_net.model.get_layer(index=i_l).get_weights())

    from keras.layers import Dense

    if const.debug:
        print('')
        print('__add_dense_drop: after adding dense')
        print('Index of adding sequence: %d' % idx)
        print('Old arch: {}'.format(base_net.arch))
        print('New arch: {}'.format(new_arch))
        print('\n\t BASE MODEL')
        print(base_net.model.summary())
        print('\n\t NEW MODEL')
        print(new_net.model.summary())
        print('')

    # 1st edge case, one before added sequence. Input matches, but output not necessarily.
    if isinstance(new_net.model.get_layer(index=idx + dim_offset - 1), Dense):
        w_a = base_net.model.get_layer(index=idx + dim_offset - 1).get_weights()
        w_n = new_net.model.get_layer(index=idx + dim_offset - 1).get_weights()

        new_weights_0 = np.array(w_a[0][:len(w_n[0])])
        if len(w_a[0]) < len(w_n[0]):
            if const.deep_debug:
                print('\t\t new_weights shape: {}'.format(new_weights_0.shape))
                print('\t\t w_n[0] add shape: {}'.format(np.array(w_n[0][len(new_weights_0):]).shape))
            new_weights_0 = np.concatenate((new_weights_0, w_n[0][len(new_weights_0):]), axis=0)

        # Output can be different, but we can reuse as many weights as possible
        new_weights_1 = np.array(w_a[1][:len(w_n[1])])
        if len(w_a[1]) < len(w_n[1]):
            if const.deep_debug:
                print(new_weights_1.shape)
                print(np.array(w_n[1][len(new_weights_1):]).shape)
            new_weights_1 = np.concatenate((new_weights_1, w_n[1][len(new_weights_1):]), axis=0)

        new_net.model.get_layer(index=idx + dim_offset - 1).set_weights(
            [new_weights_0, new_weights_1]
        )

    # 2nd edge case, one after added sequence. Output matches, but Input not necessarily.
    if isinstance(new_net.model.get_layer(index=idx + dim_offset + 2), Dense):
        w_a = base_net.model.get_layer(index=idx + dim_offset).get_weights()
        w_n = new_net.model.get_layer(index=idx + dim_offset + 2).get_weights()

        new_weights_0 = np.array(w_a[0][:len(w_n[0])])
        if len(w_a[0]) < len(w_n[0]):
            if const.deep_debug:
                print('\t\t new_weights shape: {}'.format(new_weights_0.shape))
                print('\t\t w_n[0] add shape: {}'.format(np.array(w_n[0][len(new_weights_0):]).shape))
            new_weights_0 = np.concatenate((new_weights_0, w_n[0][len(new_weights_0):]), axis=0)

        # Output can be different, but we can reuse as many weights as possible
        new_weights_1 = np.array(w_a[1][:len(w_n[1])])
        if len(w_a[1]) < len(w_n[1]):
            if const.deep_debug:
                print(new_weights_1.shape)
                print(np.array(w_n[1][len(new_weights_1):]).shape)
            new_weights_1 = np.concatenate((new_weights_1, w_n[1][len(new_weights_1):]), axis=0)

        new_net.model.get_layer(index=idx + dim_offset + 2).set_weights(
            [new_weights_0, new_weights_1]
        )

    if const.debug:
        print('')
        print('__add_dense_drop: after adding dense')
        print('Index of adding sequence: %d' % idx)
        print('Old arch: {}'.format(base_net.arch))
        print('New arch: {}'.format(new_arch))
        print('\n\t BASE MODEL')
        print(base_net.model.summary())
        print('\n\t NEW MODEL')
        print(new_net.model.summary())
        print('')

    for i_l, l in enumerate(new_net.model.layers[idx + dim_offset + 3:], start=idx + dim_offset + 1):
        if const.deep_debug:
            print('')
            print('add_dense_drop inside for loop till end')
            print('Idx old: {}'.format(i_l))
            print('Idx_new: {}'.format(i_l + 3))
            print('Old layer type: {}'.format(base_net.model.get_layer(index=i_l).get_config()))
            print('New layer type: {}'.format(l.get_config()))
            print('')
        l.set_weights(base_net.model.get_layer(index=i_l).get_weights())

    return new_net


def remove_conv_max(base_net):
    # type: (Network) -> Network
    """
    Removes a sequence of Convolution layers, followed by MaxOut layer, in a given Network.\n
    If no such sequence is found, then it adds one, instead of removing it.

    :param base_net: A Network, which copy, with mutations, will be returned.
    :return: A Network, based on base_net, but with a sequence of Conv layers and a MaxOut layer removed.
    """
    if len(const.input_shape.fget()) < 3:
        return remove_dense_drop(base_net)

    max_idx = []
    idx = 0  # Since Activation layer is always first.
    for l in base_net.arch:
        if helpers_other.arch_type(l) == 'max':
            max_idx += [idx]
        idx += 1

    if not max_idx:
        return add_conv_max(base_net)

    if len(max_idx) > 1:
        curr_idx = random.randint(1, len(max_idx))
    else:
        curr_idx = 1

    if const.deep_debug:
        print('')
        print('remove_conv_max')
        print('\tmax_idx: {}'.format(max_idx))
        print('\tcurr_idx: {}'.format(curr_idx - 1))
        print('')

    end = max_idx[curr_idx - 1]

    if curr_idx == 1:
        start = 0
    else:
        start = max_idx[curr_idx - 2] + 1

    return __remove_conv_max(base_net, start, end)


def __remove_conv_max(base_net, idx_start, idx_end):
    # type: (Network, int, int) -> Network
    new_arch = base_net.arch[:idx_start] + base_net.arch[idx_end + 1:]

    new_net = Network(
        architecture=new_arch,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )

    if const.debug:
        print('')
        print('__remove_conv_max')
        print('\told arch: {}'.format(base_net.arch))
        print('\tnew arch: {}'.format(new_arch))
        print('\tidx_start: {}'.format(idx_start))
        print('\tidx_end: {}'.format(idx_end))
        print('')

    new_first_dense, _ = helpers_other.find_first_dense(new_net.model)
    old_first_dense, _ = helpers_other.find_first_dense(base_net.model)

    for i_l, l in enumerate(new_net.model.layers[:idx_start + 1]):  # + 1 due to Activation
        old_l = base_net.model.get_layer(index=i_l)
        if const.deep_debug:
            print('')
            print('Idx: {}'.format(i_l))
            print('__remove_conv_max: inside for-loop till idx_start')
            print('Old layer type: {}'.format(type(old_l)))
            print('New layer type: {}'.format(type(l)))
            print('Old layer weights len: {}'.format(len(old_l.get_weights())))
            print('New layer weights len: {}'.format(len(l.get_weights())))
            print('')
        l.set_weights(old_l.get_weights())

    for i_l, l in enumerate(new_net.model.layers[idx_start + 1:new_first_dense - 1], start=idx_end + 2):
        from keras.layers import MaxPool2D

        assert isinstance(base_net.model.get_layer(index=i_l), type(l))

        if not isinstance(l, MaxPool2D):
            if const.deep_debug:
                print('')
                print('__rmv_conv_max: inside for-loop since end of seq, to dense start')
                print('Idx old: {}'.format(i_l))
                print('idx new: {}'.format(i_l - idx_end - 1 + idx_start))
                print('Old layer type: {}'.format(type(base_net.model.get_layer(index=i_l))))
                print('New layer type: {}'.format(type(l)))

            old_kernel = base_net.model.get_layer(index=i_l).get_weights()[1]
            rand_kernel = l.get_weights()[1]

            if const.deep_debug:
                print('Old kernel shape: {}'.format(old_kernel.shape))
                print('New (randomly initialized, with correct shape) kernel shape: {}'.format(rand_kernel.shape))
                print('')

            l.set_weights([l.get_weights()[0]] + [old_kernel])

    new_net.model.get_layer(index=new_first_dense).set_weights(
        [new_net.model.get_layer(index=new_first_dense).get_weights()[0]] +
        [base_net.model.get_layer(index=old_first_dense).get_weights()[1]]
    )

    for i_l, l in enumerate(new_net.model.layers[new_first_dense + 1:],
                            start=old_first_dense + 1):
        if const.deep_debug:
            print('')
            print('__rmv_conv_max: inside for-loop since dense start till end')
            print('Idx old: {}'.format(i_l))
            print('idx new: {}'.format(i_l + new_first_dense - old_first_dense))
            print('Old layer type: {}'.format(type(base_net.model.get_layer(index=i_l))))
            print('New layer type: {}'.format(type(l)))
            print('')
        l.set_weights(base_net.model.get_layer(index=i_l).get_weights())

    return new_net


def remove_dense_drop(base_net):
    # type: (Network) -> Network
    """
    Removes a sequence of Dense layer, followed by Dropout layer/layers, in a given Network.\n
    If no such sequence is found, then it adds one (Dropout + Dense), instead of removing it.

    :param base_net: A Network, which copy, with mutations, will be returned.
    :para: Shape of a singular x input to the Network.
    :return: A Network, based on base_net, but with a sequence of Dense layer and a Dropout layers removed.
    """
    drop_idx = []
    for i_l, l in enumerate(base_net.arch):
        if helpers_other.arch_type(l) == 'drop':
            drop_idx += [i_l]

    if not drop_idx:
        return add_dense_drop(base_net)

    if len(drop_idx) > 1:
        curr_idx = random.randint(1, len(drop_idx))
    else:
        curr_idx = 1

    if const.debug:
        print('')
        print('remove_dense_drop')
        print('drop_idxs: {}'.format(drop_idx))
        print('')

    drop_arch_idx = drop_idx[curr_idx - 1]

    return __remove_dense_drop(base_net, drop_arch_idx)


def __remove_dense_drop(base_net, drop_idx):
    # type: (Network, int) -> Network
    idx_start = drop_idx - 1
    while helpers_other.arch_type(base_net.arch[idx_start]) == 'drop':
        idx_start -= 1
    idx_end = drop_idx

    new_arch = base_net.arch[:idx_start] + base_net.arch[idx_end + 1:]
    new_net = Network(
        architecture=new_arch,
        opt=base_net.opt,
        activation=base_net.act,
        callbacks=base_net.callbacks
    )

    if const.debug:
        print('')
        print('__remove_dense_drop')
        print('Base arch: {}'.format(new_arch))
        print('Idx start: {}'.format(idx_start))
        print('Idx end: {}'.format(idx_end))
        print('Index of drop layer in arch: {}'.format(drop_idx))
        print('Drop layer: {}'.format(base_net.arch[drop_idx]))
        print('Layer before: {}'.format(base_net.arch[drop_idx - 1]))
        print('')

    dim_offset = 2 if len(const.input_shape.fget()) > 2 else 1
    for i_l, l in enumerate(new_net.model.layers[:idx_start + dim_offset - 1]):
        if const.deep_debug:
            print('')
            print('__remove_dense_drop inside for loop till idx_start')
            print('\tIdx: {}'.format(i_l))
            print('\tOld layer type: {}'.format(type(base_net.model.get_layer(index=i_l))))
            print('\tNew layer type: {}'.format(type(l)))

            print('')
        l.set_weights(base_net.model.get_layer(index=i_l).get_weights())

    print('Rest new: {}'.format(new_net.model.layers[idx_start + dim_offset + 2:]))
    print('{}'.format(idx_start + dim_offset + 2))
    print('Rest old: {}'.format(base_net.model.layers[idx_end + dim_offset + 3:]))
    print('{}'.format(idx_end + dim_offset + 3))

    for i_l, l in enumerate(new_net.model.layers[idx_start + dim_offset + 2:], start=idx_end + dim_offset + 3):
        if const.deep_debug:
            print('')
            print('__remove_dense_drop inside for loop from idx_end till end')
            print('\tIdx: {}'.format(i_l))
            print('\tOld layer type: {}'.format(type(base_net.model.get_layer(index=i_l))))
            print('\tNew layer type: {}'.format(type(l)))

            print('')
        l.set_weights(base_net.model.get_layer(index=i_l).get_weights())

    return new_net


def add_arch_dense_drop(base_arch):
    # type: (List[Union[str, int, Tuple[Tuple[int, int], int]]]) -> List[Union[str, int, Tuple[Tuple[int, int], int]]]
    drop_idx = []
    idx = 0
    for l in base_arch:
        if helpers_other.arch_type(l) == 'drop':
            drop_idx += [idx]
        elif not drop_idx and helpers_other.arch_type(l) == 'dense':
            drop_idx += [idx]
        idx += 1

    idx = random.choice(drop_idx)
    dense_params = random.choice(const.mutations.fget()['dense_size'])
    drop_params = 'drop%.2f' % random.choice(const.mutations.fget()['dropout'])

    new_arch = base_arch[:]

    if idx >= len(new_arch):
        new_arch = new_arch[:idx] + [drop_params] + new_arch[idx:]

    else:
        new_arch = new_arch[:idx + 1] + [drop_params] + new_arch[idx + 1:]

    if idx >= len(base_arch):
        new_arch = new_arch[:idx] + [dense_params] + new_arch[idx:]

    else:
        new_arch = new_arch[:idx + 1] + [dense_params] + new_arch[idx + 1:]

    return new_arch


def add_arch_conv_max(base_arch,  # type: List[Union[str, int, Tuple[Tuple[int, int], int]]]
                      conv_num=const.n_conv_per_seq  # type int
                      ):
    # type: (...) -> List[Union[str, int, Tuple[Tuple[int, int], int]]]
    if len(const.input_shape.fget()) < 3:
        return add_arch_dense_drop(base_arch)

    if not helpers_other.can_add_max_number(base_arch):
        return add_arch_dense_drop(base_arch)

    max_idx = [0]
    idx = 1
    for l in base_arch:
        if helpers_other.arch_type(l) == 'max':
            max_idx += [idx]
        idx += 1

    idx = random.choice(max_idx)
    conv_params = (random.choice(const.mutations.fget()['kernel_size']),
                   random.choice(const.mutations.fget()['conv_filters']))

    new_arch = base_arch[:]

    new_arch = new_arch[:idx] + ['max'] + new_arch[idx:]

    for l in range(conv_num):
        new_arch = new_arch[:idx] + [conv_params] + new_arch[idx:]

    return new_arch
