import copy

import numpy as np
import random
from keras import optimizers
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Activation, Dense, Flatten, Dropout, Conv2D, MaxPool2D
from keras.models import Sequential, Model
from typing import *

from helpers import helpers_other
from helpers.helpers_data import Array_Type
from program_variables.program_params import debug, deep_debug, output_shape, input_shape


class Network:
    """
    A wrapper class around :ref:`Model<keras.engine.training.Model>`.
    """

    @staticmethod
    def _set_dataset(data):
        if len(data[0]) == 2:
            Network.__x_train = data[0][0]
            Network.__x_val = data[1][0]
            Network.__y_train = data[0][1]
            Network.__y_val = data[1][1]
        elif len(data) == 2:
            Network.__x_train = data[0]
            Network.__y_train = data[1]

    default_callbacks = [EarlyStopping(patience=5), helpers_other.NaNSafer()]  # type: List[Callback]

    def __init__(
            self, architecture, copy_model=None,
            opt='adam', lr=None, activation='relu', callbacks=None
    ):
        # type: (List, Model, Union[str, optimizers.Optimizer], float, str, List[Callback]) -> None
        """
        Creates a new instance of Network.

        :param architecture: A list description of the network.
        :param copy_model: A keras Model to make a copy of.
        :param opt: Optimizer used for given network/
        :param lr: Learning rate to be used with an optimizer.
        :param activation: Activation function to be used in a network.
        :param callbacks: Callbacks to be used while training a network.
        """
        assert hasattr(architecture, "__getitem__")
        # Check that architecture vector is first tuples (for convolutions)/MaxOuts,
        # and then integers for the dense layers or 'drop0.2' for a dropout layer
        dense_started = False
        drop_prev = True
        max_prev = True
        idx_to_remove = []

        for j in range(len(architecture)):
            i = architecture[j]
            i_type = helpers_other.arch_type(i)
            if i_type == 'conv':
                max_prev = False
                if dense_started:
                    if debug:
                        print(architecture)
                    raise TypeError(
                        'Architecture is not correctly formatted.\n'
                        'All Convolution layers should appear before Dense/Dropout layers.')

                if not (hasattr(i[0], "__getitem__") and isinstance(i[1], int)):
                    raise TypeError(
                        'Architecture is not correctly formatted.\n'
                        'The part of architecture which cause the problem is ' + str(i))

            elif i_type == 'dense':
                dense_started = True
                drop_prev = False

            elif i_type == 'max':
                if dense_started:
                    if debug:
                        print(architecture)
                    raise TypeError(
                        'Architecture is not correctly formatted.\n'
                        'All MaxPool layers should appear before Dense/Dropout layers.\n'
                        'The part of architecture which cause the problem is ' + str(i)
                    )
                if max_prev:
                    idx_to_remove = [j] + idx_to_remove
                max_prev = True

            elif i_type == 'drop':
                dense_started = True
                if drop_prev or j == len(architecture) - 1:
                    idx_to_remove = [j] + idx_to_remove
                else:
                    try:
                        if i.lower().startswith('dropout'):
                            val = float(i[7:])
                        else:
                            val = float(i[4:])
                        if val >= 1. or val <= 0.:
                            raise AttributeError(
                                'Architecture is not correctly formatted.\n'
                                'Dropout value should be in range (0.0, 1.0).\n'
                                'The part of architecture which cause the problem is ' + str(i)
                            )
                    except ValueError:
                        raise ValueError(
                            'Architecture is not correctly formatted.\n'
                            'Arguments for dropout layer should be in form of \'drop\' or \'dropout\' '
                            'followed by a float.\n'
                            'In example: dropout.2, drop0.5 are valid inputs.\n'
                            'The part of architecture which cause the problem is ' + str(i)
                        )
                drop_prev = True

            else:
                raise TypeError(
                    'Architecture is not correctly formatted.\n'
                    'Arguments should either be iterable, ints, \'max\', or \'drop0.x\','
                    'where 0.x can be any float fraction.\n'
                    'The part of architecture which cause the problem is ' + str(i)
                )

        for idx in idx_to_remove:
            architecture.pop(idx)

        self.callbacks = callbacks  # type: List[Callback]
        if callbacks is None:
            self.callbacks = Network.default_callbacks
        self.arch = architecture  # type: List
        self.act = activation  # type: str
        self.__model_created = False  # type: bool
        if isinstance(opt, optimizers.Optimizer):
            self.opt = opt  # type: optimizers.Optimizer
        else:
            self.opt = self.__optimizer(opt, lr=lr)  # type: optimizers.Optimizer
        self.__score = 0.  # type: float
        self.__prev_score = 0.  # type: float
        self.__prev_weights = None  # type: np.ndarray
        if copy_model is None:
            self.model = Sequential()  # type: Sequential
            self.__create_model()
        else:
            self.model = helpers_other.clone_model(copy_model, self.act, self.opt)
            assert helpers_other.assert_model_arch_match(self.model, self.arch)

    @staticmethod
    def __optimizer(opt_name, lr=None):
        # type: (str, float) -> optimizers.Optimizer
        """
        Given a name and learning rate returns a keras optimizer based on it.

        :param opt_name: Name of optimizer to use.
                Legal arguments are:

        * adam
        * nadam
        * rmsprop
        * adamax
        * adagrad
        * adadelta

        :param lr: Learning rate of an optimizer.
        :return: A new optimizer based on given name and learning rate.
        """

        opt_name = opt_name.lower()
        if lr is None:
            if opt_name == 'adam':
                return optimizers.Adam()
            elif opt_name == 'sgd':
                return optimizers.SGD(nesterov=True)
            elif opt_name == 'nadam':
                return optimizers.Nadam()
            elif opt_name == 'rmsprop':
                return optimizers.RMSprop()
            elif opt_name == 'adamax':
                return optimizers.Adamax()
            elif opt_name == 'adagrad':
                return optimizers.Adagrad()
            elif opt_name == 'adadelta':
                return optimizers.Adadelta()

        else:
            if opt_name == 'adam':
                return optimizers.Adam(lr=lr)
            elif opt_name == 'sgd':
                return optimizers.SGD(lr=lr, nesterov=True)
            elif opt_name == 'nadam':
                return optimizers.Nadam(lr=lr)
            elif opt_name == 'rmsprop':
                return optimizers.RMSprop(lr=lr)
            elif opt_name == 'adamax':
                return optimizers.Adamax(lr=lr)
            elif opt_name == 'adagrad':
                return optimizers.Adagrad(lr=lr)
            elif opt_name == 'adadelta':
                return optimizers.Adadelta(lr=lr)
        raise AttributeError('Invalid name of optimizer given.')

    def fit(self, x, y, validation_data=None, validation_split=0.,
            epochs=20, initial_epoch=0, batch_size=32, shuffle='batch', verbose=0):
        # type: (Array_Type, Array_Type, Tuple, float, int, int, int, str, int) -> None
        """
        Trains a network on a training set.
        For parameters descriptions look at documentation for keras.models.Model.fit function.
        """
        self.__prev_score = self.__score
        self.__prev_weights = copy.deepcopy(self.model.get_weights())
        self.__score = 0.  # Resets score, so it will not collide w/ scoring it again (but w/ different weights).

        if debug:
            print(self.get_config())

        if validation_data is not None:
            self.model.fit(
                x=x, y=y, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                callbacks=self.callbacks,
                validation_data=validation_data,
                initial_epoch=initial_epoch,
                verbose=verbose
            )
        elif 0. < validation_split < 1.:
            self.model.fit(
                x=x, y=y, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                callbacks=self.callbacks,
                validation_split=validation_split,
                initial_epoch=initial_epoch,
                verbose=verbose
            )
        else:
            self.model.fit(
                x=x, y=y, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                callbacks=self.callbacks,
                initial_epoch=initial_epoch,
                verbose=verbose
            )

    def score(self, y_true, y_score, f=None):
        # type: (np.ndarray, np.ndarray, function) -> float
        """
        Scores a network on a given function/metric.

        :param y_true: Target values of y. Has to match f shape requirements (binary class matrix, if f not defined).
        :param y_score: Values of y to be scored. Has to match f shape requirements, or y_true shape.
        :param f: Function to score a function on. If not given a Area Under ROC Curve is used as a metric.
        :return: Returns a score of this network on given function/metric.
        """
        import inspect

        f = f or helpers_other.multi_roc_score
        try:
            args = inspect.getfullargspec(f).args
        except AttributeError:
            # noinspection PyDeprecation
            args = inspect.getargspec(f).args
        if not ('y_true' in args and 'y_score' in args):
            raise AttributeError('Given function f, should have parameters y_true and y_score.')

        if self.__score == 0.0:
            self.__score = f(y_true=y_true, y_score=y_score)

        if self.__score < self.__prev_score:
            self.__score = self.__prev_score
            self.__prev_score = 0.
            self.model.set_weights(self.__prev_weights)
            self.__prev_weights = None

        return self.__score

    def predict(self, x):
        # type: (List) -> List
        """
        Look at keras.models.Model.predict function docs.
        """
        return self.model.predict(x)

    def save(self, file_path, overwrite=True):
        # type: (str, bool) -> None
        """
        Given path, saves a network.

        :param file_path: A path to which network should be saved.
        :param overwrite: If such file already exists, whether it should be overwritten or not.
        """
        self.model.save(filepath=file_path, overwrite=overwrite)

    def __create_model(self):
        """
        With already set architecture, translates it into actual keras model.
        Also compiles it, so that an actual model is ready to use.
        """
        if self.__model_created:
            raise Exception('Cannot recreate a new model in the same instance.')

        self.__model_created = True

        assert hasattr(self.arch, "__getitem__")
        assert isinstance(self.model, Sequential)
        self.model.add(Activation(activation='linear', input_shape=input_shape.fget()))

        last_max_pool = True
        last_dropout = True
        for i in self.arch:
            new_layer = helpers_other.arch_to_layer(i, self.act)

            if isinstance(new_layer, Conv2D):
                last_max_pool = False
                self.model.add(new_layer)

            elif isinstance(new_layer, Dense):
                if len(self.model.output_shape) > 2:
                    self.model.add(Flatten())
                last_dropout = False
                self.model.add(new_layer)

            elif isinstance(new_layer, MaxPool2D):
                if not last_max_pool:
                    if self.model.output_shape[1] > 2:  # asserts that there's not too many maxpools
                        self.model.add(new_layer)
                        last_max_pool = True
                    else:
                        self.arch.remove(i)
                else:
                    self.arch.remove(i)

            elif isinstance(new_layer, Dropout):
                if not last_dropout:
                    if len(self.model.output_shape) > 2:
                        self.model.add(Flatten())
                    if i.lower().startswith('dropout'):
                        self.model.add(Dropout(rate=float(i[7:])))
                    else:
                        self.model.add(Dropout(rate=float(i[4:])))
                    last_dropout = True
                else:
                    self.arch.remove(i)

            else:
                raise TypeError('Architecture is not correctly formatted.')

        if len(self.model.output_shape) > 2:
            self.model.add(Flatten())
        self.model.add(Dense(units=output_shape.fget(), activation='sigmoid'))
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def get_config(self):
        # type: () -> Dict[str, Any]
        """
        :return: A dictionary, which specifies configuration of this network. It contains:

        #. architecture - a list of layers in a network.
        #. optimizer - on which this network was trained.
        #. activation - activation function on all of layers of this network.
        #. score - score of this network, if function score was called beforehand. 0 otherwise.
        #. callbacks - list of callbacks used while training this network.

        """
        opt_name = str(self.opt.__class__)
        opt_name = opt_name[opt_name.index(".") + 1:]
        opt_name = opt_name[:opt_name.index("\'")]
        opt_name = opt_name[opt_name.index(".") + 1:]
        return {
            'architecture': self.arch,
            'optimizer': '{opt} with learning rate: {lr}'.format(
                opt=opt_name,
                lr="{:.2g}".format(self.opt.get_config()['lr'])
            ),
            'activation': self.act,
            'score': self.__score,
            'callbacks': self.callbacks
        }

    @staticmethod
    def mutate(base_net_1, base_net_2, change_number_cap=3):
        # type: (Network, Network, int) -> List[Network]
        """
        Creates and returns two new Networks, based on passed in parent Networks.

        :param base_net_1: A first parent network on which mutation is based.
        :param base_net_2: A second parent network on which mutation is based.
        :param change_number_cap: Cap number of a random changes in case of random mutations.
        :return: List of 2 Networks, which are based on passed in parent Networks.
        """
        from program_variables.program_params import parent_to_rand_chance, parent_1_to_parent_2_chance

        if random.random() < parent_to_rand_chance:
            return [
                Network._mutate_random(base_net_1, change_number_cap=change_number_cap),
                Network._mutate_random(base_net_2, change_number_cap=change_number_cap)
            ]

        elif random.random() < parent_1_to_parent_2_chance:
            return Network._mutate_parent(base_net_1, base_net_2)
        else:
            return Network._mutate_parent_2(base_net_1, base_net_2)

    @staticmethod
    def _mutate_parent(base_net_1, base_net_2):
        # type: (Network, Network) -> List[Network]
        """
        Creates two new Networks, both based on combination of given parents.

        :param base_net_1: A first parent network on which mutation is based.
        :param base_net_2: A second parent network on which mutation is based.
        :return: List of 2 Networks, both of which have features of both parent Networks.
        """
        dense_idx_1, weight_idx_1 = helpers_other.find_first_dense(base_net_1.model)
        dense_idx_2, weight_idx_2 = helpers_other.find_first_dense(base_net_2.model)
        dense_idx_1 -= 2
        dense_idx_2 -= 2

        conv_1 = Network(
            architecture=base_net_1.arch[:dense_idx_1] + base_net_2.arch[dense_idx_2:],
            opt=base_net_2.opt,
            activation=base_net_2.act,
            callbacks=base_net_2.callbacks
        )

        conv_2 = Network(
            architecture=base_net_2.arch[:dense_idx_2] + base_net_1.arch[dense_idx_1:],
            opt=base_net_1.opt,
            activation=base_net_1.act,
            callbacks=base_net_1.callbacks
        )

        conv_1.model.set_weights(  # Set Conv-Max weights
            base_net_1.model.get_weights()[:weight_idx_1] + conv_1.model.get_weights()[weight_idx_1:]
        )
        conv_1.model.set_weights(  # Set Dense-Drop weights
            conv_1.model.get_weights()[:weight_idx_1 + 1] + base_net_2.model.get_weights()[weight_idx_2 + 1:]
        )

        conv_2.model.set_weights(  # Set Conv-Max weights
            base_net_2.model.get_weights()[:weight_idx_2] + conv_2.model.get_weights()[weight_idx_2:]
        )
        conv_2.model.set_weights(  # Set Dense-Drop weights
            conv_2.model.get_weights()[:weight_idx_2 + 1] + base_net_1.model.get_weights()[weight_idx_1 + 1:]
        )
        return [conv_1, conv_2]

    @staticmethod
    def _mutate_parent_2(base_net_1, base_net_2):
        # type: (Network, Network) -> List[Network]
        """
        Creates two new Networks, both based on combination of given parents.
        More random than :ref:`main _mutate_parent<mutator.__Mutator#_mutate_parent>`.

        :param base_net_1: A first parent network on which mutation is based.
        :param base_net_2: A second parent network on which mutation is based.
        :return: List of 2 Networks, both of which have features of both parent Networks.
        """
        new_nets = []
        for _ in range(2):
            max_seq_start_idx = 0
            drop_seq_start_idx = helpers_other.find_first_dense(base_net_1.model)[0] - 2
            idx = 0
            max_seq_idx = []
            drop_seq_idx = []

            for l in base_net_1.arch:
                if helpers_other.arch_type(l) == 'max':
                    max_seq_idx.append((0, max_seq_start_idx, idx))
                    max_seq_start_idx = idx + 1
                elif helpers_other.arch_type(l) in ['drop', 'dense']:
                    if max_seq_start_idx != idx:
                        max_seq_idx.append((0, max_seq_start_idx, idx - 1))
                    break
                idx += 1

            for l in base_net_1.arch[idx:]:
                if helpers_other.arch_type(l) == 'drop':
                    drop_seq_idx.append((0, drop_seq_start_idx, idx))
                    drop_seq_start_idx = idx + 1
                idx += 1

            if helpers_other.arch_type(base_net_1.arch[-1]) != 'drop':
                drop_seq_idx.append((0, drop_seq_start_idx, len(base_net_1.arch) - 1))

            n_max_seq = [len(max_seq_idx)]
            n_drop_seq = [len(drop_seq_idx)]

            idx = 0
            max_seq_start_idx = 0
            drop_seq_start_idx = helpers_other.find_first_dense(base_net_2.model)[0] - 2

            for l in base_net_2.arch:
                if helpers_other.arch_type(l) == 'max':
                    max_seq_idx.append((1, max_seq_start_idx, idx))
                    max_seq_start_idx = idx + 1
                elif helpers_other.arch_type(l) in ['drop', 'dense']:
                    if max_seq_start_idx != idx:
                        max_seq_idx.append((1, max_seq_start_idx, idx - 1))
                    break
                idx += 1

            for l in base_net_2.arch[idx:]:
                if helpers_other.arch_type(l) == 'drop':
                    drop_seq_idx.append((1, drop_seq_start_idx, idx))
                    drop_seq_start_idx = idx + 1
                idx += 1
            if helpers_other.arch_type(base_net_2.arch[-1]) != 'drop':
                drop_seq_idx.append((1, drop_seq_start_idx, len(base_net_2.arch) - 1))

            n_max_seq = random.choice(n_max_seq + [len(max_seq_idx) - n_max_seq[0], int(len(max_seq_idx) / 2)])
            n_max_seq = max(1, n_max_seq)
            n_drop_seq = random.choice(n_drop_seq + [len(drop_seq_idx) - n_drop_seq[0], int(len(drop_seq_idx) / 2)])
            n_drop_seq = max(1, n_drop_seq)

            if debug:
                print('\n_parent_mutate_2')
                print('max_seq_idx: {}'.format(max_seq_idx))
                print('drop_seq_idx: {}'.format(drop_seq_idx))
                print('n_max_seq: {}'.format(n_max_seq))
                print('n_drop_seq: {}'.format(n_drop_seq))
                print('')

            archs = [base_net_1.arch, base_net_2.arch]
            new_arch = []

            max_idxs = []
            tmp = np.random.choice(np.arange(0, len(max_seq_idx), dtype='int'),
                                   size=n_max_seq, replace=n_max_seq <= len(max_seq_idx))
            for i in tmp:
                max_idxs.append(max_seq_idx[i])
            drop_idxs = []
            tmp = np.random.choice(np.arange(0, len(drop_seq_idx), dtype='int'),
                                   size=n_drop_seq, replace=n_drop_seq <= len(drop_seq_idx))
            for i in tmp:
                drop_idxs.append(drop_seq_idx[i])
            for i in max_idxs:
                a = archs[i[0]]
                new_arch += a[i[1]:i[2] + 1]

            for i in drop_idxs:
                a = archs[i[0]]
                new_arch += a[i[1]:i[2] + 1]

            new_net = Network(
                architecture=new_arch,
                callbacks=random.choice([base_net_1.callbacks, base_net_2.callbacks]),
                opt=random.choice([base_net_1.opt, base_net_2.opt]),
                activation=random.choice([base_net_1.act, base_net_2.act])
            )

            nets = [base_net_1, base_net_2]  # type: List[Network]

            if debug:
                print('\n_parent_mutate_2')
                print('Net 1: {}'.format(base_net_1.arch))
                print('Net 2: {}'.format(base_net_2.arch))
                print('New net: {}\n'.format(new_net.arch))

            idx = 1
            for i in max_idxs:
                a = nets[i[0]]
                if deep_debug:
                    print('\tmax {}'.format(i))
                    print('\trange {}-{}'.format(i[1] + 1, i[2] + 1))
                for j in range(i[1] + 1, i[2] + 1):
                    if deep_debug:
                        print('\t\t{}'.format(j))
                        print('\t\t{}'.format(new_net.model.get_layer(index=idx)))
                        print('\t\t{}'.format(a.model.get_layer(index=j)))
                        print('\t\tfilter {}'.format(np.array(a.model.get_layer(index=j).get_weights()[1]).shape))
                        print('\t\trest {}\n'.format(
                            np.array(new_net.model.get_layer(index=idx).get_weights()[0]).shape)
                        )
                    kernel_filter = a.model.get_layer(index=j).get_weights()[1]
                    new_weights = [new_net.model.get_layer(index=idx).get_weights()[0], kernel_filter]
                    new_net.model.get_layer(index=idx).set_weights(new_weights)
                    idx += 1
                idx += 1  # for MaxPool

            idx += 1  # Flatten
            for i in drop_idxs:
                a = nets[i[0]]
                if deep_debug:
                    print('\tdense {}'.format(i))
                    print('\trange {}-{}\n'.format(i[1] + 1, i[2] + 1))
                for j in range(i[1] + 2, i[2] + 2):
                    w_a = a.model.get_layer(index=j).get_weights()
                    w_n = new_net.model.get_layer(index=idx).get_weights()
                    if deep_debug:
                        print('\t\t{}'.format(j))
                        print('\t\t a_net layer {}'.format(a.model.get_layer(index=j)))
                        print('\t\t new_net layer {}'.format(new_net.model.get_layer(index=idx)))
                        print('\t\t len w_n[0]: {}'.format(len(w_n[0])))
                        print('\t\t len w_a[0]: {}'.format(len(w_a[0])))
                        print('')
                    new_weights = np.array(w_a[0][:len(w_n[0])])
                    if len(w_a[0]) < len(w_n[0]):
                        if deep_debug:
                            print(new_weights.shape)
                            print(np.array(w_n[0][len(new_weights):]).shape)
                        new_weights = np.concatenate((new_weights, w_n[0][len(new_weights):]), axis=0)
                    new_weights = [new_weights, w_a[1]]

                    new_net.model.get_layer(index=idx).set_weights(new_weights)
                    idx += 1
                idx += 1  # for Dropout

            new_nets += [new_net]
        return new_nets

    @staticmethod
    def _mutate_random(base_net, change_number_cap=3):
        # type: (Network, int) -> Network
        """
        Given a network, returns a new Network, with a random number of mutations (capped at given number).

        :param base_net: A network to which mutations should be based. It's not affected.
        :param change_number_cap: Maximal number of changes.
        :return: A new, mutated Network.
        """
        from helpers import helpers_mutate

        possible_changes = [
            helpers_mutate.add_dense_drop,
            helpers_mutate.remove_dense_drop,
            helpers_mutate.change_opt,
            helpers_mutate.change_activation,
            helpers_mutate.change_lr_schedule
        ]

        probabilities = [11, 7, 3, 3, 1]

        if len(input_shape.fget()) > 2:
            possible_changes += [
                helpers_mutate.add_conv_max,
                helpers_mutate.remove_conv_max,
            ]

            probabilities += [9, 7]

        probabilities = np.divide(probabilities, 1. * np.sum(probabilities))  # Normalization, for probabilities.

        # Number of changes is capped, and distributed exponentially.
        n_of_changes = int(1 + np.random.exponential())
        if n_of_changes > change_number_cap:
            n_of_changes = change_number_cap

        for i in range(n_of_changes):
            base_net = np.random.choice(possible_changes, p=probabilities)(base_net)

        return base_net
