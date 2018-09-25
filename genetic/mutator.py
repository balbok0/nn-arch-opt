import collections
import os
import random

import numpy as np
from keras import Model
from typing import *

from helpers.helpers_data import Array_Type
import log_save
from network import Network
from program_variables import program_params as const


class Mutator(object):
    def __init__(self,
                 population_size=10,  # type: int
                 starting_population=None,  # type: List[Network]
                 params=None,  # type: Dict
                 generator_f=None,  # type: function
                 generator_args=None  # type: Union[List, Dict]
                 ):
        # type: (...) -> None
        """
        Creates a new instance of Mutator.

        :param population_size: Size of population of networks in this generator. > 1
        :param starting_population: Optional. If you want to start from specified networks pass them here as list.
                It can have any length, but if more: only first 'population_size' will be used, and if less
                random networks will be generated to match 'population_size'.
        :param params: Optional. Parameters which specify possible modifications of networks.
        :param generator_f: Optional. If you want to change training dataset on each generation,
                than pass in function which can do generate it. Make sure bool(generator_f) evaluates to true.
        :param generator_args: Optional. If generator_f you passed in requires some arguments pass them in as list.
        """
        assert population_size > 1

        generator_args = generator_args or []
        starting_population = starting_population or []

        if params is not None:
            for i in ['kernel_size', 'conv_filers', 'dropout', 'dense_size', 'optimizer', 'optimizer_lr', 'activation']:
                assert i in list(params.keys()), "There should be %s in params keys." % i
                assert isinstance(params[i], list), "Key %s should be pointing to the list." % i

            const.mutations.fset(params)

        self.networks = starting_population[:population_size] or []  # type: List[Network]
        self.population_size = population_size

        self.__train_data_generator = None  # type: function
        self.__train_data_generator_arg = []
        if generator_f:
            self.set_dataset_generator(generator_f, generator_args)

    def evolve(self,
               x,  # type: Array_Type
               y,  # type: Array_Type
               validation_data=None,  # type: Tuple
               validation_split=None,  # type: float
               use_generator=False,  # type: bool
               generations=20,  # type: int
               save_all_nets=False,  # type: bool
               save_each_generation_best=True,  # type: bool
               saving_dir=None,  # type: str
               save_best=True,  # type: bool
               epochs=2,  # type: int
               initial_epoch=0,  # type: int
               batch_size=32,  # type: int
               shuffle='batch',  # type: str
               verbose=0  # type: int
               ):
        # type: (...) -> Model
        """
        Main function of Mutator.\n
        Trains neural networks, and evolves them through generations, in order to find architecture, optimizer, etc.
        which allow for the best results given a dataset.

        :param x: Training input to the network.
        :param y: Expected training output of the network.
        :param validation_data: Tuple of (Validation input, Validation output). Overrides validation_split.
        :param validation_split: float between 0 and 1. Overrode by validation data, if both are not None.
        :param use_generator: Whether to generate training data after each generation.
                Make sure generator is specified (set_dataset_generator, or specify generator_f in constructor).
        :param generations: Number of generations.
        :param save_all_nets: If true, saves all networks, after each generation. If true,  Overrides both save_each_generation.
        :param save_each_generation_best: If true, saves the best network from each generation
                in specified 'saving_dir'. If true, overrides save_best.
        :param save_best: If true, saves the best network at the end of lat generation.
                Overrode by save_each_generation_best.
        :param saving_dir: Directory to which models are saved.
        :param epochs: Number of epochs each network in each generation is trained.
        :param initial_epoch: Epoch at which training is supposed to get started.
        :param batch_size: Look at keras.models.Model.fit function documentation.
        :param shuffle: Look at keras.models.Model.fit function documentation.
        :param verbose: Look at keras.models.Model.fit function documentation.
                            Additionally, it specifies printing properties between generations.
        :return: Keras Model of neural network, which was the best after the last generation,
                    with already trained weights.
        """
        assert generations > 0

        if validation_data is None:
            if validation_split is None:
                raise AttributeError('Either validation_data or validation split cannot be None.')
            else:
                assert 0. < validation_split < 1.
                split = int(validation_split * len(x))
                validation_data = (x[split:], y[split:])
                x = x[:split]
                y = y[:split]

        const.output_shape.fset(len(y[0]))
        const.input_shape.fset(x.shape[1:])

        if len(self.networks) < self.population_size:
            self.networks = self.networks[:]
            for i in range(len(self.networks), self.population_size):
                self.networks.append(self.__create_random_model())

        if saving_dir is None:
            from program_variables.file_loc_vars import models_saving_dir
            saving_dir = models_saving_dir
        assert os.path.exists(saving_dir)

        if not saving_dir.endswith('/'):
            saving_dir += '/'

        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)

        scores = {}  # Needed to suppress warnings.

        log_save.print_message('')
        log_save.print_message('Started a new job.')

        for i in range(generations):
            if verbose > 0:
                print('\n\nGeneration %d' % (i + 1))

            log_save.print_message('Starting training for generation %d' % (i + 1))

            for _, net in enumerate(self.networks):
                if verbose > 0:
                    print('Network fit {}/{}'.format(_ + 1, len(self.networks)))
                    print(epochs)
                net.fit(x, y, validation_data, validation_split,
                        epochs=epochs, initial_epoch=initial_epoch,
                        batch_size=batch_size, shuffle=shuffle, verbose=verbose)

            log_save.print_message('Finished training for generation %d' % (i + 1))

            tmp_nets = []  # type: List[Network]
            tmp_scores = []  # type: List[float]
            best_net = None  # type: Network
            best_score = 0.  # type: float

            for net in self.networks:
                tmp_nets += [net]
                score = net.score(y_true=validation_data[1], y_score=net.predict(validation_data[0]))
                tmp_scores += [score]
                if best_net is None or score > best_score:
                    best_net = net
                    best_score = score

            # Saves all networks, after their training, and evaluation.
            if save_all_nets:
                gen_dir = '{}/gen_{:03d}/'.format(saving_dir, i + 1)
                if not os.path.exists(gen_dir):
                    os.makedirs(gen_dir)
                for j_i, j in enumerate(self.networks):
                    j.save('{}net_{:03d}'.format(gen_dir, j_i + 1))

            scores = collections.OrderedDict(zip(tmp_nets, tmp_scores))

            config = best_net.get_config()

            printing = {
                0: 'Generation {gen}. Best network scored: {score}'.format(gen=i + 1, score=scores[best_net]),
                1: 'Generation {gen}. Best network, with architecture: {arch}, optimizer {opt}, and activation '
                   'function {act}. It\'s score is {score}.'.format(
                    gen=i + 1,
                    arch=config['architecture'],
                    opt=config['optimizer'],
                    act=config['activation'],
                    score=scores[best_net]
                ),
                2: 'Generation {gen}. Best network, with architecture: {arch}, optimizer {opt}, and activation '
                   'function {act}. It\'s score is {score}.'.format(
                    gen=i + 1,
                    arch=config['architecture'],
                    opt=config['optimizer'],
                    act=config['activation'],
                    score=scores[best_net]
                )
            }
            log_save.print_message(printing[verbose])

            # Save the best net of current generation.
            if save_each_generation_best:
                best_net.save(file_path='{dir}net_best_from_{num:03d}.h5'.format(dir=saving_dir, num=i + 1))

            if not i + 1 == generations:
                self.__mutate_networks(scores, self.population_size)
                if use_generator:
                    if self.__train_data_generator:
                        x, y = self.__train_data_generator(self.__train_data_generator_arg)
                    else:
                        from warnings import warn
                        warn('Generator not used. Since it\'s not specified. If you have passed in a generator, check'
                             'how it evaluates to bool.')

        best = scores.popitem()
        best_score = best[1]
        best_net = best[0]
        if save_best:
            best_net.save(file_path='{dir}best_net.h5'.format(dir=saving_dir))
        config = best_net.get_config()

        log_save.print_message(
            'Best network, with architecture: '
            '{arch}, optimizer {opt}, callbacks {call}, and activation function {act}. '
            'It\'s score is {score}.'.format(
                arch=config['architecture'],
                opt=config['optimizer'],
                call=config['callbacks'],
                act=config['activation'],
                score=best_score
            )
        )
        return best_net.model

    def set_dataset_generator(self, f=None, f_args=None):
        f_args = f_args or []
        from helpers.helpers_data import prepare_data
        f = f or prepare_data
        self.__train_data_generator = f
        self.__train_data_generator_arg = f_args

    def __mutate_networks(self, scores, population_size):
        # type: (collections.OrderedDict[Network, float], int) -> None
        """
        Modifies list of networks in such a way, that bottom half is removed,
        and new networks are added instead of them, all based on networks from top half.

        :param scores: An OrderedDict, which points from the network to its score.
        :param population_size: Size, which population of networks should have at the end.
        """
        # Remove bottom half of population.
        for _ in range(int(population_size / 2)):
            scores.popitem(last=False)
        self.networks = list(scores.keys())
        p = np.exp(list(scores.values()))
        p = np.divide(p, sum(p))

        new_nets = []
        called_pairs = []  # type: List[List[Network, Network]]

        tmp_p = p
        tmp_nets = self.networks

        if len(const.input_shape.fget()) > 2:
            for _ in range(int(np.ceil((population_size - len(self.networks)) / 2))):
                if len(tmp_nets) >= 2:
                    pair = np.random.choice(tmp_nets, size=2, replace=False, p=tmp_p)
                    for i_pair in called_pairs:
                        if all(elem in pair for elem in i_pair):
                            for j in pair:
                                idx = list(tmp_nets).index(j)
                                tmp_p = np.concatenate((tmp_p[:idx], tmp_p[idx + 1:]), axis=0)
                                tmp_nets = np.concatenate((tmp_nets[:idx], tmp_nets[idx + 1:]), axis=0)
                            tmp_p = np.divide(tmp_p, sum(tmp_p))
                            _ -= 1

                    else:
                        called_pairs += [pair]
                        new_nets += Network.mutate(pair[0], pair[1])
                elif tmp_nets:
                    new_nets += [Network._mutate_random(tmp_nets[0])]
                    tmp_p = p
                    tmp_nets = self.networks

        else:
            for _ in range(population_size - len(self.networks)):
                new_nets += [Network._mutate_random(np.random.choice(tmp_nets, p=tmp_p), 5)]

        for net in new_nets[:population_size - len(self.networks)]:
            self.networks.append(net)

    def __create_random_model(self):
        # type: () -> Network
        """
        Creates a random model, based of parameters choices given in this Mutator.

        :return: Random Network, with a random architecture, optimizers, etc.
        """
        from helpers import helpers_mutate

        if len(const.input_shape.fget()) < 3:
            architecture = [random.choice(const.mutations.fget()['dense_size'])]
        else:
            architecture = [(random.choice(const.mutations.fget()['kernel_size']),
                             random.choice(const.mutations.fget()['conv_filters'])),
                            random.choice(const.mutations.fget()['dense_size'])]

        if const.max_depth != 0:
            n_layers = np.random.randint(const.min_depth, const.max_depth)
        else:
            n_layers = np.random.randint(const.min_depth, 20)

        if len(const.input_shape.fget()) >= 3:
            # Convolution/Maxout part of architecture
            while len(architecture) + const.n_conv_per_seq + 1 < \
                    int(n_layers * (1.0 * const.n_conv_per_seq) / (const.n_conv_per_seq + 2)):
                architecture = helpers_mutate.add_arch_conv_max(architecture)

        # Dense/Dropout part of architecture
        while len(architecture) + 2 < n_layers:
            architecture = helpers_mutate.add_arch_dense_drop(architecture)

        return Network(
            architecture=architecture,
            opt=random.choice(const.mutations.fget()['optimizer']),
            lr=random.choice(const.mutations.fget()['optimizer_lr']),
            activation=random.choice(const.mutations.fget()['activation'])
        )
