from typing import *
from collections import defaultdict
import random
import numpy as np
import program_variables.program_params as const
from keras.utils.io_utils import HDF5Matrix

Array_Type = Union[HDF5Matrix, np.ndarray]
XY_Type = Union[Array_Type, Union[Array_Type]]


def get_memory_size(hdf5_data_set, n_samples=None):
    # type: (Union[HDF5Matrix, np.ndarray], int) -> int
    """
    Approximates memory cost of a given number of images in a hdf5 dataset.

    :param hdf5_data_set: Dataset, from which images are used.
    :param n_samples: Number of images in a given dataset which memory cost is to be approximated.
                        If None, number of images equals size of hdf5_data_set.
    :return: Approximated size of whole array in RAM memory.
    """
    if n_samples is None:
        n_samples = len(hdf5_data_set)
    first = hdf5_data_set[0][()]
    return n_samples * first.size * first.itemsize


def __get_masks(x_shape, y):
    # type: (Tuple[int], np.ndarray) -> (np.ndarray, np.ndarray)
    """
    Creates masks, which choose n_train random images after applying a mask.

    :param x_shape: Shape of x dataset (images).
    :param y: True classes corresponding to images of dataset, which shape is given in x_shape.
    :return: Two masks, first one for x part of dataset (images), another for y part of dataset (classes)
    """

    all_indexes = defaultdict(list)  # type: Dict[int, List[int]]
    for i in range(len(y)):
        curr = int(y[i])
        all_indexes[curr].append(i)

    ratios = defaultdict()  # type: Dict[int, float]

    for i, j in all_indexes.items():
        ratios[i] = (len(j) * 1. / len(all_indexes[0]))

    # Ratios split the whole dataset to ratios given class and first class.
    # Part scales these ratios up, so that, 'part' corresponds to size of first class.
    part = const.n_train * 1. / sum(ratios.values())
    if part == 0:  # n_train is 0.
        part = len(y) * 1. / sum(ratios.values())

    # Masks of what to keep.
    indexes_x = np.full(shape=x_shape, fill_value=False, dtype=bool)
    indexes_y = np.full(shape=y.shape, fill_value=False, dtype=bool)

    for i in all_indexes.keys():
        chosen_idxs = random.sample(all_indexes[i], int(part * ratios[i]))
        indexes_y[chosen_idxs] = True
        indexes_x[chosen_idxs, ...] = True

    return indexes_x, indexes_y


def prepare_data(dataset: str='colorflow', first_time: bool=True) -> Tuple:
    """
    Prepares a dataset of a choice, and returns it in form of pair of tuples, containing training and validation
    data-sets.

    :param dataset: Name of the dataset, valid arguments are:

    * cifar10   - 'cifar' or 'cifar10'
    * mnist     - 'mnist'
    * testing   - 'testing' - a smaller colorflow dataset, for debug purposes.
    * colorflow - 'colorflow', 'cf', followed by '-', and name of more specific dataset from:
        * Signal vs. Background:
           * 'Herwig Dipole'
           * 'Herwig Angular'
           * 'Sherpa'
           * 'Pythia Standard'
           * 'Pythia Vincia'
        * Signal 1 vs. Signal 2:
           * 'Hgg_vs_Hqq'

        If none are given default is 'Herwig Angular'.

    :param first_time: Whether a validation dataset should be returned too, or not.
        If called for the first time, should be 'True'.
        If not, can be avoided for better performance.

    :return: (x_train, y_train), (x_val, y_val),
               each being of type np.ndarray, or HDF5Matrix, depending on memory space.

    * x_train - is a input to nn, on which neural network can be trained.
    * y_train - are actual results, which compared to output of nn, allow it to learn information about data.
    * x_val   - is a input to nn, on which nn can be checked how well it performs.
    * y_val   - are actual results, against which nn can be checked how well it performs.

    """
    if isinstance(dataset, str):
        name = dataset.lower().split('-')[0]

        # Needed for typing.
        (_, _), (x_val, y_val) = (None, None), (None, None)  # type: Array_Type
        if name in ['cifar', 'cifar10']:
            from keras.datasets import cifar10
            from keras.utils.np_utils import to_categorical

            (x_train, y_train), (x_val, y_val) = cifar10.load_data()
            y_train = to_categorical(y_train)
            y_val = to_categorical(y_val)

        elif name == 'mnist':
            from keras.datasets import mnist
            from keras.utils.np_utils import to_categorical

            (x_train, y_train), (x_val, y_val) = mnist.load_data()  # type: Array_Type
            x_train = np.reshape(x_train, list(np.array(x_train).shape) + [1])
            x_val = np.reshape(x_val, list(np.array(x_val).shape) + [1])
            y_train = to_categorical(y_train)
            y_val = to_categorical(y_val)

        elif name == 'testing':
            from keras.datasets import cifar10
            from keras.utils.np_utils import to_categorical

            (x_train, y_train), (x_val, y_val) = cifar10.load_data()  # type: Array_Type
            y_train = to_categorical(y_train[:2500])
            y_val = to_categorical(y_val[:2500])
            x_train = x_train[:2500, ...]
            x_val = x_val[:2500, ...]

        elif name in ['cf', 'colorflow']:
            from get_file_names import get_ready_path

            import psutil
            import h5py as h5

            from keras.utils.io_utils import HDF5Matrix
            from keras.utils.np_utils import to_categorical

            fname = dataset.split('-')[1].strip() if len(dataset.split('-')) > 1 else 'Herwig Angular'
            fname = get_ready_path(fname)

            # Data loading
            with h5.File(fname) as hf:
                n_classes = len(np.unique(hf['train/y']))

                # Cap of training images (approximately).
                memory_cost = 122 * 4  # Buffer for creating np array
                memory_cost += get_memory_size(hf['train/x'], const.n_train)
                memory_cost += 2 * get_memory_size(hf['train/y'], const.n_train)

                indexes_x, indexes_y = __get_masks(hf['train/x'].shape, hf['train/y'][()])

            x_sing_shape = list(indexes_x.shape[1:])

            # Available memory for training
            if memory_cost < psutil.virtual_memory().available - psutil.virtual_memory().total * 0.15:
                with h5.File(fname) as hf:
                    x_train = np.array([])  # type: Array_Type
                    for i in range(int(len(hf['train/x'])/const.n_train) + 1):
                        x_train = np.concatenate((x_train,
                                                  hf['train/x'][i * const.n_train: (i + 1) * const.n_train]
                                                  [indexes_x[i * const.n_train: (i + 1) * const.n_train]]))
                    x_train = np.reshape(x_train, [int(len(x_train) / np.prod(x_sing_shape))] + x_sing_shape)
                    y_train = to_categorical(hf['train/y'][indexes_y], n_classes)  # type: Array_Type

            else:  # data too big for memory.
                x_train = HDF5Matrix(fname, 'train/x')[indexes_x]  # type: Array_Type
                x_train = np.reshape(x_train, [int(len(x_train) / np.prod(x_sing_shape))] + x_sing_shape)
                y_train = to_categorical(HDF5Matrix(fname, 'train/y')[indexes_y], n_classes)  # type: Array_Type

            if first_time:
                with h5.File(fname) as hf:
                    # Cap of training images (approximately).
                    memory_cost = 122 * 4  # Buffer for creating np array
                    memory_cost += get_memory_size(hf['val/x'])
                    memory_cost += 2 * get_memory_size(hf['val/y'])

                # Available memory for validation.
                if memory_cost < psutil.virtual_memory().available - psutil.virtual_memory().total * 0.15:
                    with h5.File(fname) as hf:
                        x_val = hf['val/x'][()]
                        y_val = to_categorical(hf['val/y'], n_classes)[()]

                else:  # data too big for memory.
                    x_val = HDF5Matrix(fname, 'val/x')
                    y_val = to_categorical(HDF5Matrix(fname, 'val/y'), n_classes)

                p = np.random.permutation(len(y_val))
                y_val = y_val[p]
                x_val = x_val[p]

            p = np.random.permutation(len(y_train))
            y_train = y_train[p]
            x_train = x_train[p]

        else:
            raise AttributeError('Invalid name of dataset.')

        if first_time:
            return (x_train, y_train), (x_val, y_val)
        else:
            return x_train, y_train
