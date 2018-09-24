"""
All variables which start with '_' are properties.
However, all property functions are hidden at the bottom of the file.
"""
from typing import List

# Network parameters
# When any of them is set to 0, then this boundary is ignored.
max_n_weights = 7500000
max_depth = 10
min_depth = 3

# Mutation parameters
_mutations = {
    'kernel_size': [(3, 3), (5, 5)],
    'conv_filters': [8, 16],
    'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
    'dense_size': [16, 32, 64, 128],
    'optimizer': ['adam', 'sgd', 'nadam'],
    'optimizer_lr': [None, .0001, .0003, .001, .003, .01],
    'learning_decay_type': ['linear', 'exp'],
    'learning_decay_rate': [0.7, 0.8, 0.9],
    'activation': ['relu', 'sigmoid', 'tanh']
}

# Above this threshold mutations are parent ones, below are random. Range is (0, 1). Used in Mutator.
# If set to 1, all mutations are random. If set to 0, all mutations are parent.
parent_to_rand_chance = 0.3
parent_1_to_parent_2_chance = 0.35

# Dataset variables
n_train = 50000

# Development variables
debug = True
deep_debug = True

# ----------------------------------------------------------------------------------------


# Auto-params. DO NOT CHANGE
_max_limit = 0
_input_shape = []
_output_shape = 0


# Properties. DO NOT CHANGE.
@property
def max_layers_limit():
    # type: () -> int
    return _max_limit


@max_layers_limit.setter
def max_layers_limit(val):
    # type: (int) -> None
    global _max_limit
    if _max_limit < val:
        import warnings
        warnings.warn("Maximum number of MaxPool layers not changed."
                      "Tried to change to {}, while number of layers cannot be higher than {}".format(val, _max_limit))
    _max_limit = val


@property
def input_shape():
    # type: () -> List[int]
    return _input_shape


@input_shape.setter
def input_shape(val):
    # type: (List[int]) -> None
    import numpy as np
    global _input_shape
    _input_shape = val
    if len(val) > 1:
        global _max_limit
        _max_limit = int(np.log2(np.min(val[:-1])))


@property
def output_shape():
    # type: () -> int
    return _output_shape


@output_shape.setter
def output_shape(val):
    # type: (int) -> None
    global _output_shape
    _output_shape = val


@property
def mutations():
    return _mutations


@mutations.setter
def mutations(val):
    global _mutations
    _mutations = val


"""
-----------------------------------------------------------------------------------
Keras/TensorFlow correction vars
"""
