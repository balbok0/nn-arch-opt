import warnings

import keras
import numpy as np
from keras.activations import softmax, softplus, softsign, elu, relu, selu, linear, tanh, sigmoid, hard_sigmoid
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Layer, MaxPool2D
from keras.models import Sequential, Model
from keras.callbacks import Callback
from scipy import interp
from sklearn.metrics import roc_curve, auc
from typing import List, Tuple, Union

import program_variables.program_params as const


activations_function_calls = {
    'softmax': softmax,
    'elu': elu,
    'selu': selu,
    'softplus': softplus,
    'softsign': softsign,
    'relu': relu,
    'tanh': tanh,
    'sigmoid': sigmoid,
    'hard_sigmoid': hard_sigmoid,
    'linear': linear
}


def get_number_of_weights(model):
    # type: (Model) -> int
    """
    :param model: A model which number of weights is to be determined.
    :return: int - Number of weights in a given model.
    """
    num_weights = 0
    for l in model.get_weights():
        _l = np.ravel(l)
        num_weights += len(_l)
    return num_weights


def multi_roc_score(y_true, y_score):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Returns area under ROC curve value for a multi-class dataset. It's a 1-D scalar value.

    :param y_true: True classes of a dataset. (In one-hot vector form)
    :param y_score: Predicted classes for the same dataset. (Also in one-hot vector form)
    :return: Area under curve, based on interpolated averaged true positive rates and actual false positive rates.
    """

    n_classes = len(y_true[0])
    gen_fpr = {}
    gen_tpr = {}
    gen_roc_auc = {}
    for i in range(n_classes):
        gen_fpr[i], gen_tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        gen_roc_auc[i] = auc(gen_fpr[i], gen_tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([gen_fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, gen_fpr[i], gen_tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr = all_fpr
    tpr = mean_tpr
    auc_score = auc(fpr, tpr)

    return auc_score


def assert_model_arch_match(model, arch):
    # type: (Model, List) -> bool
    """
    Asserts that given architecture and given model match. Specific for THIS Network implementation!

    :param model: A keras neural network model, which architecture is to be compared.
    :param arch: A list containing descriptions of layers to be compared.
    :return: True, if both arguments match, False otherwise.
    """
    arch_idx = 0
    for l in model.layers[1:-1]:  # type: Layer
        if isinstance(l, (Activation, Flatten)):
            arch_idx -= 1
        else:
            if not arch[arch_idx] in layer_to_arch(l):
                if const.debug:
                    print('assert_model_arch_match:')
                    print('Arch: {}'.format(arch))
                    print('Arch idx: {}'.format(arch_idx))
                    for i in range(len(model.layers)):
                        print('\t{}\t{}'.format(i, model.layers[i].get_config()))
                    print('Arch at arch idx: {}'.format(arch[arch_idx]))
                    print('Layer at that corresponding place in model layers: {}'.format(layer_to_arch(l)))
                    print('Config of that layer {}'.format(l.get_config()))
                    print('')
                return False

        arch_idx += 1
    return True


def find_first_dense(model):
    # type: (Model) -> (int, int)
    """
    Finds an index of first dense layer in a given model.

    :param model: A model in which index of the first dense layer is to be found.
    :return: Tuple containing two ints: <br>
               (index of first dense layer in layer count, index of first dense layer in weights count). <br>
               Returns (None, None) if no dense layer exists in the model.
    """
    layer_idx = 0
    weight_idx = 0
    for l in model.layers:
        if isinstance(l, Dense):
            break
        layer_idx += 1
        weight_idx += len(l.get_weights())
    else:
        return None, None
    return layer_idx, weight_idx


def _insert_layer(model, layer, index):
    # type: (Model, Layer, int) -> Model
    """
    Immutable. Creates copy of a model, adds a layer at a given index, and returns it.

    :param model: A model, which copy will be modified and returned.
    :param layer: A layer to be inserted.
    :param index: An index at which layer is supposed to be inserted at. > 0 and < len(layers) - 1
    :return: A new model, with specified layer inserted at specified index. Not complied yet.
    """
    # Copy of the whole model. Deep copy, due to safety.
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())

    result = Sequential()

    # Such a deep copy is needed, so that input_shape is not specified, and layers are not shared.
    for l in model_copy.layers[:index]:
        result.add(__clone_layer(l))
    result.add(layer)
    for l in model_copy.layers[index:]:
        result.add(__clone_layer(l))

    if const.deep_debug:
        print('_insert layer')
        print('layer to be added {} at index {}'.format(layer, index))
        print('base model weights shape:')
        for i in model_copy.get_weights():
            print('\t%d' % len(i))

        print('resulting model weights shape:')
        for i in result.get_weights():
            print('\t%d' % len(i))
        print('')

    # Needed to clone weights.
    weight_number_before = 0
    for l in model_copy.layers[:index]:
        weight_number_before += len(l.get_weights())

    weight_number_after = weight_number_before + len(layer.get_weights())
    if index < len(model_copy.layers):
        weight_number_after += len(model_copy.layers[index].get_weights())

    if isinstance(layer, MaxPool2D):
        # MaxPool changes shape of the output, thus weights will not have the same shape.
        new_weights = model_copy.get_weights()[:weight_number_before]

        _, first_dense = find_first_dense(result)
        new_weights += result.get_weights()[weight_number_before:first_dense + 1]  # New weights, shape changed.
        new_weights += model_copy.get_weights()[first_dense + 1:]  # Back to old shape, since Dense resets it.

    else:
        new_weights = model.get_weights()[:weight_number_before]
        if (index < len(model_copy.layers) and isinstance(model_copy.layers[index], Flatten)) or \
                (index < len(model_copy.layers) - 1 and isinstance(model_copy.layers[index], MaxPool2D) and
                    isinstance(model_copy.layers[index+1], Flatten)):
            _, first_dense = find_first_dense(result)

            if const.deep_debug:
                print('_insert_layer if path chosen')
                print('new_weights (part 1) model weights shape:')
                for i in new_weights:
                    print('\t%d' % len(i))
                print('')

            new_weights += result.get_weights()[weight_number_before:first_dense + 1]  # New weights, shape changed.

            if const.deep_debug:
                print('new_weights (part 2) model weights shape:')
                for i in new_weights:
                    print('\t%d' % len(i))
                print('')

            new_weights += model_copy.get_weights()[first_dense - 1:]  # Back to old shape, since Dense resets it.

            if const.deep_debug:
                print('new_weights (part 3) model weights shape:')
                for i in new_weights:
                    print('\t%d' % len(i))
                print('')
        else:
            if const.deep_debug:
                print('_insert_layer else path chosen')
                print('new_weights (part 1) model weights shape:')
                for i in new_weights:
                    print('\t%d' % len(i))
                print('')

            new_weights += result.get_weights()[weight_number_before:weight_number_after + 1]

            if const.deep_debug:
                print('new_weights (part 2) model weights shape:')
                for i in new_weights:
                    print('\t%d' % len(i))
                print('')

            if index >= len(model_copy.layers):
                new_weights += model_copy.\
                    get_weights()[weight_number_before + 1:]

            else:
                new_weights += model_copy.\
                    get_weights()[weight_number_before + len(model_copy.layers[index].get_weights()) + 1:]

            if const.deep_debug:
                print('new_weights (part 3) model weights shape:')
                for i in new_weights:
                    print('\t%d' % len(i))
                print('')

    result.set_weights(new_weights)
    return result


def _remove_layer(model, index):
    # type: (Model, int) -> Model
    """
    Immutable. Creates copy of a model, removes a layer at a given index, and returns it.

    :param model: A model, which copy will be modified and returned.
    :param index: index of layer to be removed. > 0 < len(layers) - 1
    :return: A new model, with a layer at specified index removed. Not compiled yet.
    """
    # Copy of the whole model. Deep copy, due to safety.
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())

    layer = model_copy.layers[index]  # type: Layer

    warnings.warn("Since a layer at a chosen index is changing shape of data, model doesn't have to compile.") \
        if isinstance(layer, Flatten) else None

    result = Sequential()

    # Such a deep copy is needed, so that input_shape is not specified, and layers are not shared.
    for l in model_copy.layers[:index]:
        result.add(__clone_layer(l))

    for l in model_copy.layers[(index+1):]:
        result.add(__clone_layer(l))

    if const.deep_debug:
        print('_remove_layer')
        print('layer to be removed at index {}'.format(index))
        print('base model weights shape:')
        for i in model_copy.get_weights():
            print('\t%d' % len(i))

        print('resulting model weights shape:')
        for i in result.get_weights():
            print('\t%d' % len(i))
        print('')

    # Needed to clone weights.
    weight_number_before = 0
    for l in model_copy.layers[:index]:
        weight_number_before += len(l.get_weights())
    layer_weight_width = len(model_copy.layers[index].get_weights())
    weight_number_after = weight_number_before + layer_weight_width

    if isinstance(layer, MaxPool2D):
        # MaxPool changes shape of the output, thus weights will not have the same shape.
        new_weights = model_copy.get_weights()[:weight_number_before]
        _, first_dense = find_first_dense(result)
        new_weights += result.get_weights()[weight_number_before:first_dense + 1]
        new_weights += model_copy.get_weights()[first_dense + 1:]
    else:
        new_weights = model_copy.get_weights()[:weight_number_before]
        new_weights += result.get_weights()[weight_number_before:int(weight_number_after - (layer_weight_width / 2))]
        new_weights += model_copy.get_weights()[int(weight_number_after + (layer_weight_width / 2)):]

    result.set_weights(new_weights)
    return result


def __clone_layer(layer):
    # type: (Layer) -> Layer
    """
    Clones a layer, with the exact same configuration as a given one.
        It removes variables not mentioned in configuration. In example:
        input shape, which is given due to adding layer to model.

    :param layer: Layer to be cloned.
    :return: An exact copy of a given layer, only based on configuration (no additional information is used).
    """
    return type(layer).from_config(layer.get_config())


def can_add_max_number(arch):
    # type: (List[str, int, Tuple, List]) -> bool
    """
    Returns whether MaxPool can be added to Network represented with such arch.

    :param arch: Representation of Network.
    :return: True, if adding a MaxPool layer is legal, False otherwise.
    """
    n = 0
    for i in arch:
        if arch_type(i) == 'max':
            n += 1
    return n < const.max_layers_limit.fget() != 0


def arch_type(layer):
    # type: (Union[List, str, int, Tuple]) -> str
    """
    Given arch description of layer, returns str representation of it's type.

    :param layer: Arch representation of layer.
    :return: String type of that layer.
    """
    if hasattr(layer, '__getitem__') and not isinstance(layer, str):
        return 'conv'

    elif isinstance(layer, int):
        return 'dense'

    elif isinstance(layer, str) and layer.lower() in ['m', 'max', 'maxout', 'maxpool']:
        return 'max'

    elif isinstance(layer, str) and layer.lower().startswith(('drop', 'dropout')):
        return 'drop'
    else:
        raise BaseException('Illegal form of argument layer. '
                            'Please modify fix the argument, or modify this file, to allow different args.')


def arch_to_layer(layer, activation):
    # type: (Union[str, Tuple, int], str) -> Layer
    """
    Given an architecture layer description, and an activation function, returns a new layer based on them.

    :param layer: Layer description. Specific to THIS Network implementation.
    :param activation: Activation function of a layer to be created.
    :return: A new layer based on given description and activation.
    """
    layer_type = arch_type(layer)

    if layer_type == 'conv':
        return Conv2D(filters=layer[1], kernel_size=layer[0], activation=activation, kernel_initializer='he_uniform',
                      padding='same')

    elif layer_type == 'dense':
        return Dense(units=layer, activation=activation)

    elif layer_type == 'max':
        return MaxPool2D()

    elif layer_type == 'drop':
        if layer.lower().startswith('dropout'):
            return Dropout(rate=float(layer[7:]))
        else:
            return Dropout(rate=float(layer[4:]))
    else:
        raise BaseException('Illegal form of argument layer. '
                            'Please modify fix the argument, or modify this file, to allow different args.')


def layer_to_arch(layer):
    # type: (Layer) -> list
    """
    Given a keras layer returns an architecture layer description. Specific to THIS Network implementation.

    :param layer: Keras layer, which is to be translated to architecture layer description.
    :return: An architecture layer description based on given layer.
    """
    if isinstance(layer, Conv2D):
        return [(layer.get_config()['kernel_size'], layer.get_config()['filters'])]
    elif isinstance(layer, MaxPool2D):
        return ['max']
    elif isinstance(layer, Dropout):
        return ['drop%.2f' % layer.get_config()['rate'], 'dropout%.2f' % layer.get_config()['rate'],
                'drop%.2g' % layer.get_config()['rate'], 'dropout%.2g' % layer.get_config()['rate']]
    elif isinstance(layer, Dense):
        return [layer.get_config()['units']]
    else:
        return [None]


def clone_model(base_model, new_act, new_opt):
    # type: (Model, str, Union[str, keras.optimizers.Optimizer]) -> keras.models.Sequential
    model = keras.models.clone_model(base_model)
    act = activations_function_calls[new_act]

    while isinstance(model.layers[-2], Dropout):
        model = _remove_layer(model, len(model.layers) - 2)

    prev = None
    idx = 1
    for l in model.layers[1:-1]:  # type: Layer
        if isinstance(prev, (MaxPool2D, Dropout)) and isinstance(prev, type(l)):
            model = _remove_layer(model, idx)
        else:
            prev = l
            idx += 1
            if not isinstance(l, (Activation, MaxPool2D, Flatten, Dropout)) and not isinstance(l.activation, type(act)):
                l.activation = act

    model.set_weights(base_model.get_weights())

    model.compile(optimizer=new_opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def is_running_gpu():
    # type: () -> bool
    from tensorflow.python.client import device_lib
    return 'CPU' in [x.device_type for x in device_lib.list_local_devices()]


def find_first_drop_dense_arch(arch):
    # type: (List) -> int
    for idx, i in enumerate(arch):
        if arch_type(i) in ['dense', 'drop']:
            return idx


class NaNSafer(Callback):

    def __init__(self):
        Callback.__init__(self)  # super call in such way for compatibility reasons between python 2 and 3
        self.weights = None

    def on_epoch_begin(self, epoch, logs=None):
        # type: (int, dict) -> None
        self.weights = self.model.get_weights()

    def on_batch_end(self, batch, logs=None):
        # type: (int, dict) -> None
        logs = logs or {}
        if 'loss' in logs.keys():
            if 'nan' == str(logs['loss']):
                self.model.set_weights(self.weights)
                self.model.stop_training = True
