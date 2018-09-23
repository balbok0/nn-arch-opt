from network import Network


def check_drop_last_in_arch():
    n2 = Network(architecture=[((3, 3), 32), 'max', ((5, 5), 6), 32, 'drop0.4'], opt='sgd', activation='relu')
    assert not (isinstance(n2.arch[-1], str) and n2.arch[-1].startswith('drop'))


def check_add_conv_max_seq():
    from helpers.helpers_mutate import add_conv_max
    n2 = Network(architecture=[((3, 3), 32), 'max', ((4, 4), 32), ((4, 4), 32), 'max', 30, 30, 'drop0.4', 10])
    print(add_conv_max(n2, 3).arch)


def check_priv_add_conv_max_seq():
    from helpers.helpers_mutate import __add_conv_max
    n2 = Network(architecture=[((3, 3), 32), 'max', ((4, 4), 32), ((4, 4), 32), 'max', 30, 30, 'drop0.4', 10])
    print(__add_conv_max(n2, 0, 3, ((3, 3), 5)).arch)
    print(__add_conv_max(n2, 2, 3, ((3, 3), 5)).arch)
    print(__add_conv_max(n2, 5, 3, ((3, 3), 5)).arch)


def check_add_dense_drop_seq():
    from helpers.helpers_mutate import add_dense_drop
    n2 = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 'drop0.4', 10])
    print(add_dense_drop(n2).arch)
    n2 = Network(architecture=[((7, 7), 16), 'max', 128])
    print(add_dense_drop(n2).arch)


def check_priv_add_dense_drop_seq():
    from helpers.helpers_mutate import __add_dense_drop
    n2 = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 'drop0.4', 10])
    print(__add_dense_drop(n2, 3, 11, 'drop0.9').arch)
    print(__add_dense_drop(n2, 5, 11, 'drop0.9').arch)
    n2 = Network(architecture=[((7, 7), 16), 'max', 128])
    print(__add_dense_drop(n2, 1, 11, 'drop0.9').arch)


def check_rmv_conv_max_seq():
    from helpers.helpers_mutate import remove_conv_max
    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((5, 5), 32), ((5, 5), 32), 'max', ((4, 4), 32), ((4, 4), 32), 'max', 10, 'drop0.2', 10
    ])
    print(remove_conv_max(n2).arch)


def check_priv_rmv_conv_max_seq():
    from helpers.helpers_mutate import __remove_conv_max
    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((5, 5), 32), ((5, 5), 32), 'max', ((4, 4), 32), ((4, 4), 32), 'max', 10, 'drop0.2', 10
    ])
    print(__remove_conv_max(n2, 0, 1).arch)
    print(__remove_conv_max(n2, 1, 4).arch)
    print(__remove_conv_max(n2, 4, 7).arch)


def check_rmv_dense_drop_seq():
    from helpers.helpers_mutate import remove_dense_drop
    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((3, 3), 32), ((3, 3), 32), 'max', 10, 'drop0.2', 20, 'drop0.3', 10
    ])
    print(remove_dense_drop(n2).arch)


def check_priv_dense_drop_seq():
    from helpers.helpers_mutate import __add_dense_drop
    n2 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    print(__add_dense_drop(n2, 11, 32, 'drop0.70').arch)


def check_mutate_parent():
    n1 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    n2 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    print([i.arch for i in Network._mutate_parent(n1, n2)])


def check_mutate_parent_2():
    n1 = Network(
        [((3, 3), 16), 32, 'drop0.30', 128, 'drop0.30', 32]
    )
    n2 = Network(
        [((3, 3), 16), 32, 'drop0.30', 32]
    )
    print([i.arch for i in Network._mutate_parent_2(n1, n2)])
    n1 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    n2 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    print([i.arch for i in Network._mutate_parent_2(n1, n2)])


def main():
    """
    Always calls all the functions in this file, in alphabetical order. Does not call itself.
    """
    from program_variables import program_params
    from helpers.helpers_data import prepare_data

    x, y = prepare_data('testing', first_time=False)
    program_params.input_shape.fset(x.shape[1:])
    program_params.output_shape.fset(len(y[0]))
    program_params.debug = True

    import sys
    import inspect

    current_module = sys.modules[__name__]
    all_functions = inspect.getmembers(current_module, inspect.isfunction)
    for key, value in all_functions:
        try:
            args = inspect.getfullargspec(value).args
        except AttributeError:
            # noinspection PyDeprecation
            args = inspect.getargspec(value).args
        if args == [] and not key == 'main':
            print('NEW TEST: {}\n\n'.format(key))
            value()
            print('\nEND TEST: {}\n\n\n\n'.format(key))


if __name__ == '__main__':
    main()
