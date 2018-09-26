from program_variables import program_params
from helpers.helpers_data import prepare_data

x, y = prepare_data('testing', first_time=False)
program_params.input_shape.fset(x.shape[1:])
program_params.output_shape.fset(len(y[0]))
program_params.debug = True


def test_drop_last_in_arch():
    from network import Network

    n2 = Network(architecture=[((3, 3), 32), 'max', ((5, 5), 6), 32, 'drop0.4'], opt='sgd', activation='relu')
    assert not (isinstance(n2.arch[-1], str) and n2.arch[-1].startswith('drop'))


def test_add_conv_max_seq():
    from helpers.helpers_mutate import add_conv_max
    from network import Network

    n2 = Network(architecture=[((3, 3), 32), 'max', ((4, 4), 32), ((4, 4), 32), 'max', 30, 30, 'drop0.4', 10])
    print(add_conv_max(n2, 3).arch)


def test_priv_add_conv_max_seq():
    from helpers.helpers_mutate import __add_conv_max
    from network import Network

    n2 = Network(architecture=[((3, 3), 32), 'max', ((4, 4), 32), ((4, 4), 32), 'max', 30, 30, 'drop0.4', 10])
    print(__add_conv_max(n2, 0, 3, ((3, 3), 5)).arch)
    print(__add_conv_max(n2, 2, 3, ((3, 3), 5)).arch)
    print(__add_conv_max(n2, 5, 3, ((3, 3), 5)).arch)


def test_add_dense_drop_seq():
    from helpers.helpers_mutate import add_dense_drop
    from network import Network

    n2 = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 'drop0.4', 10])
    print(add_dense_drop(n2).arch)
    n2 = Network(architecture=[((7, 7), 16), 'max', 128])
    print(add_dense_drop(n2).arch)


def test_priv_add_dense_drop_seq():
    from helpers.helpers_mutate import __add_dense_drop
    from network import Network

    n2 = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 'drop0.4', 10])
    print(__add_dense_drop(n2, 3, 11, 'drop0.9').arch)
    print(__add_dense_drop(n2, 5, 11, 'drop0.9').arch)
    n2 = Network(architecture=[((7, 7), 16), 'max', 128])
    print(__add_dense_drop(n2, 1, 11, 'drop0.9').arch)


def test_rmv_conv_max_seq():
    from helpers.helpers_mutate import remove_conv_max
    from network import Network

    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((5, 5), 32), ((5, 5), 32), 'max', ((4, 4), 32), ((4, 4), 32), 'max', 10, 'drop0.2', 10
    ])
    print(remove_conv_max(n2).arch)


def test_priv_rmv_conv_max_seq():
    from helpers.helpers_mutate import __remove_conv_max
    from network import Network

    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((5, 5), 32), ((5, 5), 32), 'max', ((4, 4), 32), ((4, 4), 32), 'max', 10, 'drop0.2', 10
    ])
    print(__remove_conv_max(n2, 0, 1).arch)
    print(__remove_conv_max(n2, 1, 4).arch)
    print(__remove_conv_max(n2, 4, 7).arch)


def test_rmv_dense_drop_seq():
    from helpers.helpers_mutate import remove_dense_drop
    from network import Network

    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((3, 3), 32), ((3, 3), 32), 'max', 10, 'drop0.2', 20, 'drop0.3', 10
    ])
    print(remove_dense_drop(n2).arch)


def test_priv_dense_drop_seq():
    from helpers.helpers_mutate import __add_dense_drop
    from network import Network

    n2 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    print(__add_dense_drop(n2, 11, 32, 'drop0.70').arch)


def test_mutate_parent():
    from network import Network

    n1 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    n2 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    print([i.arch for i in Network._mutate_parent(n1, n2)])


def test_mutate_parent_2():
    from network import Network

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

    n1 = Network(
        [((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 'max', ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((3, 3), 16),
         32, 32, 'drop0.60', 16, 'drop0.50', 128]
    )
    n2 = Network(
        [((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 8),
         128, 64, 'drop0.70', 128, 'drop0.30', 128]
    )
    print([i.arch for i in Network._mutate_parent_2(n1, n2)])

    n1 = Network(
        [((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 'max', ((5, 5), 16),
         128, 16, 'drop0.40', 64, 'drop0.60', 64, 'drop0.50', 128, 'drop0.40', 128, 'drop0.40', 32, 'drop0.40', 128]
    )
    n2 = Network(
        [((3, 3), 8), ((3, 3), 8), ((3, 3), 8), 'max', ((3, 3), 8), 128, 16, 'drop0.40', 64, 'drop0.60', 64, 'drop0.50',
         128, 'drop0.40', 128, 'drop0.40', 32, 'drop0.40', 128]
    )
    print([i.arch for i in Network._mutate_parent_2(n1, n2)])
