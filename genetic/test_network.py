from program_variables import program_params
from helpers.helpers_data import prepare_data
from deprecated import deprecated

x, y = prepare_data('testing', first_time=False)
program_params.input_shape.fset(x.shape[1:])
program_params.output_shape.fset(len(y[0]))
program_params.debug = True


@deprecated
def test_drop_last_in_arch():
    from network import Network

    n2 = Network(architecture=[((3, 3), 32), 'max', ((5, 5), 6), 32, 'drop0.4'], opt='sgd', activation='relu')
    # assert not (isinstance(n2.arch[-1], str) and n2.arch[-1].startswith('drop'))


def test_add_conv_max_seq():
    from helpers.helpers_mutate import add_conv_max
    from network import Network

    n2 = Network(architecture=[((3, 3), 32), 'max', ((4, 4), 32), ((4, 4), 32), 'max', 30, 30, 'drop0.4', 10])
    print(add_conv_max(n2, 3).arch)


def test_priv_add_conv_max_seq():
    from helpers.helpers_mutate import __add_conv_max
    from network import Network
    n2 = Network([((3, 3), 8), ((3, 3), 8), ((3, 3), 8), 'max', ((3, 3), 8), ((5, 5), 16), 16, 'drop0.70', 64,
                  'drop0.70', 32, 16, 'drop0.40', 16, 'drop0.70', 64, 'drop0.70', 32, 16, 'drop0.40'])
    print(__add_conv_max(n2, 0, 3, ((5, 5), 16)).arch)

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

    n2 = Network([((3, 3), 16), ((3, 3), 16), ((3, 3), 8), ((3, 3), 8), ((3, 3), 8), 'max', 64, 'drop0.50', 16,
                  'drop0.60', 128, 16, 'drop0.70', 16, 'drop0.40', 128, 'drop0.60'])
    print(__add_dense_drop(n2, 6, 128, 'drop0.30'))

    n2 = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 'drop0.4', 10])
    print(__add_dense_drop(n2, 4, 11, 'drop0.9').arch)
    print(__add_dense_drop(n2, 6, 11, 'drop0.9').arch)

    n2 = Network(architecture=[((7, 7), 16), 'max', 128])
    print(__add_dense_drop(n2, 2, 11, 'drop0.9').arch)

    n2 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    print(__add_dense_drop(n2, 11, 32, 'drop0.70').arch)


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


def test_priv_rmv_dense_drop_seq():
    from helpers.helpers_mutate import __remove_dense_drop
    from network import Network

    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((3, 3), 32), ((3, 3), 32), 'max', 10, 'drop0.2', 20, 'drop0.3', 10
    ])
    print(__remove_dense_drop(n2, 6).arch)
    print(__remove_dense_drop(n2, 8).arch)

    n2 = Network(architecture=[
        ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 8), ((5, 5), 8), ((3, 3), 16), ((3, 3), 16),
        ((3, 3), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 16), ((5, 5), 8), ((3, 3), 16),
        64, 'drop0.60', 32, 'drop0.50'
    ])
    print(__remove_dense_drop(n2, 19))
    print(__remove_dense_drop(n2, 21))


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


def test_quick():
    from network import Network
    import pandas as pd
    n1 = Network(
        [((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), ((5, 5), 16),
         ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max',
         ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 32, 16, 16]
    )
    n2 = Network(
        [((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), (
            (5, 5), 16), ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max',
         ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16),
         ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16),
         32, 16, 16, 32, 16, 64, 'drop0.40', 16, 32]
    )
    print(pd.Series(n1.arch))
    print(pd.Series(n2.arch))

    print([i.arch for i in Network._mutate_parent_2(n1, n2)])


def test_priv_mutate_parent_2():
    from network import Network
    import pandas as pd

    n1 = Network(
        [((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 'max', ((5, 5), 16),
         128, 16, 'drop0.40', 64, 'drop0.60', 64, 'drop0.50', 128, 'drop0.40', 128, 'drop0.40', 32, 'drop0.40', 128]
    )
    n2 = Network(
        [((3, 3), 8), ((3, 3), 8), ((3, 3), 8), 'max', ((3, 3), 8), 128, 16, 'drop0.40', 64, 'drop0.60', 64, 'drop0.50',
         128, 'drop0.40', 128, 'drop0.40', 32, 'drop0.40', 128]
    )

    print(pd.Series(n1.arch))
    print(pd.Series(n2.arch))

    print(Network._helper_parent_2(
        n1,
        n2,
        [(0, 4, 7), (1, 0, 3), (1, 0, 3), (0, 4, 7), (0, 4, 7), (0, 8, 8)],
        [(1, 12, 13), (1, 8, 9), (0, 16, 17), (1, 10, 11), (0, 14, 15), (1, 16, 17), (1, 5, 7)]
        ).arch
    )

    n1 = Network(
        [((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), ((5, 5), 16),
         ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max',
         ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 32, 16, 16]
    )
    n2 = Network(
        [((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16),
         ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max',
         ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16),
         ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 32, 16,
         16, 32, 16, 64, 'drop0.40', 16, 32]
    )

    print(pd.Series(n1.arch))
    print(pd.Series(n2.arch))

    print(Network._helper_parent_2(
        n1,
        n2,
        [(0, 0, 3), (0, 12, 16), (0, 17, 19), (1, 0, 6), (1, 16, 20), (1, 7, 15)],
        [(0, 20, 22), (1, 32, 38), (1, 32, 38), (1, 32, 38)]
        ).arch
    )

    n1 = Network(
        [((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), ((5, 5), 16),
         ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max',
         ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 32, 16, 16]
    )
    n2 = Network(
        [((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16),
         ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max',
         ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16),
         ((5, 5), 16), ((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((5, 5), 16), ((5, 5), 16), ((5, 5), 16), 32, 16,
         16, 32, 16, 64, 'drop0.40', 16, 32]
    )

    print(pd.Series(n1.arch))
    print(pd.Series(n2.arch))

    print(Network._helper_parent_2(
        n1,
        n2,
        [(1, 29, 31), (0, 12, 16), (1, 21, 28), (0, 17, 19), (1, 16, 20), (0, 4, 11)],
        [(1, 39, 40), (1, 32, 38), (1, 32, 38), (1, 32, 38)]
        ).arch
    )

    n1 = Network(
        [((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((5, 5), 8),
         128, 64, 'drop0.60', 64, 'drop0.40', 64, 'drop0.30', 64]
    )
    n2 = Network(
        [((5, 5), 8), ((5, 5), 8), ((5, 5), 8), 'max', ((3, 3), 16), 16, 16, 'drop0.30', 64, 'drop0.30', 64]
    )
    print(pd.Series(n1.arch))
    print(pd.Series(n2.arch))
    print(Network._helper_parent_2(
        n1,
        n2,
        [(1, 0, 3), (1, 4, 4), (0, 4, 7), (0, 8, 8), (0, 8, 8), (1, 4, 4)],
        [(0, 16, 16), (1, 10, 10), (0, 12, 13), (1, 5, 7)]
    ).arch
          )


def test_save_load_net():
    from network import Network
    import shutil

    n2 = Network([((3, 3), 8), ((3, 3), 8), ((3, 3), 8), 'max', 10, 11, 23])
    n2.save_network('test/test_net')
    n1 = Network.load_network('test/test_net')
    shutil.rmtree('test/')
    assert n1.arch == n2.arch


def test_save_load_model():
    from network import Network
    import shutil

    n2 = Network([((3, 3), 8), ((3, 3), 8), ((3, 3), 8), 'max', 10, 11, 23])
    n2.save_model('test/test_net')
    n1 = Network.load_model('test/test_net')
    shutil.rmtree('test/')
    print(n1.arch)
    print(n2.arch)
    assert n1.arch == n2.arch
