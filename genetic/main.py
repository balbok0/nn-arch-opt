def main():
    from mutator import Mutator
    from helpers.helpers_data import prepare_data
    m = Mutator(population_size=4)  # , generator_f=prepare_data, generator_args=['testing', False])
    (x, y), (x_val, y_val) = prepare_data('colorflow - Hgg_vs_Hqq 65', first_time=True)
    print(m.evolve(x, y, validation_data=(x_val, y_val), batch_size=100, verbose=1, generations=20,
                   use_generator=False))


if __name__ == '__main__':
    from helpers.helpers_data import prepare_data
    main()
