def main():
    from mutator import Mutator
    from helpers.helpers_data import prepare_data
    m = Mutator(population_size=10)
    (x, y), (x_val, y_val) = prepare_data('testing')
    print(m.evolve(x, y, validation_data=(x_val, y_val), batch_size=1000, verbose=1, epochs=1, generations=20))


if __name__ == '__main__':
    main()
