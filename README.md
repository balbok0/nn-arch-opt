# Neural Network Architecture Optimization
This repository allows one to find approximation of the 
best neural network architecture given 1D, 2D or 3D input. <br>
It also depends on parameters which user can set on their own, but there's default ones.
Parameters describe possible choices of number of neurons in a layer, callbacks etc.

## Requirements
```
pip install --user keras numpy sklearn typing h5py warnings datetime tqdm collections deprecated
```

## Basic usage
In order to use: 
1. Clone/Download this repository and cd into it.
1. Prepare data x and y 
(you can use `from genetic import prepare_data`, or pass in your own data-set).
1. Create a mutator given a population size `from genetic import Mutator`
1. In order to evolve call Mutator's function `evolve` with your x y dataset 
and either validation_split or validation_data argument.
* Please make sure y is in binary class matrix form. `keras.utils.np_utils.to_categorical`
enables to change integer vector to this form.
* If you want to use a different data training set on each generation, you can pass in
`generator_f` and `generator_args` to Mutator constructor, or call Mutator's `set_dataset_generator`.
### Example 1

```
from genetic import prepare_data, Mutator

x, y = prepare_data('testing', first_time=False)
m = Mutator(population_size=5)
m.evolve(x, y, validation_split=0.6, generations=5)
```

### Example 2 - with generator
```
from genetic import prepare_data, Mutator

(x, y), (x_val, y_val) = prepare_data('testing')
m = Mutator(population_size=5, generator_f=prepare_data, generator_args=['testing', False])
m.evolve(x, y, validation_data=(x_val, y_val), generations=5)
```
## More options
* There is an option of starting with preset Networks as starting population.
However they need to be initialized by user (`from genetic import Network`), 
and passed in to Mutator constructor as `starting_population=`.
    * Please note that if you are creating network from architecture,
    you have to look at how architecture describes different layer.
    * Also, when a Network, is created from just architecture, random weights are initialized.
    To avoid that you can pass in `copy_model=` with `keras.Model.Sequential`, or use 
    `keras.Model.set_weights` function.
* If you want to use different parameters than default, you can either pass them in Mutator
constructor, or change them file `genetic/program_variables/program_params.py` under `_mutations`
field. Please note, that all keys which are there by default have to stay, since program would not work otherwise.
