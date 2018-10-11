#Projects used as motivation
## Direct influence
First approach was based on a [blog posted on medium](https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164),
and trying to adapt this approach to 2 dimensional data-sets. However, it quickly diverged, as number of parameters 
increased, and number of combinations that genetic algorithm can explore had to be restricted. Additionally, some more
sophisticated techniques were used in this repository, than in the above mentioned.
### Small things
There are some choices (mostly to possible mutations), which were based purely on work of others:
* Size of kernel filters limited to 3x3, as it gives better performance with same information storage as explained on
pg.3 of [paper](https://arxiv.org/abs/1409.1556) by Simonyan et al.
##Other approaches
There are other solutions to problem of finding the best architecture, besides genetic algorithm, or well established
methods, such as Grid Search etc. 
In this particular case, there is a [paper](https://arxiv.org/abs/1609.00074v2) by Jin, Yan et al,
which shows an alternative approach, which can lead to possibly better results.
##Other similar github repositories
During the time of creation of this repository, I have found that there are other similar, yet different repositories, 
with the same aim on github:
* [gentun](https://github.com/gmontamat/gentun)
