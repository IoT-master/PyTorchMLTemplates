from itertools import product
parameters = dict(
    lr=[.01, .001],
    batch_size=[10, 100, 1000],
    shuffle=[True, False]
)

param_values = [v for v in parameters.values()]

for lr, batch_size, shuffle in product(*param_values):
    print(lr, batch_size, shuffle)
