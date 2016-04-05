import itertools

# Specify values to constants in a list for each constant, then it is
# possible to use grid search over all the constants, be aware of the
# running time when many constants are chosen.

# parameter C for svm
C = [0.25, 0.50, 0.75, 1]

# kernel for svm
kernel = ['rbf', 'linear']

# num components for csp
n_comp = [5, 6, 7]

# ranges for bandpassing
band_list = [[[4, 8], [12, 16]],
             [[4, 8], [12, 16], [17, 30]]]

''' Set parameters for the Grid search '''
grid_parameters = [band_list]
grid = ()


def save_results():
    pass


def grid_combinator(grid_search):
    grid_list = list(itertools.product(*grid_search))
    return grid_list
