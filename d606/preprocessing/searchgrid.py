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
band_list = [[[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 30], [30, 36]],
             [[16, 20], [20, 24], [24, 30], [30, 36]]]

''' Set parameters for the Grid search '''
grid_parameters = []
grid = ()


def save_results(result, parameters):
    with open("results.txt", "a") as result_file:
        result_file.write(str(result) + str('\n') + str(parameters) + str('\n'))


def grid_combinator(grid_search):
    grid_list = list(itertools.product(*grid_search))
    return grid_list
