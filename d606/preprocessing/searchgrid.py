
# Specify values to constants in a list for each constant, then it is
# possible to use grid search over all the constants, be aware of the
# running time when many constants are chosen.

# parameter C for svm
C = [0.25, 0.50, 0.75, 1]

# kernel for svm
kernel = ['rbf', 'linear']

# num componants for csp
n_components = [5, 6, 7]


