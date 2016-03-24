import numpy as np
import scipy.linalg as la
from d606.preprocessing.dataextractor import *

# CSP takes any number of arguments, but each argument must be a collection of
# trials associated with a task. That is, for N tasks, N arrays are passed to
# CSP each with dimensionality (num trials of task N) x (feature vector)
# Trials may be of any dimension, provided that each trial for each task has
# the same dimensionality, otherwise there can be no spatial filtering since
# the trials cannot be compared


def common_spatial_patterns(*tasks):
    if len(tasks) < 2:
        print "Must have at least 2 tasks for filtering."
        return (None,) * len(tasks)
    else:
        filters = ()
        # CSP algorithm
        # For each task x, find the mean variances rx and not_rx, which will
        # be used to compute spatial filter sfx

        # create iterator for length of tasks
        iterator = range(0, len(tasks))
        for x in iterator:
            # Find rx
            rx = covariance_matrix(tasks[x][0])
            for t in range(1, len(tasks[x])):
                rx += covariance_matrix(tasks[x][t])
            rx /= len(tasks[x])

            # Find not_Rx
            count = 0
            not_rx = rx * 0
            for not_x in [element for element in iterator if element != x]:
                for t in range(0, len(tasks[not_x])):
                    not_rx += covariance_matrix(tasks[not_x][t])
                    count += 1
            not_rx = not_rx / count

            # Find the spatial filter SFx
            sfx = spatial_filter(rx, not_rx)
            filters += (sfx,)

            # Special case: only two tasks, no need to compute any more mean
            # variances
            if len(tasks) == 2:
                filters += (spatial_filter(not_rx, rx),)
                break
        return filters


# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled
# by the variance
def covariance_matrix(a):
    ca = np.dot(a, np.transpose(a)) / np.trace(np.dot(a, np.transpose(a)))
    return ca


# spatialFilter returns the spatial filter SFa for mean covariance
# matrices Ra and Rb
def spatial_filter(ra, rb):
    r = ra + rb
    e, u = la.eig(r)

    # CSP requires the eigenvalues E and eigenvector U be sorted in
    # descending order
    ord = np.argsort(e)
    ord = ord[::-1]  # argsort gives ascending order, flip to get descending
    e = e[ord]
    u = u[:, ord]

    # Find the whitening transformation matrix
    p = np.dot(np.sqrt(la.inv(np.diag(e))), np.transpose(u))

    # The mean covariance matrices may now be transformed
    sa = np.dot(p, np.dot(ra, np.transpose(p)))
    sb = np.dot(p, np.dot(rb, np.transpose(p)))

    # Find and sort the generalized eigenvalues and eigenvector
    e1, u1 = la.eig(sa, sb)
    ord1 = np.argsort(e1)
    ord1 = ord1[::-1]
    e1 = e1[ord1]
    u1 = u1[:, ord1]

    # The projection matrix (the spatial filter) may now be obtained
    sfa = np.dot(np.transpose(u1), p)
    return sfa.astype(np.float32)


# load data
data = load_data(3, 'T')
matrix, trials, labels, artifacts = data[3]

# get part of matrix which is relevant
matrix, trials = extract_trials(matrix, trials)

# split matrix into array of trials
matrix = trial_splitter(matrix, trials)
print len(matrix)
# feed shit to csp algorithm
common_spatial_patterns(matrix)
