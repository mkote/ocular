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
            print "You must have two tasks for filtering"
    else:
        print len(tasks)

# load data
data = load_data(1, 'T')
matrix, trials, labels, artifacts = data[3]

# get part of matrix which is relevant
matrix, trials = extract_trials(matrix, trials)
