import autograd.numpy as np
from autograd import grad
from preprocessing.oacl import find_artifact_signals, extract_trials_array


global inputs
global targets
global weights
global mov
global ranges

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1)

def logistic_predictions(weights, inputs):
    r = ranges
    m = mov
    # Construct z
    theta = weights[1:3]
    b = weights[0]
    # 1. Construct artifact signals for every trial
    z_values = []
    for trial in inputs:
        A = find_artifact_signals(trial, mov, r)
        num_pads = m/2
        padding = np.array([[0]*num_pads])
        padding = np.concatenate((padding, padding))
        A = np.concatenate((A, padding), axis=1)
        A = np.concatenate((padding, A), axis=1)
        # pad both sides of all artifact signals by m/2
        var = np.var(trial)
        covar = np.dot(A, np.transpose(A))
        corr = np.dot(A, np.transpose(trial))
        # Compute z with weights (thetas)
        term1 = np.dot(np.dot(np.transpose(theta), covar), theta)
        term2 = np.dot(2*np.transpose(theta), corr)
        z = term1 + term2 + var + b
        z_values.append(z)

    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.array(z_values))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

def magic(trials_from_runs, labels, m, range_list):
    # Compute inputs
    global inputs
    global targets
    global weights
    global ranges
    global mov
    inputs = trials_from_runs
    targets = np.array(labels)
    mov = m
    ranges = range_list
    weights = np.array([0.0, 0.0, 0.0])

    # Define a function that returns gradients of training loss using autograd.
    training_gradient_fun = grad(training_loss)

    # Optimize weights using gradient descent.
    print "Initial loss:", training_loss(weights)
    for i in xrange(100):
        print "gradient dankness", i
        weights -= training_gradient_fun(weights) * 0.01

    print str(weights)
    print  "Trained loss:", training_loss(weights)