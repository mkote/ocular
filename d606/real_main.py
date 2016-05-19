import autograd.numpy as np
from autograd import grad
from autograd.util import quick_grad_check

from preprocessing.oacl import find_artifact_signals, extract_trials_array


global inputs
global targets
global w
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

def rearrange_target(target, label):
    new_targets = []
    for tar in target:
        if tar == label:
            new_targets.append(True)
        else:
            new_targets.append(False)
    return np.array(new_targets)

def magic(trials_from_runs, labels, m, range_list):
    # Compute inputs

    global inputs
    global targets
    global w
    global ranges
    global mov
    inputs = trials_from_runs
    targets = np.array(labels)
    mov = m
    ranges = range_list
    w = np.array([0.0, 0.0, 0.0])
    weight_list = []
    # Fit a logistic classifier for each class (4) one-vs-rest
    for target in set(targets.tolist()):
        print("fitting logistic regression for class: " + str(target))
        old_targets = targets
        labels = rearrange_target(old_targets, target) # Rearrange labels for one-vs-rest
        fit_logit(inputs, labels)

    # Define a function that returns gradients of training loss using autograd.
    training_gradient_fun = grad(training_loss)

    # Optimize weights using gradient descent.
    print "Initial loss:", training_loss(w)
    for i in xrange(100):
        print "gradient dankness", i
        w -= training_gradient_fun(w) * 0.01

    print str(w)
    print  "Trained loss:", training_loss(w)

def fit_logit(inputs, target):
    global w
    weight_list = []
    old_targets = targets
    training_gradient_fun = grad(training_loss)  # get derivative of loss function.
    # Optimize weights using gradient descent.
    print("Initial loss:", training_loss(w))
    for i in range(100):
        print("Optimizing: " + str(i))
        w -= training_gradient_fun(w) * 0.01

    print("Trained loss:", training_loss(w))
    print(w)
    weight_list.append(w)

    print("ALl weights: ")
    print(weight_list)