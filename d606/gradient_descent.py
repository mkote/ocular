from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
from autograd import grad
from autograd.util import quick_grad_check
from builtins import range
from numpy import mat

from preprocessing.oacl import objective_function, variance, column

n_trials = 0
trial_signals = []
trial_artifact_signals = []
labels = []

def sigmoid(z):
    z = theta # define Z
    return np.tanh(z)

def logistic_predictions(weights, b, inputs):
    theta_t = weights.transpose()
    theta = weights
    for i in range(n_trials):
        A = inputs
        R_a = np.dot(A,A.transpose())
        k1 = np.dot(np.dot(-1*np.transpose(theta),R_a), theta)
        k2 = np.dot(2 * theta_t, np.dot(A, x.transpose()))
        var = np.var(x)
    z = k1 - k2 + var + b
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(z)

def training_loss(weights, b):
    # Training loss is the negative log-likelihood of the training labels.
    for i in range(n_trials):
        X = None # Get trial data
        y = None # Get trial label
        preds = logistic_predictions(weights, b, X)

    label_probabilities = preds * y + (1 - preds) * (1 - y)
    return -np.sum(np.log(label_probabilities))

# Build a toy dataset.
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
A = np.array([[0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 7, 0, 0, 0]])
weights = np.array([0.0, 0.0])
b = np.array([1])
y = np.array([1,2])
#inputs = np.array([[0.52, 1.12,  0.77],
#                   [0.88, -1.08, 0.15],
#                   [0.52, 0.06, -1.30],
#                   [0.74, -2.49, 1.39]])
#targets = np.array([True, True, False, True])

# Build a function that returns gradients of training loss using autograd.
training_gradient_fun = grad(training_loss)

# Check the gradients numerically, just to be safe.

quick_grad_check(training_loss, weights, b)

# Optimize weights using gradient descent.
print("Initial loss:", training_loss(weights, b))
for i in range(100):
    weights -= training_gradient_fun(weights, b) * 0.01

print("Trained loss:", training_loss(weights, b))
print("weights: ", weights)