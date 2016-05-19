# from __future__ import absolute_import
# from __future__ import print_function
# import autograd.numpy as np
# from autograd import grad
# from autograd.util import quick_grad_check
# from builtins import range
# import numpy as nump
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
#
# def logistic_predictions(weights, inputs, raw_signal):
#     # Outputs probability of a label being true according to logistic model.
#
#     # A dot At
#     term1 = np.dot(inputs, np.transpose(inputs))
#     term2 = np.dot(np.transpose(weights[1:]), term1)
#     term3 = nump.dot(term2, weights[1:])
#
#     # A dot X0
#     term4 = nump.dot(inputs, np.transpose(raw_signal))
#     term5 = np.dot(2 * np.transpose(weights[1:]), term4)
#
#     # r0 + b
#     term6 = np.var(raw_signal) + weights[0]
#
#     # Full term
#     fullterm = term3 - term5 + term6
#
#     return sigmoid(fullterm)
#
#
# def training_loss(weights):
#     # Training loss is the negative log-likelihood of the training labels.
#     preds = logistic_predictions(weights, inputs, raw_signal)
#     label_probabilities = preds * targets + (1 - preds) * (1 - targets)
#     return -np.sum(np.log(label_probabilities))
#
#
# def rearrange_target(target, label):
#     new_targets = []
#     for tar in target:
#         if tar == label:
#             new_targets.append(True)
#         else:
#             new_targets.append(False)
#     return np.array(new_targets)
#
#
# # Build a toy dataset.
# inputs = np.array([[0.52, 1.12,  0.77],
#                    [0.88, -1.08, 0.15],
#                    [0.52, 0.06, -1.30],
#                    [0.74, -2.49, 1.39]])
#
# raw_signal = np.array([[0.22, 4.12, 1.77],
#                        [1.23, 4.12, 2.34],
#                        [3.23, 6.12, 2.34],
#                        [2.23, 3.12, 7.34]])
#
#
# old_targets = np.array([1, 2, 3, 4])
#
# weight_list = []
# # Build a function that returns gradients of training loss using autograd.
#
# # Run OVRRR!!!
# for target in set(old_targets.tolist()):
#     targets = rearrange_target(old_targets, target)
#     training_gradient_fun = grad(training_loss)
#
#     # Check the gradients numerically, just to be safe.
#     weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
#     quick_grad_check(training_loss, 0)
#
#     # Optimize weights using gradient descent.
#     # print("Initial loss:", training_loss(weights))
#     for i in range(100):
#         weights -= training_gradient_fun(weights) * 0.01
#
#     #print("Trained loss:", training_loss(weights))
#     print(weights)
#     weight_list.append(weights)
#
# print(weight_list)