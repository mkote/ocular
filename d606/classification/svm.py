from sklearn import *


# Note: if you want to plot, the input data should not be scaled.
# otherwise, we should perform some scaling/normalization
# before we apply the data on SVM.
# This svm learner function learns a polynomial svm.
# The standard regularization parameter is 1.0


def learn_svm(self, feature_values, target_values, poly_degree, reg_param):
    poly_svc = svm.SVC(kernel='poly', degree=poly_degree, C=reg_param)\
        .fit(feature_values, target_values)
    return poly_svc
