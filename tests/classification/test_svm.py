# This file contains unit tests for svm in
# the classification submodule d606.classification

import unittest
from d606.classification import svm as learnsvm
from sklearn import datasets


class BaseSVMTestCase(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris()
        # we only take the first two features.
        # avoid this ugly slicing by using a two-dim dataset
        self.feature_values = iris.data[:, :2]
        self.target_values = iris.target


class MySkippedTestCase(BaseSVMTestCase):
    @unittest.skip("WIP")
    def test_linear_svc(self):
        # Arrange
        kernel = 'linear'
        regularization_param = 1.0

        # Act
        linsvc = learnsvm.learn_svm(self.feature_values,
                                    self.target_values,
                                    kernel,
                                    regularization_param)

        # Assert
        # TODO: check stuff about linsvc

    @unittest.skip("WIP")
    def test_poly_svc(self):
        # TODO: write unit tests
        # Arrange
        # Act
        # Assert
        pass

    @unittest.skip("WIP")
    def test_rbf_svc(self):
        # TODO: write unit tests
        # Arrange
        # Act
        # Assert
        pass
