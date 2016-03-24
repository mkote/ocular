# This is the tests for preprocessing.oacl module.
from __future__ import division
import unittest
from numpy.testing import *
import numpy as np
from d606.preprocessing.oacl import *


class TestOACLfunctions(unittest.TestCase):

    def test_moving_avg_filter(self):
        # Arrange
        data = [56, 99, 20, 36, 56, 15, 5, 97, 91, 88, 35]
        expected = [175/3, 155/3, 112/3, 107/3, 76/3, 39, 193/3, 92, 214/3]
        # Act
        actual = moving_avg_filter(data, 3)

        # Assert
        assert_array_almost_equal(expected, actual)

    def test_moving_avg(self):
        # Arrange
        t = 5
        m = 5  # 5 point moving average
        data = [56, 99, 20, 36, 56, 15, 5, 97, 91, 88, 35]  # Odd list assumed
        expected = 0.2 * (36 + 56 + 15 + 5 + 97)

        # Act
        actual = symmetric_moving_avg(data, m, t)

        # Assert
        error_msg = "Moving average of ", \
                    str(actual), \
                    " was found. Expected "\
                    , str(expected)
        self.assertEqual(actual, expected, error_msg)

    def test_moving_avg_startof_series(self):
        # Tests invalid MA on the beginning of the series.
        # Arrange
        t = 0
        m = 3  # 3 point moving average
        data = [56, 99, 20, 36, 56, 15, 5, 97, 91, 88, 35]  # Odd list assumed

        # Act + Assert
        self.assertRaises(ValueError, symmetric_moving_avg, data, m, t)

    def test_moving_avg_endof_series(self):
        # Tests invalid MA on the end of the series.
        # Arrange
        t = 10
        m = 3  # 2 point moving average
        data = [56, 99, 20, 36, 56, 15, 5, 97, 91, 88, 35]  # Odd list assumed

        # Assert
        self.assertRaises(ValueError, symmetric_moving_avg, data, m, t)

    def test_relative_height_pass(self):
        # Arrange
        data = [1, 3, 2]
        expected = 2
        index_t = 1 # 2nd element
        # Act
        actual = relative_height(data, index_t)
        # Assert
        err_msg = "Actual: ", str(actual), ". Expected: ", str(expected)
        self.assertEqual(actual, expected)
        pass

    def test_relative_height_low_time(self):
        # Arrange
        data = [1, 3, 2]
        index_t = 0 # first element, too low!

        # Act + Assert
        self.assertRaises(ValueError, relative_height, data, index_t)

    def test_relative_height_high_time(self):
        # Arrange
        data = [1, 3, 2]
        index_t = 2 # 2nd element

        # Act + Assert
        self.assertRaises(ValueError, relative_height, data, index_t)

    def test_find_time_indexes(self):
        # Arrange
        # Act
        # Assert
        pass
