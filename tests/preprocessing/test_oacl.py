# This is the tests for preprocessing.oacl module.
from __future__ import division
import unittest
from numpy.testing import *
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
        error_msg = "Moving average of ", str(actual), " was found. " \
                    "Expected ", str(expected)
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
        index_t = 1  # 2nd element
        # Act
        actual = find_relative_height(data, index_t)
        # Assert
        err_msg = "Actual: ", str(actual), ". Expected: ", str(expected)
        self.assertEqual(actual, expected, err_msg)
        pass

    def test_relative_height_low_time(self):
        # Arrange
        data = [1, 3, 2]
        index_t = 0  # first element, too low!

        # Act + Assert
        self.assertRaises(ValueError, find_relative_height, data, index_t)

    def test_relative_height_high_time(self):
        # Arrange
        data = [1, 3, 2]
        index_t = 2  # 2nd element

        # Act + Assert
        self.assertRaises(ValueError, find_relative_height, data, index_t)

    def test_find_relative_heights(self):
        # Arrange
        data = [56, 99, 20, 36, 56, 15, 5, 97, 91, 88, 35]
        expected = [79, 79, 20, 41, 41, 92, 92, 6, 53]

        # Act
        actual = find_relative_heights(data)

        # Assert
        assert_array_almost_equal(actual, expected)

    def test_find_time_indexes(self):
        # TODO: Write test
        pass

    def test_artifact_signal(self):
        # TODO: How should we test this ?
        pass

    def test_is_cross_zero_false(self):
        # Arrange
        a = 48.02
        b = 27.18
        expected = False

        # Act
        actual = is_cross_zero(a, b)

        # Assert
        self.assertEqual(expected, actual)

    def test_is_cross_zero_true(self):
        # Arrange
        a = 31.17
        b = -25.5
        expected = True

        # Act
        actual = is_cross_zero(a, b)

        # Assert
        self.assertEqual(expected, actual)

    def test_nearest_zero_point(self):
        # Arrange
        t_data = [48.02, 27.18, -10.06, 5.27, 31.17, -25.5, -11.6, 27.3, -8.08,
                  3.94]
        a = 0
        b = 1
        expected = 2  # here we expect index of -10.06 since it is closest to 0

        # Act
        actual = nearest_zero_point(t_data, a, b)

        # Assert
        self.assertEqual(expected, actual)

    def test_nearest_zero_point_false(self):
        # Arrange
        t_data = [48.02, 27.18, -10.06, 5.27, 31.17, -25.5, -11.6, 27.3, -8.08,
                  3.94]
        a = 9
        b = 10  # 10 is out of range here.
        expected = 9  # since b is out of range, we expect last sample in
        # signal.

        # Act
        actual = nearest_zero_point(t_data, a, b)

        # Assert
        self.assertEqual(expected, actual)

    def test_find_artifact_range(self):
        # Arrange
        t_data = [1.0, 2.0, 0.0, 3.0, 10.0, 2.0, -1.0, 0.0]
        peak = 4
        expected = (2, 6)

        # Act
        actual = find_artifact_range(t_data, peak)

        # Assert
        self.assertEqual(expected, actual)
