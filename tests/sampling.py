import unittest
from unittest import TestCase

from gamexplainer.sampling_startegies import *
import numpy as np


def feat_min_max(thresholds_dict: dict, epsilon_perc: float):
    feat_range = {feat: np.max(values) - np.min(values) for feat, values in thresholds_dict.items()}
    feat_epsilon = {feat: f_range * epsilon_perc for feat, f_range in feat_range.items()}
    res = {
        feat: (np.min(thresholds_dict[feat]) - epsilon, np.max(thresholds_dict[feat]) + epsilon) for
        feat, epsilon in feat_epsilon.items()}
    return res


class TestSampling(TestCase):
    def setUp(self):
        thresholds_dict = {
            "f1": [1, 5, 10, 15, 20],
            "f2": [1, 0.5, -0.5, -1],
            "f3": [4, 8, 16, 32, 64, 128, 256]
        }
        self.thresholds_dict = thresholds_dict

    def test_rnd_sampling(self):
        epsilon_perc = 0.2
        f_min_max = feat_min_max(self.thresholds_dict, epsilon_perc)
        for i in range(1, 6):
            sampled = rnd_sampling(self.thresholds_dict, i, epsilon_perc=epsilon_perc, categories=[])
            for feat, sampled_values in sampled.items():
                self.assertEqual(len(sampled_values), i)
                for value in sampled_values:
                    self.assertGreaterEqual(value, np.min(f_min_max[feat]))
                    self.assertLessEqual(value, np.max(f_min_max[feat]))

    def test_all_sampling(self):
        epsilon_perc = 0.1
        sampled = all_sampling(self.thresholds_dict, 1, epsilon_perc=epsilon_perc, categories=[])
        f_min_max = feat_min_max(self.thresholds_dict, epsilon_perc)
        for key, values in self.thresholds_dict.items():
            sampled_values = sampled[key]

            # check the minimum and the maximum
            expected_min = min(f_min_max[key])
            expected_max = max(f_min_max[key])
            self.assertAlmostEqual(expected_min, min(sampled_values))
            self.assertAlmostEqual(expected_max, max(sampled_values))

            # check all the midpoints
            for i in range(len(values) - 1):
                mid_point = (values[i] + values[i + 1]) / 2
                # sampled_values[i + 1] because the first is the minimum outside the range
                self.assertAlmostEqual(sampled_values[i + 1], mid_point)

    def test_equal_dist_sampling(self):
        epsilon_perc = 0.1
        f_min_max = feat_min_max(self.thresholds_dict, epsilon_perc)
        # test with various sampling sizes
        for sample_size in range(3, 10):
            sampled = equal_dist_sampling(self.thresholds_dict, sample_size, epsilon_perc, categories=[])
            for feat, sampled_values in sampled.items():
                # check size
                self.assertEqual(len(sampled_values), sample_size)
                # check that minimum and maximum values are respected
                self.assertGreaterEqual(min(sampled_values), min(f_min_max[feat]))
                self.assertLessEqual(max(sampled_values), max(f_min_max[feat]))

                # check equal dist for all sampled values
                dist = abs(sampled_values[1] - sampled_values[0])
                for i in range(len(sampled_values) - 1):
                    self.assertAlmostEqual(abs(sampled_values[i + 1] - sampled_values[i]), dist)

    def test_quantile_sampling(self):
        # test with various sampling sizes
        for sample_size in range(3, 10):
            sampled = quantile_sampling(self.thresholds_dict, sample_size, categories=[])
            for feat, sampled_values in sampled.items():
                # Since the function is based mainly on np.quantile, we just check that the sample has the correct size
                self.assertEqual(len(sampled_values), min(sample_size, len(self.thresholds_dict[feat])))

    def test_equi_size_sampling(self):
        # test with manually computed tests
        res_3 = {
            "f1": [1, 5, 15],
            "f2": [-1, -0.5, 0.75],
            "f3": [6, 24, 149.3333333333333333333]
        }
        sampled = equi_size_sampling(self.thresholds_dict, 3, categories=[])
        for feat, sampled_values in sampled.items():
            self.assertIsNone(np.testing.assert_array_equal(res_3[feat], sampled_values))

        res_4 = {
            "f1": [1, 5, 10, 17.5],
            "f2": [-1, -0.5, 0.5, 1],
            "f3": [4, 8, 16, 120]
        }
        sampled = equi_size_sampling(self.thresholds_dict, 4, categories=[])
        for feat, sampled_values in sampled.items():
            self.assertIsNone(np.testing.assert_array_equal(res_4[feat], sampled_values))


if __name__ == '__main__':
    unittest.main()
