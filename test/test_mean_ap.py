import os
from unittest import TestCase


class TestMeanAP(TestCase):

    def test_mean_ap(self):
        path_to_results_dir = '../results'
        path_to_mean_ap_txt = os.path.join(path_to_results_dir, 'mean_ap.txt')
        self.assertTrue(os.path.isfile(path_to_mean_ap_txt))

        with open(path_to_mean_ap_txt, 'r') as fp:
            mean_ap = fp.readline()

        mean_ap = float(mean_ap)
        self.assertGreaterEqual(mean_ap, 0.6)
