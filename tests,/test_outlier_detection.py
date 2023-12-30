#tests/test_outlier_detection.py

import unittest
from outlier_detection import AutomaticOutlierDetection
from linear_regression_utils import load_dataset

class TestAutomaticOutlierDetection(unittest.TestCase):
    def setUp(self):
        self.outlier_detection = AutomaticOutlierDetection()
        self.data = load_dataset()

    def test_fit_and_detect_outliers(self):
        adjusted_model = self.outlier_detection.fit_and_detect_outliers(self.data, target_column="target")

    def test_plot_results(self):
        pass

if __name__ == '__main__':
    unittest.main()
