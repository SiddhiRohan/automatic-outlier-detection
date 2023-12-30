# tests/test_outlier_detection.py

import unittest
from src.outlier_detection import AutomaticOutlierDetection
from src.linear_regression_utils import load_dataset

class TestAutomaticOutlierDetection(unittest.TestCase):
    def setUp(self):
        self.outlier_detector = AutomaticOutlierDetection()
        self.data = load_dataset()

    def test_fit_and_detect_outliers(self):
        adjusted_model = self.outlier_detector.fit_and_detect_outliers(self.data, target_column="target")
        # Add assertions based on your expectations

    def test_plot_results(self):
        # Implement tests for the plot_results method
        pass

if __name__ == '__main__':
    unittest.main()
