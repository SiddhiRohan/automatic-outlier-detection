#src/outlier_detection.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

class AutomaticOutlierDetection:
    def __init__(self):
        self.model = LinearRegression()
        self.outliers = None

    def detect_outliers(self, data, target_column):
        z_scores = stats.zscore(data[target_column])
        self.outliers = data.index[abs(z_scores) > 3].tolist()

    def fit_and_detect_outliers(self, data, target_column):
        X = data.drop(columns=[target_column])
        y = data[target_column]

        self.model.fit(X, y)

        self.detect_outliers(data, target_column)

        clean_data = data.drop(index = self.outliers)
        self.model.fit(clean_data.drop(columns=[target_column]), clean_data[target_column])

        return self.model

    def plot_results(self, data, target_column):
        plt.scatter(data[target_column], self.model.predict(data.drop(columns=[target_column])), label='Adjusted Model')
        plt.scatter(data[target_column].iloc[self.outliers], self.model.predict(data.drop(columns=[target_column])).iloc[self.outliers, color='red', label='Outliers'])
        plt.xlabel(target_column)
        plt.ylabel('Predicted' + target_column)
        plt.legend()
        plt.title('Linear Regression with Outlier Detection')
        plt.show()