# src/outlier_detection.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

class AutomaticOutlierDetection:
    def __init__(self):
        self.model = LinearRegression()
        self.outliers = None

    def detect_outliers(self, data, target_column):
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Fit the model
        self.model.fit(X, y)

        # Calculate residuals
        residuals = y - self.model.predict(X)

        # Use Z-score to detect outliers
        z_scores = stats.zscore(residuals)
        self.outliers = data.index[abs(z_scores) > 2].tolist()

    def fit_and_detect_outliers(self, data, target_column):
        self.detect_outliers(data, target_column)

        clean_data = data.drop(index=self.outliers)

        # Refit the model on clean data
        X_clean = clean_data.drop(columns=[target_column])
        y_clean = clean_data[target_column]
        self.model.fit(X_clean, y_clean)

    def plot_results(self, data, target_column):
        predictions = self.model.predict(data.drop(columns=[target_column]))

        plt.scatter(data[target_column], predictions, label='Adjusted Model')
        plt.scatter(data[target_column][self.outliers], predictions[self.outliers], color='red', label='Outliers')
        plt.xlabel(target_column)
        plt.ylabel('Predicted ' + target_column)
        plt.legend()
        plt.title('Linear Regression with Outlier Detection')
        plt.show()
