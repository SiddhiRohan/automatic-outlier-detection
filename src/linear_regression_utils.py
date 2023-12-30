#src/linear_regression_utils.py

from sklearn.datasets import load_boston
import pandas as pd

def load_dataset():
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['target'] = boston.target
    return data