# linear_regression_utils.py

import pandas as pd
import numpy as np

def load_dataset():
    # Fetch the Boston Housing dataset from the original source
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # Create a DataFrame from the data
    data_df = pd.DataFrame(data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

    # Add the target column to the DataFrame
    data_df['target'] = target

    return data_df
