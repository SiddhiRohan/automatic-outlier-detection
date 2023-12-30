# create_dataset.py

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

def create_boston_housing_dataset():
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

def save_dataset(data, filename):
    # Save the dataset as a CSV file
    data.to_csv(filename, index=False)
    print(f"Dataset saved as {filename}")

if __name__ == "__main__":
    # Specify the filename for the dataset
    dataset_filename = "data/boston_housing_dataset.csv"
    
    # Create the Boston Housing dataset
    boston_dataset = create_boston_housing_dataset()
    
    # Save the dataset
    save_dataset(boston_dataset, dataset_filename)
