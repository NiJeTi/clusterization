import os

import pandas as pd


def get_data():
    dataset_path = os.path.join(os.path.dirname(__file__), 'Data.csv')
    dataset = pd.read_csv(dataset_path)

    return dataset.iloc[:, [3, 4]].values
