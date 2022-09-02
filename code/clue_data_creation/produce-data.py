import os
from sqlite3 import paramstyle
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
from pathlib import Path
from operator import concat

num_points_per_layer = 10000
csv_list = []
points_per_layer = np.arange(num_points_per_layer, dtype=float)
path = os.path.abspath("test_data")

if not os.path.exists(path):
    os.makedirs(path)

for i in range(100):
    X, y = datasets.make_blobs(n_samples=num_points_per_layer, n_features=2, centers=1000, cluster_std=3, center_box=(-250,250))
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1]))
    layer = np.full_like(points_per_layer, i)
    weight = np.full_like(points_per_layer, 1)
    df['layer'] = layer
    df['weight'] = weight
    csv_list.append(df)
csv_merged = pd.concat(csv_list, ignore_index=True) 
csv_merged.to_csv(path + '/test_100_layers.csv', index=False, header=False)