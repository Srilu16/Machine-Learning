import pandas as pd
import numpy as np

data = pd.read_csv("dataset.csv")

y = data["LABEL"]

p = y.value_counts(normalize=True)

gini = 1 - np.sum(p**2)

print("Gini Index:", gini)
