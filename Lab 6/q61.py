import numpy as np
import pandas as pd

df = pd.read_csv("dataset.csv")
df = df.dropna()

y = df.iloc[:, -1]

def calculate_entropy(labels):

    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()

    entropy = 0
    for p in probabilities:
        entropy += -p * np.log2(p)

    return entropy

entropy_value = calculate_entropy(y)

print("Entropy of dataset:", entropy_value)
