import pandas as pd
import numpy as np

# load dataset
data = pd.read_csv("dataset.csv")

target = "LABEL"

def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    probs = counts / np.sum(counts)
    
    ent = 0
    for p in probs:
        ent += -p * np.log2(p)
    return ent

def information_gain(data, feature, target):

    total_entropy = entropy(data[target])

    values, counts = np.unique(data[feature], return_counts=True)

    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[feature] == values[i]]
        weight = counts[i] / np.sum(counts)
        weighted_entropy += weight * entropy(subset[target])

    ig = total_entropy - weighted_entropy
    return ig

for col in data.columns:
    if col != target:
        if data[col].dtype != "object":
            data[col] = pd.cut(data[col], bins=4, labels=False)

best_feature = None
best_ig = -1

for feature in data.columns:
    if feature != target:
        ig = information_gain(data, feature, target)
        print("Information Gain for", feature, ":", ig)

        if ig > best_ig:
            best_ig = ig
            best_feature = feature


print("\nRoot Node Feature:", best_feature)
print("Highest Information Gain:", best_ig)
