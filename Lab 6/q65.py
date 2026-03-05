import numpy as np
import pandas as pd

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    ent = 0
    for p in probs:
        ent -= p * np.log2(p)
    return ent

def information_gain(X_column, y):
    total_entropy = entropy(y)
    values, counts = np.unique(X_column, return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset_y = y[X_column == v]
        if len(subset_y) == 0:
            continue
        weighted_entropy += (c / len(X_column)) * entropy(subset_y)
    return total_entropy - weighted_entropy

def best_feature(X, y):
    best_gain = -1
    best_index = None
    for i in range(X.shape[1]):
        gain = information_gain(X[:, i], y)
        if gain > best_gain:
            best_gain = gain
            best_index = i
    return best_index

class Node:
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label
        self.children = {}

def build_tree(X, y):
    if len(np.unique(y)) == 1:
        return Node(label=y[0])
    if len(y) < 3:
        values, counts = np.unique(y, return_counts=True)
        return Node(label=values[np.argmax(counts)])
    feature = best_feature(X, y)
    if feature is None:
        values, counts = np.unique(y, return_counts=True)
        return Node(label=values[np.argmax(counts)])
    node = Node(feature=feature)
    values = np.unique(X[:, feature])
    for v in values:
        mask = X[:, feature] == v
        if np.sum(mask) == 0:
            continue
        child = build_tree(X[mask], y[mask])
        node.children[v] = child
    return node

df = pd.read_csv("dataset.csv")
X = df.iloc[:, :-1].values
y = df["LABEL"].values

tree = build_tree(X, y)
print("Decision Tree built successfully")
print("Root Feature Index:", tree.feature)
