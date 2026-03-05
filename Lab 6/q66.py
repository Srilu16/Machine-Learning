import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1].values
y = df["LABEL"].values

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

plt.figure(figsize=(15,10))
plot_tree(
    model,
    feature_names=df.columns[:-1],
    class_names=True,
    filled=True
)

plt.title("Decision Tree Visualization")
plt.show()
