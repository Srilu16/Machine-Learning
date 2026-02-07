import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ml_dataset.csv")
df = df.dropna()
X = df.iloc[:, :-1].values
feature = X[:, 0]
hist,bin_edges=np.histogram(feature,bins=10)

print("Histogram counts : ",hist)
print("Bin edges : ",bin_edges)

plt.hist(feature, bins=10)
plt.xlabel("Feature values")
plt.ylabel("Frequency")
plt.show()

m = np.mean(feature)
v = np.var(feature)
print("Mean of the feature:", m)
print("Variance of the feature:", v)
