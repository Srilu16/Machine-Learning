import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1].values

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_train)

print("Cluster Labels:")
print(kmeans.labels_)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)
