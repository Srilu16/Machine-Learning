import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1].values

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42).fit(X_train)

sil_score = silhouette_score(X_train, kmeans.labels_)
ch_score = calinski_harabasz_score(X_train, kmeans.labels_)
db_score = davies_bouldin_score(X_train, kmeans.labels_)

print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_score)
