import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1].values

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

k_values = range(2, 11)

sil_scores = []
ch_scores = []
db_scores = []

print("k | Silhouette | CH Score | DB Index")
print("------------------------------------------")

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    labels = kmeans.labels_

    sil = silhouette_score(X_train, labels)
    ch = calinski_harabasz_score(X_train, labels)
    db = davies_bouldin_score(X_train, labels)

    sil_scores.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)

    print(f"{k} | {sil:.4f} | {ch:.4f} | {db:.4f}")

plt.figure()
plt.plot(k_values, sil_scores)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.show()

plt.figure()
plt.plot(k_values, ch_scores)
plt.xlabel("k")
plt.ylabel("CH Score")
plt.title("Calinski-Harabasz Score vs k")
plt.show()

plt.figure()
plt.plot(k_values, db_scores)
plt.xlabel("k")
plt.ylabel("DB Index")
plt.title("Davies-Bouldin Index vs k")
plt.show()
