import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1].values

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

distorsions = []  

for k in range(2, 20):
    kmeans = KMeans(n_clusters=k).fit(X_train)
    distorsions.append(kmeans.inertia_)

plt.plot(distorsions)
plt.xlabel("k (index represents k from 2 to 19)")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.show()
