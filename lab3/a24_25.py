import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

df = pd.read_csv("ml_dataset.csv")
df = df.dropna()
X = df.iloc[:, :-1].values
v1=X[0]
v2=X[1]

def min_dis(a,b,p):
    s=0
    for i in range(len(a)):
        s=s+abs(a[i]-b[i])** p
    return s ** (1/p)

p_values=range(1,11)
distances=[]

for p in p_values:
    d=min_dis(v1,v2,p)
    distances.append(d)

plt.plot(p_values, distances, marker='o')
plt.xlabel("p value")
plt.ylabel("Minkowski Distance")
plt.show()

#A5
own=min_dis(v1,v2,p)
pack=minkowski(v1,v2,p)
print("Minkowski distance using function :", own)
print("Minkowski distance using package :", pack)
