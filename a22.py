import numpy as np
import pandas as pd

df=pd.read_csv("ml_dataset.csv")
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

def mean(data):
    s=0
    for x in data:
        s=s+x
    return s / len(data)
def variance(data):
    m=mean(data)
    s=0
    for x in data:
        s=s+ (x-m) **2
    return s / len(data)
def s_d(data):
    return variance(data) ** 0.5
def dataset_mean_std(matrix):
    means = []
    stds = []

    for i in range(matrix.shape[1]):
        column = matrix[:, i]
        means.append(mean(column))
        stds.append(s_d(column))

    return np.array(means), np.array(stds)
labels =np.unique(y)
class0=X[y==labels[0]]
class1=X[y==labels[1]]
centroid0=np.mean(class0,axis=0)
centroid1=np.mean(class1,axis=0)
print("Centroid of Class 0: ",centroid0)
print("Centroid of Class 1: ",centroid1)

spread0=np.std(class0,axis=0)
spread1=np.std(class1,axis=0)
print("Spread of Class 0: ",spread0)
print("Spread of Class 1: ",spread1)

interclass_distance=np.linalg.norm(centroid0 - centroid1)
print("Interclass Distance: ",interclass_distance)
