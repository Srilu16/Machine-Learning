import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score
df = pd.read_csv("ml_dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

labels =np.unique(y)
X = X[(y == labels[0]) | (y == labels[1])]
y = y[(y == labels[0]) | (y == labels[1])]

#A6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

#A7
neigh =  KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

#A8
accuracy = neigh.score(X_test, y_test)
print("kNN Classification Accuracy:", accuracy)

#A9
predictions = neigh.predict(X_test)
print("Predicted classes for test set:", predictions)

#A10
def distance(a,b):
    return np.sqrt(np.sum((a-b) ** 2))
def knn(X_train,y_train,test,k):
    d = [(distance(x, test), y) for x, y in zip(X_train, y_train)]
    d.sort()
    labels = [y for (_, y) in d[:k]]
    return max(set(labels), key=labels.count)
preds = np.array([knn(X_train, y_train, x, 3) for x in X_test])
accuracy = np.mean(preds == y_test)

print("Own kNN Accuracy:",accuracy)

#A11
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train, y_train)
acc_k1 = knn_1.score(X_test, y_test)
print("Accuracy for k = 1 :", acc_k1)
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)
acc_k3 = knn_3.score(X_test, y_test)
print("Accuracy for k = 3 :", acc_k1)

k_vals=range(1,12)
accu=[]
for k in k_valus:
     model = KNeighborsClassifier(n_neighbors=k)
     model.fit(X_train, y_train)
     accu.append(model.score(X_test, y_test))

plt.plot(k_vals, accu, marker='o')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()

#A12
cm= confusion_matrix(y_test,predictions)
print("Confusion Matrix :  ", cm)
print("Precision : ", precision_score(y_test,predictions))
print("Recall : ", recall_score(y_test,predictions))
print("F1 Score : ", f1_score(y_test,predictions))

#A13
TP=TN=FP=FN=0
for yt,yp in zip(y_test,predictions):
    if yt == labels[1] and yp == labels[1]: TP += 1
    elif yt == labels[0] and yp == labels[0]: TN += 1
    elif yt == labels[0] and yp == labels[1]: FP += 1
    elif yt == labels[1] and yp == labels[0]: FN += 1

accuracy1 =(TP+TN)/(TP+TN+FP+FN)
precision1 =(TP)/(TP+FP)
recall1 =(TP)/(TP+FN)
f11 = 2*precision1*recall1/(precision1 +recall1)



