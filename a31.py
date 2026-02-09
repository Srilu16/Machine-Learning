import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

df=pd.read_csv("ml_dataset.csv")
df=df.dropna()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
labels = np.unique(y)
print("X shape:", X.shape)
print("y shape:", y.shape)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)

knn =KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print("Training Performance Metrics")
cm_train = confusion_matrix(y_train,y_train_pred)
print("Confusion Matrix : ",cm_train)
print("Precision : ",precision_score(y_train,y_train_pred, average='weighted'))
print("Recall : ",recall_score(y_train,y_train_pred, average='weighted'))
print("F1 Score : ",f1_score(y_train,y_train_pred, average='weighted'))

print("Testing Performance Metrics")
cm_test = confusion_matrix(y_test,y_test_pred)
print("Confusion Matrix : ",cm_test)
print("Precision : ",precision_score(y_test,y_test_pred, average='weighted'))
print("Recall : ",recall_score(y_test,y_test_pred, average='weighted'))
print("F1 Score : ",f1_score(y_test,y_test_pred, average='weighted'))
