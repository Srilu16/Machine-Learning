import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("ml_dataset.csv")
df=df.dropna()

X=df.iloc[:,[0,1]].values
y=df.iloc[:,-1].values

#A6(i)
labels =np.unique(y)
X= X[(y == labels[0]) | (y == labels[1])]
y= y[(y == labels[0]) | (y == labels[1])]

plt.scatter(X[y == labels[0],0] , X[y == labels[0],1] , color='blue',label='Class 0')
plt.scatter(X[y == labels[1],0] , X[y == labels[1],1] , color='red',label='Class 1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("A6 - 1")
plt.legend()
plt.show()

#A6(ii)
x_vals = np.arange(X[:,0].min()-1, X[:,0].max()+1, 0.5)
y_vals = np.arange(X[:,1].min()-1, X[:,1].max()+1, 0.5)
xx,yy = np.meshgrid(x_vals,y_vals)

X_test=np.c_[xx.ravel(),yy.ravel()] 

knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)

y_pred=knn.predict(X_test)

plt.figure(figsize =(6,6))
plt.scatter(X_test[y_pred == 0,0],X_test[y_pred==0,1],color='blue',s=1)
plt.scatter(X_test[y_pred == 1,0],X_test[y_pred==1,1],color='red',s=1)

plt.scatter(X[:,0],X[:,1],c=y,edgecolor='black',s=60)

plt.title("A6 - 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#A6(iii)
k_values=[1,3,5,7]

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    plt.figure(figsize=(6,6))

    plt.scatter(X_test[y_pred == 0,0],X_test[y_pred==0,1],color='blue',s=1)
    plt.scatter(X_test[y_pred == 1,0],X_test[y_pred==1,1],color='red',s=1)
    
    plt.scatter(X[:,0],X[:,1],c=y,edgecolor='black',s=60)
    
    plt.title("A6 - 3")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

