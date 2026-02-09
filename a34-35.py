import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
X_train =np.random.uniform(1,10,(20,2))
y_train=np.where(X_train[:,0] + X_train[:,1]>10,1,0)

x_vals=np.arange(0,10,0.1)
y_vals=np.arange(0,10,0.1)
xx,yy = np.meshgrid(x_vals,y_vals)

X_test=np.c_[xx.ravel(),yy.ravel()] #to create 2D grids

knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

plt.figure(figsize =(6,6))
plt.scatter(X_test[y_pred == 0,0],X_test[y_pred==0,1],color='blue',s=1,label='Class 0')
plt.scatter(X_test[y_pred == 1,0],X_test[y_pred==1,1],color='red',s=1,label='Class 1')

plt.scatter(X_train[:,0],X_train[:,1],c=y_train,edgecolor='black',s=60)

plt.title("A4: kNN Classification (k = 3)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#A5
k_values=[1,3,5,7,9]

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    plt.figure(figsize=(6,6))

    plt.scatter(X_test[y_pred == 0,0],X_test[y_pred==0,1],color='blue',s=1,label='Class 0')
    plt.scatter(X_test[y_pred == 1,0],X_test[y_pred==1,1],color='red',s=1,label='Class 1')
    
    plt.scatter(X_train[:,0],X_train[:,1],c=y_train,edgecolor='black',s=60)
    

    plt.title("A5")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

