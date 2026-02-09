import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("ml_dataset.csv")
df=df.dropna()

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#A6(i)
labels =np.unique(y)
X= X[(y == labels[0]) | (y == labels[1])]
y= y[(y == labels[0]) | (y == labels[1])]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

param_grid = {'n_neighbors' : list(range(1,12))}
knn= KNeighborsClassifier()
grid= GridSearchCV(knn,param_grid,cv=5)
grid.fit(X_train,y_train)

print("Ideal k value :", grid.best_params_['n_neighbors'])
