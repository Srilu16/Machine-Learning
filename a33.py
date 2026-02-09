import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(42)
X=np.random.uniform(1,10,20)
Y=np.random.uniform(1,10,20)

classes = np.where(X + Y > 10, 1,0)
#if X+Y>10 then Red (class 1) else Blue (class 0)

plt.figure(figsize =(6,6))
plt.scatter(X[classes == 0] , Y[classes == 0] ,color="blue",label='Class 0 (Blue)')
plt.scatter(X[classes == 1], Y[classes == 1], color='red', label='Class 1 (Red)')

plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.show()
