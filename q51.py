import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1].values  

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

X_single = X_train[:, 0].reshape(-1, 1)

y_single = X_train[:, 1]

reg = LinearRegression()
reg.fit(X_single, y_single)

y_pred = reg.predict(X_single)

print("Slope (Coefficient):", reg.coef_[0])
print("Intercept:", reg.intercept_)

plt.figure(figsize=(6,6))
plt.scatter(X_single, y_single)
plt.plot(X_single, y_pred)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
