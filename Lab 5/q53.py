import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("dataset.csv")
df = df.dropna()

data = df.iloc[:, :-1].values

target_index = 1

y = data[:, target_index]
X = np.delete(data, target_index, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("----- TRAIN METRICS -----")
print("MSE :", train_mse)
print("RMSE:", train_rmse)
print("MAPE:", train_mape)
print("R2  :", train_r2)

print("\n----- TEST METRICS -----")
print("MSE :", test_mse)
print("RMSE:", test_rmse)
print("MAPE:", test_mape)
print("R2  :", test_r2)
