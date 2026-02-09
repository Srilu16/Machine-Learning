import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("price.csv")
df.columns = df.columns.str.strip()

df['Price'] = df['Price'].str.replace(',', '').astype(float)
df = df.dropna(subset=['Price'])

y_actual = df['Price'].values

mean_price=np.mean(y_actual)
y_pred=np.full(len(y_actual),mean_price)

mse=mean_squared_error(y_actual,y_pred)
rmse=np.sqrt(mse)
mape=np.mean(np.abs((y_actual - y_pred ) / y_actual)) * 100
r2 = r2_score(y_actual,y_pred)

print("MSE : ",mse)
print("RMSE : ",rmse)
print("MAPE : ",mape)
print("R2 : ",r2)
