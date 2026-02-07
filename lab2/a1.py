import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("purchase_data.csv")
X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = df["Payment (Rs)"].values
print(np.linalg.matrix_rank(X))
print(np.linalg.pinv(X) @ y)
