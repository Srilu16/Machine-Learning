import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("purchase_data.csv")
X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = df["Payment (Rs)"].values
y_class = (y > 200).astype(int)
#1 is rich 0 is poor
model = LogisticRegression()
model.fit(X, y_class)
pred = model.predict(X)
print(pred)
