import pandas as pd
import numpy as np

df = pd.read_csv("thy.csv")
binary_cols = df.columns[df.nunique() == 2]
binary_data = df[binary_cols]

vec1 = binary_data.iloc[0]
vec2 = binary_data.iloc[1]

f11 = np.sum((vec1 == 1) & (vec2 == 1))
f00 = np.sum((vec1 == 0) & (vec2 == 0))
f10 = np.sum((vec1 == 1) & (vec2 == 0))
f01 = np.sum((vec1 == 0) & (vec2 == 1))

if (f11 + f10 + f01) == 0:
    JC = 0
else:
    JC = f11 / (f11 + f10 + f01)

if (f11 + f10 + f01 + f00) == 0:
    SMC = 0
else:
    SMC = (f11 + f00) / (f11 + f10 + f01 + f00)

print(f"f11={f11}, f00={f00}, f10={f10}, f01={f01}")
print(f"Jaccard Coefficient: {JC:.3f}")
print(f"Simple Matching Coefficient: {SMC:.3f}")
