import pandas as pd
import numpy as np

df = pd.read_csv("thy.csv")
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)
vec1 = df.iloc[0].values
vec2 = df.iloc[1].values
cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print(f"Cosine Similarity: {cos_sim:.3f}")
