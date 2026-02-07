import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("thy.csv")
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)

df20 = df.iloc[:20]

binary_cols = df20.columns[df20.nunique() == 2]
binary_data = df20[binary_cols].values

def calc_jc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    return f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0

def calc_smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    return (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) != 0 else 0

def calc_cos(v1, v2):
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.dot(v1, v2) / d if d != 0 else 0

JC_matrix = np.zeros((20, 20))
SMC_matrix = np.zeros((20, 20))
COS_matrix = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        JC_matrix[i, j] = calc_jc(binary_data[i], binary_data[j])
        SMC_matrix[i, j] = calc_smc(binary_data[i], binary_data[j])
        COS_matrix[i, j] = calc_cos(df20.iloc[i].values, df20.iloc[j].values)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(JC_matrix, annot=True)
plt.title("Jaccard Coefficient")

plt.subplot(1, 3, 2)
sns.heatmap(SMC_matrix, annot=True)
plt.title("Simple Matching Coefficient")

plt.subplot(1, 3, 3)
sns.heatmap(COS_matrix, annot=True)
plt.title("Cosine Similarity")

plt.tight_layout()
plt.show()
