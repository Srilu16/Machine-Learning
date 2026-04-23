import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1]

corr_matrix = X.corr()

plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
