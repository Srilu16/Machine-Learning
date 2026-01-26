import pandas as pd
import numpy as np

df = pd.read_csv("thy.csv")
print(df.head())

print(df.dtypes)

attribute_types = {}
for col in df.columns:
    if df[col].dtype == 'object':
        n_unique = df[col].nunique()
        if n_unique <= 5:
            attr_type = 'Ordinal (Categorical)'
        else:
            attr_type = 'Nominal (Categorical)'
    else:
        attr_type = 'Numeric (Continuous)'
    attribute_types[col] = attr_type

for col, t in attribute_types.items():
    print(f"{col}: {t}")

for col, t in attribute_types.items():
    if 'Categorical' in t:
        unique_vals = df[col].unique()
        if 'Ordinal' in t:
            encoding = 'Label Encoding'
        else:
            encoding = 'One-Hot Encoding'
        print(f"{col}: {encoding}, Unique Values: {unique_vals}")

numeric_cols = [col for col, t in attribute_types.items() if t.startswith('Numeric')]
for col in numeric_cols:
    print(f"{col}: min={df[col].min()}, max={df[col].max()}")

missing = df.isnull().sum()
print(missing)

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers found")

for col in numeric_cols:
    mean_val = df[col].mean()
    var_val = df[col].var()
    std_val = df[col].std()
    print(f"{col}: mean={mean_val:.3f}, variance={var_val:.3f}, std_dev={std_val:.3f}")
