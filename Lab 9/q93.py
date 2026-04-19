import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from lime.lime_tabular import LimeTabularExplainer

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.drop(columns=["LABEL"])
y = df["LABEL"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(probability=True))
])

pipeline.fit(X_train, y_train)

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns.tolist(),
    class_names=[str(c) for c in np.unique(y)],
    mode='classification'
)

exp = explainer.explain_instance(
    X_test.iloc[0].values,
    pipeline.predict_proba
)

print(exp.as_list())
