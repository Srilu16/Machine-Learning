import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

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

explainer_lime = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=[str(i) for i in le.classes_],
    mode='classification'
)

exp = explainer_lime.explain_instance(
    X_test.iloc[0].values,
    pipeline.predict_proba,
    num_features=10
)

print(exp.as_list())

X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)

explainer_shap = shap.KernelExplainer(
    pipeline.named_steps['classifier'].predict_proba,
    X_train_scaled[:100]
)

shap_values = explainer_shap.shap_values(X_test_scaled[:1])

shap.initjs()
shap.force_plot(explainer_shap.expected_value[0], shap_values[0][0], X_test.iloc[0])
