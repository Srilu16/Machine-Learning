import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataset.csv")
df = df.dropna()

X = df.drop(columns=["LABEL"])
y = df["LABEL"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_models = [
    ('svm', SVC(probability=True)),
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier()),
    ('ada', AdaBoostClassifier()),
    ('nb', GaussianNB()),
    ('mlp', MLPClassifier(max_iter=500))
]

meta_models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

results = []

for name, meta_model in meta_models.items():
    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3
    )
    
    stack.fit(X_train, y_train)
    
    y_pred = stack.predict(X_test)
    
    results.append([
        name,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='weighted', zero_division=0),
        recall_score(y_test, y_pred, average='weighted', zero_division=0),
        f1_score(y_test, y_pred, average='weighted', zero_division=0)
    ])

results_df = pd.DataFrame(results, columns=[
    "Meta Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

print(results_df)
