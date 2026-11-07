"""Train five classifiers on the UCI Breast Cancer Wisconsin (Diagnostic) dataset."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# The scikit-learn copy is the UCI Wisconsin Diagnostic Breast Cancer dataset.
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()  # 0 = malignant, 1 = benign

# Keep the target human-readable in the exported test CSV.
class_names = {0: "malignant", 1: "benign"}
y_named = y.map(class_names).rename("diagnosis")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ]),
}

rows = []
for display_name, estimator in models.items():
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    scores = {
        "Model": display_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    rows.append(scores)
    filename = display_name.lower().replace(" ", "_").replace("-", "") + ".joblib"
    joblib.dump(estimator, MODEL_DIR / filename)
    print(f"Saved {display_name}: model/{filename}")

metrics = pd.DataFrame(rows).sort_values("F1", ascending=False)
metrics.to_csv(ROOT / "training_metrics.csv", index=False)

# Export only held-out test rows, with the target included as required by the brief.
test_export = X_test.copy()
test_export["diagnosis"] = y_test.map(class_names)
test_export.to_csv(ROOT / "test_data.csv", index=False)

metadata = {
    "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
    "source": "UCI Machine Learning Repository; distributed in scikit-learn",
    "instances": int(X.shape[0]),
    "features": int(X.shape[1]),
    "target_column": "diagnosis",
    "target_classes": ["malignant", "benign"],
    "positive_class_for_metrics": "benign",
    "random_state": RANDOM_STATE,
    "test_size": 0.20,
    "feature_columns": list(X.columns),
}
(ROOT / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))
print("\nHeld-out test metrics:")
print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(f"\nCreated {ROOT / 'test_data.csv'} with {len(test_export)} rows.")
