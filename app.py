"""Streamlit interface: evaluate saved classifiers on an uploaded labelled CSV."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "diagnosis"
CLASS_TO_NUMBER = {"malignant": 0, "benign": 1}
NUMBER_TO_CLASS = {0: "malignant", 1: "benign"}
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Lab", page_icon="🧪", layout="wide")
st.title("🧪 Breast Cancer Classification Lab")
st.write("Compare five trained machine-learning classifiers on a labelled test CSV.")
st.info("Upload held-out test data containing all 30 feature columns and a 'diagnosis' target column. This app only loads saved models; it never retrains them.")

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run `python train.py` first.")
        loaded[name] = joblib.load(path)
    return loaded

try:
    models = load_models()
except Exception as error:
    st.error(f"Could not load trained models: {error}")
    st.stop()

sample_path = ROOT / "test_data.csv"
with st.sidebar:
    st.header("Evaluation settings")
    st.caption("You can use the provided test_data.csv or upload another labelled CSV.")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    selected_model = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    st.caption("Positive class used for precision, recall, F1 and AUC: benign")

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Unable to read CSV: {error}")
        st.stop()
elif sample_path.exists():
    data = pd.read_csv(sample_path)
    st.success("Using the included held-out test_data.csv. Upload another CSV from the sidebar to evaluate it instead.")
else:
    st.warning("Upload a labelled test CSV to begin.")
    st.stop()

if TARGET not in data.columns:
    st.error("The CSV must include a 'diagnosis' target column with values 'malignant' or 'benign'.")
    st.stop()

feature_columns = list(models[selected_model].feature_names_in_) if hasattr(models[selected_model], "feature_names_in_") else None
# Pipelines may not expose feature_names_in_ consistently across sklearn versions.
if feature_columns is None:
    metadata_path = ROOT / "dataset_metadata.json"
    if metadata_path.exists():
        import json
        feature_columns = json.loads(metadata_path.read_text())["feature_columns"]
    else:
        st.error("Feature metadata is missing. Run train.py again.")
        st.stop()

missing = [column for column in feature_columns if column not in data.columns]
if missing:
    st.error(f"CSV is missing {len(missing)} required feature column(s): {', '.join(missing)}")
    st.stop()

unknown = sorted(set(data[TARGET].dropna().astype(str).str.lower()) - set(CLASS_TO_NUMBER))
if unknown:
    st.error(f"Unknown target value(s): {unknown}. Use 'malignant' or 'benign'.")
    st.stop()
if data[TARGET].isna().any():
    st.error("The target column contains missing values. Please provide labels for every row.")
    st.stop()

X = data[feature_columns].apply(pd.to_numeric, errors="coerce")
if X.isna().any().any():
    st.error("Feature columns must contain numeric values and no missing/non-numeric cells.")
    st.stop()
y_true = data[TARGET].astype(str).str.lower().map(CLASS_TO_NUMBER)

st.subheader("Uploaded test data")
col1, col2, col3 = st.columns(3)
col1.metric("Test records", f"{len(data):,}")
col2.metric("Input features", len(feature_columns))
col3.metric("Target classes", data[TARGET].nunique())
st.dataframe(data.head(10), use_container_width=True)

# Evaluate all five saved models so their test-set results are visible together.
all_rows = []
all_predictions = {}
for name, estimator in models.items():
    pred = estimator.predict(X)
    proba = estimator.predict_proba(X)[:, 1]
    all_predictions[name] = pred
    all_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, proba),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    })
comparison = pd.DataFrame(all_rows).sort_values("F1", ascending=False)

st.subheader("Model comparison on this test CSV")
st.dataframe(comparison.style.format({c: "{:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}), use_container_width=True)
st.caption("AUC uses the predicted probability of the benign class. All other metrics use the predicted class label.")

st.subheader(f"Detailed results: {selected_model}")
selected = comparison.set_index("Model").loc[selected_model]
metric_cols = st.columns(6)
for col, metric in zip(metric_cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    col.metric(metric, f"{selected[metric]:.4f}")

predictions = all_predictions[selected_model]
cm = confusion_matrix(y_true, predictions, labels=[0, 1])
cm_df = pd.DataFrame(cm, index=["Actual malignant", "Actual benign"], columns=["Predicted malignant", "Predicted benign"])
left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    st.dataframe(cm_df, use_container_width=True)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y_true, predictions, labels=[0, 1], target_names=["malignant", "benign"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

output = data.copy()
output["predicted_diagnosis"] = [NUMBER_TO_CLASS[int(value)] for value in predictions]
output["predicted_probability_benign"] = models[selected_model].predict_proba(X)[:, 1]
st.download_button("Download predictions as CSV", output.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
