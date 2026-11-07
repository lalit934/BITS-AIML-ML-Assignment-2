# Machine Learning Assignment 2 — Classification Model Comparison

## a. Problem statement
Build and compare five supervised classification algorithms for predicting whether a breast tumour is malignant or benign. The same training/test split is used for all five models. The deployed Streamlit application loads saved estimators and evaluates a labelled test CSV; it does not train models on uploaded data.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository.
- **Access in code:** `sklearn.datasets.load_breast_cancer`, which distributes this UCI dataset.
- **Size:** 569 instances and 30 numeric input features (meets the assignment minimums of 500 instances and 12 features).
- **Target:** `diagnosis`; `malignant` or `benign`.
- **Split:** 80% training / 20% held-out testing, stratified, `random_state=42`.
- **Positive class for binary metrics:** benign.
- The dataset contains computed cell-nucleus measurements from digitized fine-needle aspirate images. It is used here only as an academic classification dataset, not for medical diagnosis.

## c. GitHub Repository Link
Add your repository URL here after creating and pushing your own GitHub repository: **[PASTE YOUR GITHUB REPOSITORY URL]**.

## d. Models used and evaluation
The five models are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine. Accuracy, ROC AUC, precision, recall, F1 and Matthews Correlation Coefficient (MCC) are computed on the same held-out test set. Run `python train.py` to generate `training_metrics.csv`; copy its results into the table below for your final report.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance
The following observations are based on the fixed 80/20 stratified split (`random_state=42`) produced by `train.py`. Results can vary if you change the split or model settings.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | It achieved 0.9825 accuracy, 0.9954 AUC and 0.9861 F1. Precision and recall are both 0.9861, indicating a balanced result on this test split. |
| Decision Tree | It achieved 0.9211 accuracy and 0.9362 F1, the lowest overall accuracy/F1 among the five models in this experiment. The depth limit of 5 helps constrain tree complexity. |
| KNN | It achieved 0.9561 accuracy, 0.9788 AUC and 0.9655 F1. Scaling is important because KNN uses distances; five neighbours are used. |
| Naive Bayes | It achieved 0.9298 accuracy, 0.9868 AUC and 0.9444 F1. Its AUC is strong, but its threshold-based accuracy/F1 are lower than Logistic Regression and SVM. |
| Support Vector Machine | It tied Logistic Regression on accuracy (0.9825), precision (0.9861), recall (0.9861), F1 (0.9861) and MCC (0.9623); its AUC was 0.9950, slightly below Logistic Regression. |
| **Overall winner for this dataset** | Logistic Regression is the overall winner for this split: it ties SVM on accuracy, F1 and MCC, and has the highest AUC (0.9954). SVM is a very close second. |

## Project files
- `train.py` — loads the dataset, splits it, trains/evaluates all five models, saves the models and exports the held-out test CSV.
- `app.py` — Streamlit interface for uploading labelled test data, comparing all models, viewing detailed metrics/confusion matrix/classification report, and downloading predictions.
- `model/` — saved trained estimators (`.joblib`).
- `test_data.csv` — held-out test data including the target column.
- `training_metrics.csv` — measured training/test-split metrics.
- `dataset_metadata.json` — feature and dataset metadata used by the app.

## Run locally or on BITS Virtual Lab
1. Create and activate a Python virtual environment (optional but recommended).
2. Install dependencies: `pip install -r requirements.txt`.
3. Train and save the models: `python train.py`.
4. Start the app: `streamlit run app.py`.
5. Open the local URL printed by Streamlit (usually `http://localhost:8501`).
6. Upload `test_data.csv` or another CSV with all 30 feature columns and the `diagnosis` column.

## Deploy on Streamlit Community Cloud
1. Push this complete project folder to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **New app**, choose your repository and branch, and set the main file path to `app.py`.
4. Deploy. The repository must include the saved `model/*.joblib` files and `dataset_metadata.json`.
5. Test the live URL using `test_data.csv`, then add the live app URL here: **[PASTE YOUR STREAMLIT APP URL]**.

## Assignment submission checklist
- [ ] GitHub repository URL works and contains all source/model/data/dependency files.
- [ ] Live Streamlit app URL opens and works.
- [ ] App shows the five-model comparison and selected-model metrics/report.
- [ ] Screenshot captured while executing the assignment on BITS Virtual Lab.
- [ ] This README content and the two working links are included in the single PDF submission.

**Important:** The assignment asks for a screenshot from BITS Virtual Lab. A local run or Streamlit Cloud deployment is not a substitute for that required screenshot. Also, make your own Git commits, review the code, and write your own observations; the assignment explicitly warns against direct copy-paste submissions.
