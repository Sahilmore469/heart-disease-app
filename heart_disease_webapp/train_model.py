"""
train_model.py
---------------
Reproduces heartdiseaseprediction.ipynb: trains Logistic Regression, Random Forest,
and XGBoost on the UCI heart disease dataset, compares them (Accuracy / Precision /
Recall / F1 / ROC-AUC), and pickles the BEST model (plus its scaler, if it needs one,
plus feature metadata for the web form) into model.pkl.
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("heart disease.csv")
print("Dataset shape:", df.shape)

X = df.drop("target", axis=1)
y = df["target"]
FEATURE_COLS = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 2. Train & compare models (same 3 as the notebook)
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

results = []
trained_models = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    })
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\nModel Comparison")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Pick the best model (highest ROC-AUC), same as the notebook
# ---------------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
needs_scaling = best_model_name == "Logistic Regression"
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------------------------
# 4. Feature metadata for the web form
# ---------------------------------------------------------------------------
NUMERIC_RANGE_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
    "fbs": {0: "\u2264 120 mg/dl", 1: "> 120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"},
    "thal": {0: "Unknown", 1: "Fixed defect", 2: "Normal", 3: "Reversible defect"},
}

num_ranges = {c: (float(df[c].min()), float(df[c].max())) for c in NUMERIC_RANGE_COLS}

# ---------------------------------------------------------------------------
# 5. Save everything the app needs
# ---------------------------------------------------------------------------
artifact = {
    "model": best_model,
    "model_name": best_model_name,
    "scaler": scaler,
    "needs_scaling": needs_scaling,
    "feature_cols": FEATURE_COLS,
    "numeric_range_cols": NUMERIC_RANGE_COLS,
    "num_ranges": num_ranges,
    "categorical_cols": CATEGORICAL_COLS,
    "metrics": results_df.to_dict(orient="records"),
    "best_metrics": results_df.iloc[0].to_dict(),
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nSaved model.pkl")
