"""
train_model.py
---------------
Reproduces the "Extended Model" from Student_Performance_Analysis_kaggle_dataset.ipynb
(numeric + categorical features -> OneHotEncoder -> LinearRegression) and saves
the fitted pipeline (preprocessing + model bundled together) as model.pkl,
so the Flask app can load one file and call .predict() directly on raw input.
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Load & clean data (same steps as the notebook)
# ---------------------------------------------------------------------------
df = pd.read_csv("StudentPerformanceFactors.csv")

# Remove the one Exam_Score > 100 outlier
df = df[df["Exam_Score"] <= 100]

# Fill missing values with the most common (mode) value, same columns as notebook
for col in ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Cleaned shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Feature setup (matches "4.2 Extended Model" in the notebook)
# ---------------------------------------------------------------------------
num_cols = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
            "Tutoring_Sessions", "Physical_Activity"]
cat_cols = ["Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
            "Motivation_Level", "Internet_Access", "Family_Income", "Teacher_Quality",
            "School_Type", "Peer_Influence", "Learning_Disabilities",
            "Parental_Education_Level", "Distance_from_Home", "Gender"]

target = "Exam_Score"

X = df[num_cols + cat_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 3. Build ONE pipeline (preprocessing + model) so we can pickle a single
#    object that accepts a raw DataFrame at prediction time.
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 4. Evaluate (sanity check against notebook numbers)
# ---------------------------------------------------------------------------
pred = pipeline.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"R^2 Score : {r2:.3f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")

# ---------------------------------------------------------------------------
# 5. Save the whole pipeline + metadata needed to build the web form
# ---------------------------------------------------------------------------
artifact = {
    "pipeline": pipeline,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "cat_options": {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols},
    "num_ranges": {c: (float(df[c].min()), float(df[c].max())) for c in num_cols},
    "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nSaved model.pkl")
