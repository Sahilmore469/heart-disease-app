"""
app.py — Flask web app that loads model.pkl and serves a heart-disease risk form.
Run with:  python app.py   (then open http://127.0.0.1:5001)
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    ARTIFACT = pickle.load(f)

MODEL = ARTIFACT["model"]
MODEL_NAME = ARTIFACT["model_name"]
SCALER = ARTIFACT["scaler"]
NEEDS_SCALING = ARTIFACT["needs_scaling"]
FEATURE_COLS = ARTIFACT["feature_cols"]
NUMERIC_RANGE_COLS = ARTIFACT["numeric_range_cols"]
NUM_RANGES = ARTIFACT["num_ranges"]
CATEGORICAL_COLS = ARTIFACT["categorical_cols"]
BEST_METRICS = ARTIFACT["best_metrics"]

FIELD_META = {
    "age": {"label": "Age", "help": "Years"},
    "trestbps": {"label": "Resting blood pressure", "help": "mm Hg on admission"},
    "chol": {"label": "Serum cholesterol", "help": "mg/dl"},
    "thalach": {"label": "Max heart rate achieved", "help": "bpm during exercise test"},
    "oldpeak": {"label": "ST depression (oldpeak)", "help": "Induced by exercise, relative to rest"},
    "sex": {"label": "Sex"},
    "cp": {"label": "Chest pain type"},
    "fbs": {"label": "Fasting blood sugar"},
    "restecg": {"label": "Resting ECG results"},
    "exang": {"label": "Exercise-induced angina"},
    "slope": {"label": "Slope of peak exercise ST segment"},
    "ca": {"label": "Major vessels colored by fluoroscopy"},
    "thal": {"label": "Thalassemia"},
}


@app.route("/")
def index():
    return render_template(
        "index.html",
        numeric_cols=NUMERIC_RANGE_COLS,
        num_ranges=NUM_RANGES,
        categorical_cols=CATEGORICAL_COLS,
        field_meta=FIELD_META,
        model_name=MODEL_NAME,
        metrics=BEST_METRICS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        row = {col: [float(data[col])] for col in FEATURE_COLS}
        X = pd.DataFrame(row)[FEATURE_COLS]

        if NEEDS_SCALING:
            X_input = SCALER.transform(X)
        else:
            X_input = X

        proba = float(MODEL.predict_proba(X_input)[0, 1])
        pred = int(proba >= 0.5)

        return jsonify({
            "ok": True,
            "prediction": pred,
            "probability": round(proba * 100, 1),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
