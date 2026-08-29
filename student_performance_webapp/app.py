"""
app.py — Flask web app that loads model.pkl and serves a prediction form.
Run with:  python3 app.py   (then open http://127.0.0.1:5000)
"""

import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    ARTIFACT = pickle.load(f)

PIPELINE = ARTIFACT["pipeline"]
NUM_COLS = ARTIFACT["num_cols"]
CAT_COLS = ARTIFACT["cat_cols"]
CAT_OPTIONS = ARTIFACT["cat_options"]
NUM_RANGES = ARTIFACT["num_ranges"]
METRICS = ARTIFACT["metrics"]

# Friendly labels + short helper text for the form
FIELD_META = {
    "Hours_Studied": {"label": "Hours studied / week", "help": "Average weekly study hours"},
    "Attendance": {"label": "Attendance (%)", "help": "Class attendance rate"},
    "Sleep_Hours": {"label": "Sleep hours / night", "help": "Average nightly sleep"},
    "Previous_Scores": {"label": "Previous exam score", "help": "Most recent prior score (0-100)"},
    "Tutoring_Sessions": {"label": "Tutoring sessions / month", "help": "Extra tutoring per month"},
    "Physical_Activity": {"label": "Physical activity (hrs/week)", "help": "Weekly exercise hours"},
    "Parental_Involvement": {"label": "Parental involvement"},
    "Access_to_Resources": {"label": "Access to resources"},
    "Extracurricular_Activities": {"label": "Extracurricular activities"},
    "Motivation_Level": {"label": "Motivation level"},
    "Internet_Access": {"label": "Internet access"},
    "Family_Income": {"label": "Family income"},
    "Teacher_Quality": {"label": "Teacher quality"},
    "School_Type": {"label": "School type"},
    "Peer_Influence": {"label": "Peer influence"},
    "Learning_Disabilities": {"label": "Learning disabilities"},
    "Parental_Education_Level": {"label": "Parental education level"},
    "Distance_from_Home": {"label": "Distance from home"},
    "Gender": {"label": "Gender"},
}


@app.route("/")
def index():
    return render_template(
        "index.html",
        num_cols=NUM_COLS,
        cat_cols=CAT_COLS,
        cat_options=CAT_OPTIONS,
        num_ranges=NUM_RANGES,
        field_meta=FIELD_META,
        metrics=METRICS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        row = {}
        for col in NUM_COLS:
            row[col] = [float(data[col])]
        for col in CAT_COLS:
            row[col] = [data[col]]
        X = pd.DataFrame(row)[NUM_COLS + CAT_COLS]

        pred = PIPELINE.predict(X)[0]
        pred_clamped = max(0, min(100, round(float(pred), 1)))

        return jsonify({"ok": True, "prediction": pred_clamped})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
