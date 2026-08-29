# Cardiac Risk Monitor

A web app built from `heartdiseaseprediction.ipynb`. It reproduces the notebook's model
comparison (Logistic Regression vs. Random Forest vs. XGBoost), automatically picks the
best one by ROC-AUC, pickles it, and serves it behind a Flask app with a cardiac-monitor
themed risk assessment form.

## Contents

```
heart_disease_app/
├── heart disease.csv     # dataset (same one used in the notebook)
├── train_model.py        # trains all 3 models, picks the best, saves model.pkl
├── model.pkl              # pickled best model + scaler + metadata (generated)
├── app.py                 # Flask server (loads model.pkl, serves form + /predict API)
├── templates/index.html   # risk assessment form UI
├── static/style.css       # cardiac-monitor styling
└── static/script.js       # form submission + gauge animation
```

## Setup

```bash
pip install flask scikit-learn pandas numpy xgboost
```

## 1. Train the model (creates model.pkl)

```bash
python train_model.py
```

This trains Logistic Regression (on scaled features), Random Forest, and XGBoost —
same as the notebook — compares them on Accuracy / Precision / Recall / F1 / ROC-AUC,
and picks the highest-ROC-AUC model. On this dataset that's usually **Random Forest**
(~91% ROC-AUC, ~82% accuracy). The winning model, its scaler (used only if Logistic
Regression wins), and the form's dropdown/range metadata all get pickled into
`model.pkl`.

## 2. Run the website

```bash
python app.py
```

Then open **http://127.0.0.1:5001** (note: different port from the student-performance
app, so you can run both at once). Fill in the patient's vitals and test results and
click **"Assess risk"** — the app calls `/predict`, which loads `model.pkl` and returns
a risk probability, shown on an animated gauge that turns green (low risk) or red
(high risk).

## Notes

- This is a demo/educational tool trained on a small (303-row) public dataset. It is
  **not a diagnostic tool** and shouldn't be used for real medical decisions.
- To retrain on updated data, replace `heart disease.csv` and rerun `train_model.py`.
- For a real deployment, swap Flask's dev server for a production WSGI server (e.g.
  `waitress` on Windows, or `gunicorn app:app` on Linux/Mac).
