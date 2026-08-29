# Exam Score Predictor

A web app built from `Student_Performance_Analysis_kaggle_dataset.ipynb`. It reproduces the notebook's
**Extended Linear Regression model** (all 19 features, one-hot encoded), pickles the fitted pipeline,
and serves it behind a small Flask app with a chalkboard/gradebook-themed prediction form.

## Contents

```
student_performance_app/
├── StudentPerformanceFactors.csv   # dataset (same one used in the notebook)
├── train_model.py                  # cleans data, trains the pipeline, saves model.pkl
├── model.pkl                       # pickled sklearn Pipeline + metadata (generated)
├── app.py                          # Flask server (loads model.pkl, serves form + /predict API)
├── templates/index.html            # prediction form UI
├── static/style.css                # chalkboard / report-card styling
└── static/script.js                # form submission + result animation
```

## Setup

```bash
pip install flask scikit-learn pandas numpy
```

## 1. Train the model (creates model.pkl)

```bash
python3 train_model.py
```

This cleans the dataset the same way the notebook does (drops the one `Exam_Score > 100` outlier,
fills missing `Teacher_Quality` / `Parental_Education_Level` / `Distance_from_Home` with the mode),
fits a `ColumnTransformer(OneHotEncoder) + LinearRegression` pipeline on all 19 features, and pickles
the whole pipeline — plus the dropdown options and numeric ranges needed to build the form — into
`model.pkl`.

Expect around **R² ≈ 0.82**, matching the notebook's "Extended Model" results.

## 2. Run the website

```bash
python3 app.py
```

Then open **http://127.0.0.1:5000** in your browser. Fill in the study-sheet form and click
**"Grade this student"** — the app calls `/predict`, which loads `model.pkl` and returns a predicted
exam score, shown circled in red pen like a graded paper.

## Notes

- `model.pkl` bundles the *entire* pipeline (preprocessing + model), so `app.py` only needs to build a
  raw-feature DataFrame and call `.predict()` — no manual encoding required.
- To retrain on updated data, just replace `StudentPerformanceFactors.csv` and rerun `train_model.py`.
- For a real deployment, swap Flask's dev server for a production WSGI server (e.g. `gunicorn app:app`).
