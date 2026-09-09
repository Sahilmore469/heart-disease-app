<div align="center">

# ❤️ Cardiac Risk Monitor

**A Flask web app that predicts heart disease risk from patient vitals — trained, compared, and served end-to-end.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat-square&logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

[Live Demo](#) · [Report Bug](../../issues) · [Request Feature](../../issues)

</div>

---

## 📋 Overview

Cardiac Risk Monitor reproduces the model-comparison workflow from [`heartdiseaseprediction.ipynb`](./heart_disease_webapp/heartdiseaseprediction.ipynb) — training **Logistic Regression**, **Random Forest**, and **XGBoost** on the Cleveland heart disease dataset, automatically selecting the best model by ROC-AUC, and serving it behind a clean, cardiac-monitor-themed risk assessment form.

Enter a patient's age, chest pain type, cholesterol, ECG results, and other clinical values, and get an instant, interpretable risk estimate.

## ✨ Features

- 🩺 **13-field clinical intake form** — the same features used in the original Cleveland dataset
- 🧠 **Three models compared automatically**, best one selected by ROC-AUC and pickled for serving
- ⚡ **Instant predictions** — no page reloads, results appear as soon as you submit
- 📊 **Transparent performance** — accuracy, precision, recall, and ROC-AUC reported for every model, not just the winner
- 🚀 **One-click deploy** — ships with everything Render needs out of the box

## 🏆 Model Performance

Held-out test set (20% split, stratified):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** 🥇 | **82.0%** | 76.2% | 97.0% | 85.3% | **0.908** |
| Logistic Regression | 80.3% | 76.9% | 90.9% | 83.3% | 0.869 |
| XGBoost | 80.3% | 75.6% | 93.9% | 83.8% | 0.856 |

> Random Forest is selected automatically at training time and saved to `model.pkl` — high recall was prioritized since missing a true positive (an at-risk patient) is costlier than a false alarm in a screening context.

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Model training | Python, pandas, scikit-learn, XGBoost |
| Web app | Flask, gunicorn |
| Dataset | [UCI / Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (303 records, 13 features) |
| Deployment | Render |

## 📁 Project Structure

```
heart-disease-app/
└── heart_disease_webapp/
    ├── app.py                        # Flask app — loads model.pkl, serves the form & predictions
    ├── train_model.py                # Trains & compares all three models, pickles the best one
    ├── model.pkl                     # Serialized best-performing model (Random Forest)
    ├── heart disease.csv             # Dataset (same one used in the notebook)
    ├── heartdiseaseprediction.ipynb  # Original exploration & model-comparison notebook
    ├── requirements.txt              # Python dependencies
    └── README.md
```

## 🚀 Getting Started

### Run locally

```bash
# Clone the repo
git clone https://github.com/Sahilmore469/heart-disease-app.git
cd heart-disease-app/heart_disease_webapp

# Install dependencies
pip install -r requirements.txt

# (Optional) retrain the model from scratch
python train_model.py

# Start the app
python app.py
```

Then open `http://localhost:5000` in your browser.

### Deploy on Render

1. Push this repo to GitHub.
2. On Render, create a **New Web Service** from the repo.
3. Set **Root Directory** to `heart_disease_webapp`.
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

That's it — Render handles the rest.

## 📊 Input Features

<details>
<summary>Click to expand the 13 clinical fields used by the model</summary>

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Max heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0–4) colored by fluoroscopy |
| `thal` | Thalassemia result |

</details>

## ⚠️ Disclaimer

This project is an **educational demo** built on a public research dataset. It is **not a diagnostic tool** and should never replace evaluation by a qualified medical professional.

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">

Built as part of an academic project on ML-based clinical decision support.

</div>
