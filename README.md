# 🩺 DiabetesClassifier — Predicting Diabetes Diagnosis with Machine Learning

A **scikit-learn** classification project that predicts whether a patient will be diagnosed with diabetes, trained on a 100,000-record synthetic healthcare dataset with 31 clinical, demographic, and lifestyle features.

Two classifiers — **Random Forest** and **Logistic Regression** — are benchmarked end-to-end: from raw data through EDA, preprocessing, hyperparameter tuning, and evaluation, with full interpretability analysis and actionable healthcare recommendations.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Ethical Considerations](#ethical-considerations)
- [Tech Stack](#tech-stack)

---

## Project Overview

Early detection of diabetes enables timely clinical intervention and significantly improves patient outcomes. This project demonstrates how machine learning can support — not replace — clinical decision-making by identifying high-risk patients from routinely collected health data.

**Business value:** A healthcare provider deploying this model could prioritise screening and preventive care for patients flagged as high risk, reducing long-term complications and treatment costs.

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | Synthetic healthcare dataset (Kaggle) |
| Rows | 100,000 |
| Features | 31 |
| Target | `diagnosed_diabetes` (0 = No, 1 = Yes) |
| Class balance | ~60% positive (diagnosed), ~40% negative |
| Missing values | None |
| Duplicates | None |

### Feature Groups

| Group | Features |
|-------|---------|
| **Demographics** | age, gender, ethnicity, education_level, income_level, employment_status |
| **Lifestyle** | smoking_status, alcohol_consumption_per_week, physical_activity_minutes_per_week, diet_score, sleep_hours_per_day, screen_time_hours_per_day |
| **Medical history** | family_history_diabetes, hypertension_history, cardiovascular_history |
| **Anthropometric** | bmi, waist_to_hip_ratio |
| **Vitals** | systolic_bp, diastolic_bp, heart_rate |
| **Blood panel** | cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides, glucose_fasting, glucose_postprandial, insulin_level, hba1c |
| **Risk scores** | diabetes_risk_score, diabetes_stage |

---

## Pipeline

```
Raw CSV
   │
   ▼
1. EDA ──────────── class distribution, correlation heatmap,
   │                feature distributions, outlier detection
   ▼
2. Preprocessing ── LabelEncoder for categoricals, StandardScaler
   │                for numericals, class weight computation,
   │                80/20 stratified train/test split
   ▼
3. Model A ──────── Random Forest (GridSearchCV)
   │                n_estimators, max_depth, min_samples_split
   ▼
4. Model B ──────── Logistic Regression (GridSearchCV)
   │                C, solver, max_iter
   ▼
5. Evaluation ───── Accuracy, F1, ROC-AUC, confusion matrix,
   │                classification report, ROC curve overlay
   ▼
6. Interpretability  RF feature importances (Gini), LR coefficients
   │                (polarity-coloured bar chart)
   ▼
7. Outputs ──────── saved models, figures, results CSV
```

---

## Results Summary

| Metric | Random Forest | Logistic Regression |
|--------|--------------|---------------------|
| Accuracy | ~0.92 | ~0.88 |
| F1 (weighted) | ~0.92 | ~0.88 |
| ROC-AUC | ~0.97 | ~0.94 |
| CV ROC-AUC | ~0.97 | ~0.93 |

**Random Forest wins on all metrics** due to its ability to capture non-linear interactions between features. Logistic Regression is faster to train, easier to explain to clinicians, and still achieves strong AUC — making it viable where interpretability is paramount (e.g. regulatory compliance).

---

## Ethical Considerations

- **Dataset sensitivity:** Health data is among the most sensitive personal data categories under GDPR/HIPAA. This project uses a synthetic dataset; real deployment must enforce strict access control and anonymisation.
- **Demographic fairness:** Features like ethnicity, gender, and income are included. Models trained on such features risk amplifying existing healthcare disparities. Fairness metrics (demographic parity, equalised odds) should be computed before any clinical deployment.
- **Human oversight:** Model output should be a *risk flag* surfaced to a clinician — never an autonomous diagnostic decision. A false negative (missed diabetes case) has serious health consequences.
- **Consent and transparency:** Patients whose data trains such models should be informed, and model-based risk scores presented to them in plain language.

---

## Project Structure

```
DiabetesClassifier/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── diabetes_classifier.ipynb    ← Full end-to-end notebook (runnable)
│
├── src/
│   ├── eda/
│   │   └── exploratory_analysis.py  ← EDA helpers and visualisation functions
│   ├── preprocessing/
│   │   └── prepare_data.py          ← Encoding, scaling, splitting helpers
│   ├── models/
│   │   ├── random_forest.py         ← RF grid search wrapper
│   │   └── logistic_regression.py   ← LR grid search wrapper
│   ├── evaluation/
│   │   └── metrics.py               ← Evaluation, confusion matrix, ROC helpers
│   └── visualisation/
│       └── plots.py                 ← All chart functions
│
├── data/
│   └── README.txt                   ← Dataset source and placement instructions
│
├── outputs/
│   ├── figures/                     ← Saved PNG charts
│   └── models/                      ← Serialised model files (.pkl)
│
└── docs/
    └── design_notes.md              ← Algorithm choices, ethical analysis,
                                        business recommendations
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| pandas / numpy | Data loading and manipulation |
| scikit-learn | Preprocessing, models, evaluation |
| matplotlib / seaborn | Visualisation |
| joblib | Model serialisation |

---

## License

MIT — free to use, extend, and adapt.
