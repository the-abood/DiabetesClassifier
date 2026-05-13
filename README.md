# 🩺 DiabetesClassifier — Predicting Diabetes Diagnosis with Machine Learning

A **scikit-learn** classification project that predicts whether a patient will be diagnosed with diabetes, trained on a 100,000-record synthetic healthcare dataset with 31 clinical, demographic, and lifestyle features.

Two classifiers — **Random Forest** and **Logistic Regression** — are benchmarked end-to-end: from raw data through EDA, preprocessing, hyperparameter tuning, and evaluation, with full interpretability analysis and actionable healthcare recommendations.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Examples](#examples)
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

## Examples

### Example 1 — Class Distribution Check

```python
ax = sns.countplot(data=df, x='diagnosed_diabetes', palette='Set2')
ax.set_title('Target Class Distribution')
ax.set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'])
plt.tight_layout()
```

The dataset is mildly imbalanced (~60/40). Class weights are computed and passed to both models to prevent bias toward the majority class.

---

### Example 2 — Correlation Heatmap

```python
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Heatmap')
```

Strongest positive correlations with the target: `hba1c`, `glucose_fasting`, `glucose_postprandial`, `diabetes_risk_score`. These align with established clinical diagnostic criteria (e.g. HbA1c ≥ 6.5% is a WHO diagnostic threshold).

---

### Example 3 — Preprocessing: Encoding + Scaling + Split

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Encode categoricals
categorical_cols = ['gender','ethnicity','education_level',
                    'income_level','employment_status','smoking_status']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Drop multicollinear / leakage columns
df.drop(columns=['diabetes_stage'], inplace=True)

# Features / target
X = df.drop(columns=['diagnosed_diabetes'])
y = df['diagnosed_diabetes']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Class weights
cw = compute_class_weight('balanced', classes=[0, 1], y=y)
class_weights = {0: cw[0], 1: cw[1]}

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

> **Note on `diabetes_stage`:** This column is a direct encoding of the target variable (`Type 2`, `No Diabetes`, etc.) and would constitute target leakage if retained. It is dropped before modelling.

---

### Example 4 — Random Forest with GridSearchCV

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(class_weight=class_weights, random_state=42, n_jobs=-1)

param_grid_rf = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

gs_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc',
                     n_jobs=-1, verbose=1)
gs_rf.fit(X_train, y_train)

best_rf = gs_rf.best_estimator_
print(f"Best RF params : {gs_rf.best_params_}")
print(f"CV ROC-AUC     : {gs_rf.best_score_:.4f}")
```

**Why Random Forest?** Ensemble of decision trees that handles non-linear relationships, mixed feature types, and is robust to outliers without requiring feature scaling. Built-in Gini importance provides free interpretability.

---

### Example 5 — Logistic Regression with GridSearchCV

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(class_weight=class_weights, random_state=42, max_iter=1000)

param_grid_lr = {
    'C':      [0.001, 0.01, 0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear'],
}

gs_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='roc_auc',
                     n_jobs=-1, verbose=1)
gs_lr.fit(X_train, y_train)

best_lr = gs_lr.best_estimator_
print(f"Best LR params : {gs_lr.best_params_}")
print(f"CV ROC-AUC     : {gs_lr.best_score_:.4f}")
```

**Why Logistic Regression?** Linear, probabilistic, and highly interpretable — coefficients directly show the direction and magnitude of each feature's effect. Serves as a strong, transparent baseline to contrast against the RF ensemble.

---

### Example 6 — Evaluation: Classification Report & ROC-AUC

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

for name, model in [('Random Forest', best_rf), ('Logistic Regression', best_lr)]:
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred,
                                 target_names=['No Diabetes', 'Diabetes']))
    print(f"ROC-AUC: {auc:.4f}")
```

---

### Example 7 — ROC Curve Comparison

```python
fig, ax = plt.subplots(figsize=(8, 6))
for name, model, colour in [
    ('Random Forest',     best_rf, 'steelblue'),
    ('Logistic Regression', best_lr, 'tomato'),
]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{name}  (AUC = {auc:.3f})', color=colour, lw=2)

ax.plot([0,1],[0,1], 'k--', lw=1, label='Random baseline')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/roc_comparison.png', dpi=150)
```

---

### Example 8 — Random Forest Feature Importance

```python
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top_20 = importances.nlargest(20).sort_values()

top_20.plot(kind='barh', color='steelblue', figsize=(10, 7))
plt.title('Top 20 Features — Random Forest Gini Importance',
          fontsize=13, fontweight='bold')
plt.xlabel('Mean Decrease in Gini Impurity')
plt.tight_layout()
plt.savefig('outputs/figures/rf_feature_importance.png', dpi=150)
```

Top drivers: `hba1c`, `glucose_fasting`, `diabetes_risk_score`, `glucose_postprandial`, `insulin_level` — consistent with clinical knowledge.

---

### Example 9 — Logistic Regression Coefficient Plot

```python
coef_df = pd.DataFrame({
    'Feature':     X.columns,
    'Coefficient': best_lr.coef_[0]
}).sort_values('Coefficient', ascending=False)

coef_df['Color'] = coef_df['Coefficient'].apply(
    lambda x: 'steelblue' if x > 0 else 'indianred'
)

plt.figure(figsize=(10, 7))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=coef_df['Color'])
plt.axvline(0, color='black', linewidth=1)
plt.title('Logistic Regression Feature Coefficients',
          fontsize=13, fontweight='bold')
plt.xlabel('Coefficient Value')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/figures/lr_coefficients.png', dpi=150)
```

Blue bars push toward a positive diagnosis; red bars push against. `hba1c`, `glucose_fasting`, and `diabetes_risk_score` are the strongest positive predictors in the linear model.

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
