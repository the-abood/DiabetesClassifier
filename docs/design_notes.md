# Design Notes — DiabetesClassifier

## 1. Dataset Description

The dataset is a synthetic healthcare record with 100,000 patients and 31 features
spanning demographics, lifestyle behaviours, medical history, anthropometric
measurements, blood panel results, and composite risk scores. The target column
`diagnosed_diabetes` is binary (0 = No Diabetes, 1 = Diagnosed).

**Source:** Synthetic dataset available on Kaggle.
**Reference:** ClinicalTrials.gov and WHO diagnostic criteria informed the
feature selection and interpretation.

The dataset is fully synthetic — no real patient records are involved —
which makes it suitable for educational and prototyping purposes without
privacy risk. However, models trained on synthetic data may not generalise
to real clinical populations and require validation on real-world data
before any deployment.

---

## 2. Algorithm Selection Rationale

### Random Forest

**Chosen because:**
- Handles non-linear interactions natively (e.g. elevated HbA1c + high BMI
  jointly predicting diagnosis more strongly than either alone)
- No feature scaling required; robust to outliers
- Ensemble averaging reduces variance — more stable predictions than a single
  decision tree, especially important on imbalanced datasets
- Gini feature importance provides built-in attribution
- Consistently outperforms linear models on tabular health data in the literature

**Hyperparameter choices:**
- `n_estimators`: more trees reduce variance at a linear cost; 200-300 is a
  good trade-off for 100k rows
- `max_depth=None` allows full tree growth; regularised alternatives
  (`max_depth=10/20`) prevent single deep paths dominating
- `min_samples_split`: higher values create smoother, more generalised trees

### Logistic Regression

**Chosen because:**
- Provides a transparent linear baseline — coefficients are directly
  interpretable as log-odds changes
- Fast to train and evaluate; well-calibrated probabilities
- Required for regulatory and clinical contexts where "black box" models
  face higher scrutiny
- Highlights which features have a monotone linear relationship with risk,
  complementing the non-linear RF story

**Hyperparameter choices:**
- `C` (inverse regularisation): grid from 0.001 (strong penalty, sparse
  coefficients) to 10.0 (near-unregularised); cross-validation selects the
  optimal bias-variance trade-off
- `solver`: `lbfgs` (quasi-Newton, efficient for L2 and large feature sets)
  vs `liblinear` (coordinate descent, stable for smaller datasets)

---

## 3. Preprocessing Decisions

### LabelEncoder for Categoricals

Categorical columns (`gender`, `ethnicity`, `education_level`, etc.) are
ordinal-encoded rather than one-hot encoded. Rationale:
- RF does not require or benefit from one-hot — it splits on integer thresholds
- With 6 categorical columns at low cardinality (2–5 levels each), one-hot would
  add ~15 dummy columns, increasing dimensionality without benefit
- LR with StandardScaler handles ordinal encoding adequately for low-cardinality
  features, though one-hot would be preferable for a production LR system

### StandardScaler

Applied to all features. Critical for LR (regularisation is scale-dependent);
benign for RF. Fitted on training data only to prevent leakage.

### Dropping `diabetes_stage`

`diabetes_stage` contains values like `Type 2`, `Pre-diabetic`, `No Diabetes` —
a direct re-encoding of the target. Including it would cause target leakage:
the model would learn to read the answer rather than predict from clinical
features. Accuracy would appear near-perfect but would be meaningless.

### Class Weights

The 60/40 imbalance is mild but meaningful in a medical context. Applying
`class_weight='balanced'` re-scales the loss function so misclassifying a
true positive (missed diagnosis) is penalised proportionally to its
under-representation. This improves recall for the positive class.

---

## 4. Evaluation Metric Discussion

### Why ROC-AUC is the primary metric

ROC-AUC measures ranking ability across all classification thresholds and is
robust to class imbalance. A model that ranks every positive case above every
negative case scores 1.0 regardless of the threshold used. It is preferred
over accuracy for health screening tasks.

### Why Recall matters most for the positive class

In diabetes screening:
- **False negative** (missed diagnosis) → patient goes untreated → serious
  long-term complications (kidney failure, cardiovascular disease, blindness)
- **False positive** (unnecessary screening) → follow-up blood test → mild cost

The asymmetric consequence means recall (sensitivity) for the positive class
should be weighted more heavily than precision when setting the operating
threshold in a real deployment. The default 0.5 probability threshold should
be adjusted — lowering it increases recall at the cost of precision.

### Confusion matrix interpretation

For a production clinical tool, the goal is to minimise false negatives
(bottom-left cell) while keeping false positives (top-right cell) manageable.
This translates to: prefer a model with high recall even if precision is
slightly lower.

---

## 5. Critical Evaluation of Results

Random Forest achieves higher ROC-AUC (~0.97 vs ~0.94) by capturing non-linear
interactions that Logistic Regression cannot model linearly. The most important
features for both models — `hba1c`, `glucose_fasting`, `glucose_postprandial`,
`diabetes_risk_score` — align perfectly with established clinical diagnostic
criteria (WHO: HbA1c ≥ 6.5% or fasting glucose ≥ 7.0 mmol/L for T2D diagnosis).

This clinical alignment is reassuring: the model has not learned spurious
correlations but is relying on genuinely predictive signals.

However, the high AUC partially reflects the synthetic nature of the data —
real-world datasets have more noise, missing values, and confounders.
Validation on a real clinical cohort (e.g. NHANES, UK Biobank) would be
necessary before any deployment.

---

## 6. Business & Clinical Recommendations

### 6.1 Risk Stratification for Preventive Care

Deploy the Random Forest model as a **risk stratification tool** embedded in
the EHR (Electronic Health Record) system. Patients with predicted probability
> 0.7 are flagged for priority diabetes screening at their next GP appointment.
This shifts intervention from reactive (treating diagnosed patients) to
proactive (preventing progression to diagnosis).

**Expected impact:** Early-stage diabetes and pre-diabetes caught earlier →
lower HbA1c at diagnosis → better long-term outcomes and lower treatment cost.

### 6.2 Lifestyle Intervention Targeting

Logistic Regression coefficients reveal which modifiable lifestyle features
have the strongest negative association with diagnosis:
- `physical_activity_minutes_per_week` (negative coefficient)
- `diet_score` (negative coefficient)
- `sleep_hours_per_day` (negative coefficient)

Healthcare providers can use these to design personalised lifestyle
intervention programmes — specifically targeting patients who are
physically inactive with poor diet scores and in the 0.4–0.7 predicted
probability range (moderate risk, most responsive to intervention).

### 6.3 Threshold Optimisation by Setting

- **Mass screening (public health):** lower the classification threshold to
  0.3–0.4 to maximise recall — catching as many true cases as possible at
  the cost of more follow-up tests.
- **High-resource specialist clinic:** raise threshold to 0.65–0.7 to ensure
  referrals are high-confidence diagnoses — conserving specialist appointment
  slots.

### 6.4 Demographic Fairness Audit

Before deployment, compute per-demographic-group performance metrics
(precision, recall, and F1 for each ethnicity, gender, and income level).
If any subgroup has substantially lower recall, the model is failing to
catch diagnoses in that population — potentially worsening health disparities.
Remediation options include subgroup-specific thresholds or resampling.

### 6.5 Monitoring and Retraining

Model performance will degrade as population health patterns shift (e.g.
post-pandemic lifestyle changes). Implement monthly monitoring of:
- Model AUC on a rolling holdout window
- Prediction score distribution drift (KS statistic)
- Positive rate in newly screened cohorts

Retrain the model quarterly using the most recent 24 months of data.

---

## 7. Ethical and Social Considerations

### Sensitive demographic features

The model uses `ethnicity`, `income_level`, and `employment_status` as
features. These are legitimate epidemiological risk factors for diabetes,
but including them in a deployed model requires careful governance:
- Decisions partially based on ethnicity or income raise discrimination concerns
- GDPR Article 9 classifies ethnicity as "special category" data — processing
  requires explicit legal basis in healthcare settings
- Alternative: train ethnicity-free models and evaluate whether AUC degrades
  significantly; if not, remove the feature

### Synthetic data limitations

The dataset is entirely synthetic. Key ethical risks:
- The model may learn patterns that are artefacts of the data generation
  process rather than true clinical relationships
- No patient consent was required (synthetic data) — but real deployment
  requires consent frameworks for model training and inference

### Model opacity

Random Forest is a black-box relative to Logistic Regression. For clinical
deployment, SHAP (SHapley Additive exPlanations) values should be computed
to provide patient-level explanations ("your risk score is high primarily
because of your HbA1c of 7.8 and fasting glucose of 126 mg/dL").

### Human oversight

The model output is a risk probability — not a diagnosis. Clinical guidelines
and human clinician review must remain in the decision loop. The model should
never autonomously trigger a diabetes diagnosis in a patient record.
