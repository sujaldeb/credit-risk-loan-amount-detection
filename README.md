# Credit Risk & Loan Amount Prediction
### End-to-End Machine Learning Pipeline | LendingClub Dataset
**Python · Scikit-learn · LightGBM · XGBoost · Streamlit**

---

## One-Line Summary

> Built an end-to-end ML pipeline on 2.26M real LendingClub loan records — engineering 5 domain-driven financial features, validating all predictors with hypothesis tests, and training a LightGBM classifier that identifies defaulters with **68% recall and 0.774 ROC-AUC** alongside a Lasso regressor that predicts loan amounts with **R² = 0.930 and RMSE of $2,364** — all using strictly application-time features with zero post-issuance leakage.

---

## Live Demo

🚀 **Streamlit App** — [Try the deployed predictor](YOUR_STREAMLIT_URL_HERE)

---

## Problem Statement

LendingClub connects borrowers with investors across the United States. When a loan is issued, the platform takes on financial risk on behalf of its investors — if a borrower defaults, investors lose capital. Identifying high-risk borrowers before a loan is issued is therefore critical to platform health.

**Two core prediction problems:**
- **Classification** — Will this borrower default on their loan? (binary: yes/no)
- **Regression** — What loan amount should this borrower qualify for? (continuous: $1,000–$40,000)

These mirror the two decisions any lending institution makes daily — managing credit risk and determining loan eligibility.

**Why this is hard:** Most published benchmarks on this dataset achieve 0.90+ AUC by incorporating post-issuance payment history — data that does not exist at application time. We enforce strict application-time feature constraints, making this a genuinely production-realistic prediction system.

---

## Repository Structure

```
credit-risk-loan-amount-detection/
│
├── notebooks/
│   ├── 01_problem_statement_and_eda.ipynb
│   ├── 02_hypothesis_testing_and_preprocessing.ipynb
│   ├── 03_feature_engineering_scaling_pca.ipynb
│   ├── 04_classification_models.ipynb
│   ├── 05_regression_models.ipynb
│   └── 06_model_evaluation_and_summary.ipynb
│
├── models/
│   ├── classifier_lgbm.pkl
│   ├── classifier_xgb.pkl
│   ├── classifier_rf.pkl
│   ├── regressor_lasso.pkl
│   ├── scaler.pkl
│   └── pca.pkl
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── requirements.txt
└── README.md
```

---

## Data Description

| Source | Records | Features | Period |
|---|---|---|---|
| LendingClub Accepted Loans (Kaggle) | 2,260,701 | 151 | 2007–2018 |
| After outcome filtering (2015–2018) | 915,420 | 151 | 2015–2018 |
| After full preprocessing pipeline | 914,430 | 87 | 2015–2018 |

**Target Variables:**
- `default` — binary label: 1 = Charged Off / Late 31–120 days / Default, 0 = Fully Paid
- `loan_amnt` — continuous: loan amount in USD ($1,000–$40,000)

**Outcome filtering rationale:** Current, In Grace Period, and Late (16–30 days) loans have no confirmed final outcome and were excluded to prevent label noise. This removed 866,666 rows but ensured every training example has a verified real-world result.

---

## Methodology

### Stage 1 — Exploratory Data Analysis (`01_problem_statement_and_eda.ipynb`)
- Loaded 2.26M records directly from `.gz` compressed file — no manual extraction needed
- Filtered to 2015–2018 for data quality and relevance
- Identified 44 columns exceeding 40% missing threshold — all event-driven administrative fields
- Constructed binary default label with 23.29% default rate — moderate class imbalance
- Key finding: loan grade shows perfect monotonic default rate from 7.0% (Grade A) to 55.4% (Grade G)
- Identified `loan_amnt` and `installment` correlation of 0.95 — primary justification for PCA

### Stage 2 — Hypothesis Testing (`02_hypothesis_testing_and_preprocessing.ipynb`)

All three tests rejected H0 with p-values effectively equal to zero:

| Test | Question | Statistic | Result |
|---|---|---|---|
| Independent T-Test | Does income differ by default? | t = -39.40 | Defaulters earn $7,226 less on average |
| Chi-Square | Is grade associated with default? | χ² = 72,453 | Highly significant association |
| One-Way ANOVA | Does interest rate differ by purpose? | F = 2,650 | Significant across all 14 purposes |

**Preprocessing decisions driven by these results:**
- `annual_inc`, `grade`, `purpose` — all three statistically validated and retained
- 46 high-missing columns dropped (>40% threshold)
- 16 post-issuance leakage columns removed
- Outliers fixed: `annual_inc` capped at 99th percentile ($266K), `dti` hard-capped at [0, 100], `revol_util` capped at 100%

### Stage 3 — Feature Engineering, Scaling & PCA (`03_feature_engineering_scaling_pca.ipynb`)

**5 engineered features — all appeared in top 10 model importances:**

| Feature | Formula | Rationale |
|---|---|---|
| `loan_to_income_ratio` | loan_amnt / annual_inc | Affordability relative to income |
| `installment_to_income_ratio` | installment / (annual_inc/12) | Monthly payment burden |
| `credit_utilization_risk` | revol_util × (850 − fico_range_low) | Compound risk signal |
| `delinquency_score` | Sum of 4 delinquency indicators | Composite past behavior |
| `credit_history_depth` | total_acc + open_acc | Breadth of credit experience |

**PCA results:**
- 86 features → 53 principal components (38% dimensionality reduction)
- 95.39% variance retained
- PC1 explains 12.16% of variance — information broadly distributed across features
- Justification: `loan_amnt`/`installment` correlation = 0.95 confirmed multicollinearity requiring PCA

### Stage 4 — Classification Models (`04_classification_models.ipynb`)

**Class imbalance handling:** `class_weight="balanced"` used across all scikit-learn models. `scale_pos_weight` and `is_unbalance=True` for XGBoost and LightGBM respectively. Sample-based GridSearchCV (10% stratified sample) used for Random Forest tuning to manage compute cost — a standard industry practice on large datasets.

| Model | ROC-AUC | Default Recall | Default Precision | F1 (Default) |
|---|---|---|---|---|
| Logistic Regression | 0.7585 | 0.63 | 0.42 | 0.51 |
| Decision Tree | 0.7511 | 0.66 | 0.40 | 0.50 |
| Random Forest (tuned) | 0.7637 | 0.55 | 0.47 | 0.51 |
| XGBoost | 0.7741 | 0.67 | 0.42 | 0.52 |
| **LightGBM** | **0.7740** | **0.68** | **0.42** | **0.52** |

**Selected: LightGBM** — highest default recall (0.68) and joint highest AUC (0.774). Default recall is the primary business metric — a missed defaulter represents a direct loan loss.

**LightGBM confusion matrix on 182,886 test records:**
- True Negatives: 100,738 (safe borrowers correctly identified)
- True Positives: 28,770 (defaulters correctly caught)
- False Negatives: 13,817 (defaulters missed)
- False Positives: 39,561 (good borrowers incorrectly flagged)

At average loan size of $14,500, the model identifies approximately $417M in potential default exposure.

### Stage 5 — Regression Models (`05_regression_models.ipynb`)

Log transformation applied to `loan_amnt` before modeling — skewness reduced from 0.784 to -0.642. Predictions clipped to platform limits ($1,000–$40,000) using domain knowledge.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.8054 | $3,952 | $2,223 |
| Ridge Regression (α=1.0) | 0.8054 | $3,952 | $2,223 |
| Lasso Regression (α=0.001) | 0.8070 | $3,936 | $2,225 |
| **Lasso Final (clipped)** | **0.9304** | **$2,364** | **$1,774** |

**Selected: Lasso (α=0.001)** — 34 out of 86 features driven to exactly zero, making it the most interpretable model with the best predictive performance after domain-knowledge clipping.

**Why Linear and Ridge perform identically:** With 731,544 training rows, the data volume itself acts as a natural regularizer — coefficients are already well-constrained without explicit penalty.

---

## Key Findings

1. **Interest rate is the dominant default predictor** — used in 429 splits by LightGBM, nearly double the second-ranked feature. It encodes LendingClub's own risk assessment as a single powerful signal.

2. **Loan grade is a perfect risk signal** — default rate rises monotonically from 7.0% at Grade A to 55.4% at Grade G across all 182,886 test records.

3. **Engineered features validated by model** — `loan_to_income_ratio` ranked 5th and `installment_to_income_ratio` ranked 3rd in Lasso coefficients, directly confirming the feature engineering decisions.

4. **Gradient boosting outperforms ensemble averaging** — XGBoost and LightGBM both achieved 0.774 AUC vs Random Forest's 0.764, confirming the sequential error-correction advantage of boosting on tabular financial data.

5. **Domain knowledge improves regression more than tuning** — clipping predictions to the platform's $1,000–$40,000 range improved R² from 0.807 to 0.930, a larger gain than any hyperparameter optimization step.

6. **Application-time AUC ceiling** — all five models converge near 0.75–0.77 AUC, consistent with published benchmarks on strictly application-time feature sets. This reflects the genuine difficulty of default prediction without post-issuance behavioral data.

---

## Note on Benchmark Comparison

Published papers and Kaggle notebooks achieving 0.90+ AUC on this dataset frequently incorporate post-issuance features — payment amounts received, days since last payment, total principal recovered — that only exist after a loan has been partially or fully repaid. Including these constitutes data leakage for any real prediction system where the decision must be made before the loan is issued.

Our 0.774 AUC represents genuine out-of-sample performance on a production-realistic, leakage-free feature set. In a real lending system, this is the number that matters.

---

## Statistical Validation Summary

| Test | H0 | p-value | Conclusion |
|---|---|---|---|
| T-Test (income vs default) | No income difference | ~0 | Defaulters earn $7,226 less |
| Chi-Square (grade vs default) | No association | ~0 | Grade strongly predicts default |
| ANOVA (rate vs purpose) | No rate difference | ~0 | Purpose significantly affects rate |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistical Testing | SciPy (t-test, chi-square, ANOVA) |
| Machine Learning | Scikit-learn, LightGBM, XGBoost |
| Dimensionality Reduction | PCA (sklearn) |
| Model Serialization | Joblib |
| Deployment | Streamlit, Streamlit Community Cloud |
| Environment | Jupyter Notebook, Anaconda |
| Version Control | Git, GitHub |

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/sujaldeb/credit-risk-loan-amount-detection.git
cd credit-risk-loan-amount-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# Place accepted_2007_to_2018Q4.csv.gz in the data/ folder
# Dataset: https://www.kaggle.com/datasets/wordsforthewise/lending-club

# 4. Run notebooks in order
jupyter notebook notebooks/01_problem_statement_and_eda.ipynb
```

> **Note:** The `data/` folder is excluded from this repository via `.gitignore` due to file size (~383MB compressed). Trained model files in `models/` are included and can be used directly with the Streamlit app.

---

## Limitations & Future Work

- External data (macroeconomic indicators, credit bureau tradelines) would likely push classification AUC above 0.85
- Threshold optimization on the LightGBM classifier can shift the precision-recall tradeoff based on business risk appetite
- Time-series cross-validation (walk-forward) would better simulate production deployment conditions
- SHAP values would provide per-prediction explainability required for regulatory compliance in real lending

---

## Author

**Sujal Deb** — [GitHub](https://github.com/sujaldeb)
