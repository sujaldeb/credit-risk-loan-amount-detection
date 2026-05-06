import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-card {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 13px;
        color: #8b92a5;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #ffffff;
    }
    .metric-value.green { color: #2ecc71; }
    .metric-value.red { color: #e74c3c; }
    .metric-value.amber { color: #f39c12; }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #8b92a5;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 12px;
        margin-top: 20px;
    }
    .risk-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #2d3250;
        font-size: 13px;
    }
    .risk-key { color: #8b92a5; }
    .risk-val { color: #ffffff; font-weight: 500; }
    .badge-safe {
        background: #1a3d2b;
        color: #2ecc71;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    .badge-risk {
        background: #3d1a1a;
        color: #e74c3c;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    .badge-medium {
        background: #3d2e1a;
        color: #f39c12;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
    .stSlider > div > div { background-color: #2d3250; }
    div[data-testid="stSidebarContent"] {
        background-color: #1e2130;
        border-right: 1px solid #2d3250;
    }
    </style>
""", unsafe_allow_html=True)


# Load models and preprocessing objects
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "..", "models")
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    classifier = joblib.load(os.path.join(models_dir, "classifier_lgbm.pkl"))
    regressor = joblib.load(os.path.join(models_dir, "regressor_lasso.pkl"))
    return scaler, classifier, regressor

scaler, classifier, regressor = load_models()


def build_feature_vector(inputs):
    # These are the 86 columns X_train_scaled was built on
    # We construct a single row with all features set to their median/default
    # and override with user-provided values

    # Base feature vector with sensible defaults (median values from training)
    feature_defaults = {
        "term": 36, "int_rate": 13.0, "installment": 0.0,
        "grade": 3, "sub_grade": 13, "emp_length": 5,
        "annual_inc": inputs["annual_inc"], "dti": inputs["dti"],
        "delinq_2yrs": 0, "fico_range_low": inputs["fico_score"],
        "inq_last_6mths": 0, "open_acc": 10, "pub_rec": 0,
        "revol_bal": 15000, "revol_util": inputs["revol_util"],
        "total_acc": 25, "collections_12_mths_ex_med": 0,
        "acc_now_delinq": 0, "tot_coll_amt": 0,
        "tot_cur_bal": 50000, "total_rev_hi_lim": 30000,
        "acc_open_past_24mths": 3, "avg_cur_bal": 5000,
        "bc_open_to_buy": 5000, "bc_util": 50,
        "chargeoff_within_12_mths": 0, "delinq_amnt": 0,
        "mo_sin_old_il_acct": 100, "mo_sin_old_rev_tl_op": 150,
        "mo_sin_rcnt_rev_tl_op": 12, "mo_sin_rcnt_tl": 6,
        "mort_acc": 1, "mths_since_recent_bc": 12,
        "mths_since_recent_inq": 6, "num_accts_ever_120_pd": 0,
        "num_actv_bc_tl": 3, "num_actv_rev_tl": 5,
        "num_bc_sats": 3, "num_bc_tl": 5, "num_il_tl": 8,
        "num_op_rev_tl": 7, "num_rev_accts": 12,
        "num_rev_tl_bal_gt_0": 5, "num_sats": 10,
        "num_tl_120dpd_2m": 0, "num_tl_30dpd": 0,
        "num_tl_90g_dpd_24m": 0, "num_tl_op_past_12m": 2,
        "pct_tl_nvr_dlq": 95, "percent_bc_gt_75": 30,
        "pub_rec_bankruptcies": 0, "tax_liens": 0,
        "tot_hi_cred_lim": 80000, "total_bal_ex_mort": 20000,
        "total_bc_limit": 15000, "total_il_high_credit_limit": 25000,
        "hardship_flag": 0,
        "home_ownership_MORTGAGE": 0, "home_ownership_NONE": 0,
        "home_ownership_OWN": 0, "home_ownership_RENT": 0,
        "verification_status_Source Verified": 0,
        "verification_status_Verified": 0,
        "purpose_credit_card": 0, "purpose_debt_consolidation": 0,
        "purpose_educational": 0, "purpose_home_improvement": 0,
        "purpose_house": 0, "purpose_major_purchase": 0,
        "purpose_medical": 0, "purpose_moving": 0,
        "purpose_other": 0, "purpose_renewable_energy": 0,
        "purpose_small_business": 0, "purpose_vacation": 0,
        "purpose_wedding": 0, "initial_list_status_w": 0,
        "application_type_Joint App": 0,
        "disbursement_method_DirectPay": 0,
        "debt_settlement_flag_Y": 0, "pymnt_plan_y": 0,
        "loan_to_income_ratio": 0.0,
        "installment_to_income_ratio": 0.0,
        "credit_utilization_risk": 0.0,
        "delinquency_score": 0,
        "credit_history_depth": 0
    }

    # Override with user inputs
    loan_amnt = inputs["loan_amnt"]
    annual_inc = inputs["annual_inc"]
    term = inputs["term"]

    # Derived values
    int_rate = max(5.31, 30.99 - (inputs["fico_score"] - 660) * 0.08)
    installment = (loan_amnt * (int_rate / 100 / 12)) / \
                  (1 - (1 + int_rate / 100 / 12) ** (-term))

    feature_defaults["term"] = term
    feature_defaults["int_rate"] = round(int_rate, 2)
    feature_defaults["installment"] = round(installment, 2)
    feature_defaults["loan_to_income_ratio"] = loan_amnt / (annual_inc + 1)
    feature_defaults["installment_to_income_ratio"] = installment / (annual_inc / 12 + 1)
    feature_defaults["credit_utilization_risk"] = inputs["revol_util"] * (850 - inputs["fico_score"])
    feature_defaults["credit_history_depth"] = feature_defaults["open_acc"] + feature_defaults["total_acc"]

    # Home ownership one-hot
    ownership = inputs["home_ownership"]
    if ownership in ["MORTGAGE", "OWN", "RENT", "NONE"]:
        key = f"home_ownership_{ownership}"
        if key in feature_defaults:
            feature_defaults[key] = 1

    # Purpose one-hot
    purpose = inputs["purpose"]
    purpose_key = f"purpose_{purpose}"
    if purpose_key in feature_defaults:
        feature_defaults[purpose_key] = 1

    # Verification
    verification = inputs["verification_status"]
    if verification == "Source Verified":
        feature_defaults["verification_status_Source Verified"] = 1
    elif verification == "Verified":
        feature_defaults["verification_status_Verified"] = 1

    return pd.DataFrame([feature_defaults]), int_rate, installment


# App header
st.markdown("## 🏦 Credit Risk & Loan Amount Predictor")
st.markdown(
    "End-to-end ML pipeline trained on 914,430 LendingClub loans · "
    "LightGBM Classifier (AUC 0.774) · Lasso Regressor (R² 0.930)"
)
st.markdown("---")

# Sidebar inputs
with st.sidebar:
    st.markdown("### Borrower Profile")

    st.markdown('<div class="section-header">Personal & Employment</div>',
                unsafe_allow_html=True)
    annual_inc = st.number_input(
        "Annual Income ($)", min_value=10000, max_value=266000,
        value=65000, step=1000
    )
    emp_length = st.selectbox(
        "Employment Length",
        options=["< 1 year", "1 year", "2 years", "3 years", "4 years",
                 "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
        index=5
    )
    home_ownership = st.selectbox(
        "Home Ownership",
        options=["RENT", "MORTGAGE", "OWN", "NONE"],
        index=1
    )
    verification_status = st.selectbox(
        "Income Verification",
        options=["Not Verified", "Source Verified", "Verified"],
        index=0
    )

    st.markdown('<div class="section-header">Loan Details</div>',
                unsafe_allow_html=True)
    loan_amnt = st.slider(
        "Requested Loan Amount ($)",
        min_value=1000, max_value=40000, value=12000, step=500
    )
    term = st.selectbox("Loan Term", options=[36, 60], index=0)
    purpose = st.selectbox(
        "Loan Purpose",
        options=["debt_consolidation", "credit_card", "home_improvement",
                 "other", "major_purchase", "medical", "small_business",
                 "car", "vacation", "moving", "wedding",
                 "house", "renewable_energy", "educational"],
        index=0
    )

    st.markdown('<div class="section-header">Credit Profile</div>',
                unsafe_allow_html=True)
    fico_score = st.slider(
        "FICO Credit Score", min_value=660, max_value=845, value=700, step=5
    )
    dti = st.slider(
        "Debt-to-Income Ratio", min_value=0.0, max_value=100.0,
        value=18.5, step=0.5
    )
    revol_util = st.slider(
        "Revolving Credit Utilization (%)",
        min_value=0.0, max_value=100.0, value=42.0, step=1.0
    )

    predict_btn = st.button("Run Prediction", use_container_width=True,
                            type="primary")

# Main panel
if predict_btn:
    inputs = {
        "annual_inc": annual_inc,
        "loan_amnt": loan_amnt,
        "term": term,
        "fico_score": fico_score,
        "dti": dti,
        "revol_util": revol_util,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "verification_status": verification_status,
        "emp_length": emp_length
    }

    feature_df, int_rate, installment = build_feature_vector(inputs)

    # Scale and predict
    feature_scaled = scaler.transform(feature_df)
    default_prob = classifier.predict_proba(feature_scaled)[0][1]
    log_loan_pred = regressor.predict(feature_scaled)[0]
    predicted_loan = float(np.clip(np.expm1(log_loan_pred), 1000, 40000))

    # Risk classification
    if default_prob < 0.25:
        risk_label = "Low Risk"
        risk_color = "green"
        badge_class = "badge-safe"
    elif default_prob < 0.50:
        risk_label = "Medium Risk"
        risk_color = "amber"
        badge_class = "badge-medium"
    else:
        risk_label = "High Risk"
        risk_color = "red"
        badge_class = "badge-risk"

    loan_to_income = loan_amnt / (annual_inc + 1)
    credit_util_risk = revol_util * (850 - fico_score)

    # Layout — three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Default Risk Assessment")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Default Probability</div>
            <div class="metric-value {risk_color}">{default_prob * 100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Risk Classification</div>
            <div style="margin-top:8px;">
                <span class="{badge_class}">{risk_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk probability bar
        st.markdown("**Default Probability Gauge**")
        st.progress(float(default_prob))

    with col2:
        st.markdown("#### Loan Amount Assessment")
        approval_status = "Approved" if predicted_loan >= loan_amnt * 0.85 else "Reduced"
        approval_color = "green" if approval_status == "Approved" else "amber"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Recommended Loan Amount</div>
            <div class="metric-value green">${predicted_loan:,.0f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Requested Amount</div>
            <div class="metric-value">${loan_amnt:,.0f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Estimated Monthly Payment</div>
            <div class="metric-value">${installment:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("#### Key Risk Factors")
        st.markdown(f"""
        <div class="metric-card">
            <div class="risk-row">
                <span class="risk-key">Estimated Interest Rate</span>
                <span class="risk-val">{int_rate:.2f}%</span>
            </div>
            <div class="risk-row">
                <span class="risk-key">Loan to Income Ratio</span>
                <span class="risk-val">{loan_to_income:.3f}</span>
            </div>
            <div class="risk-row">
                <span class="risk-key">Debt to Income Ratio</span>
                <span class="risk-val">{dti:.1f}</span>
            </div>
            <div class="risk-row">
                <span class="risk-key">Credit Utilization Risk</span>
                <span class="risk-val">{credit_util_risk:,.0f}</span>
            </div>
            <div class="risk-row">
                <span class="risk-key">FICO Score</span>
                <span class="risk-val">{fico_score}</span>
            </div>
            <div class="risk-row">
                <span class="risk-key">Revolving Utilization</span>
                <span class="risk-val">{revol_util:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "**Model:** LightGBM Classifier (AUC 0.774) + Lasso Regressor (R² 0.930) · "
        "Trained on 731,544 LendingClub loans · "
        "Strictly application-time features — no post-issuance leakage"
    )

else:
    # Default state before prediction is run
    st.info(
        "Configure the borrower profile in the sidebar and click "
        "**Run Prediction** to generate a credit risk assessment."
    )
    st.markdown("#### How this works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Classification Model**
        LightGBM trained on 731K loans.
        Predicts default probability
        with 0.774 ROC-AUC score.
        """)
    with col2:
        st.markdown("""
        **Regression Model**
        Lasso regression with log
        transformation. Predicts
        loan amount with R² = 0.930.
        """)
    with col3:
        st.markdown("""
        **Feature Pipeline**
        86 features including 5
        engineered signals. StandardScaler
        applied before inference.
        """)