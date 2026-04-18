# =============================================================
# STEP 3: STREAMLIT WEB APP
# European Bank Customer Churn - Risk Intelligence Dashboard
# =============================================================
# Run with: streamlit run 3_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1923; }
    .stApp { background-color: #0f1923; color: white; }
    .metric-card {
        background: linear-gradient(135deg, #1a2a3a, #1e3a5f);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2e4a6a;
    }
    .risk-high   { background: linear-gradient(135deg, #7b1c1c, #c0392b); border-radius:12px; padding:20px; text-align:center; }
    .risk-medium { background: linear-gradient(135deg, #7b5e1c, #e67e22); border-radius:12px; padding:20px; text-align:center; }
    .risk-low    { background: linear-gradient(135deg, #1c5e2e, #27ae60); border-radius:12px; padding:20px; text-align:center; }
    h1, h2, h3 { color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

@st.cache_data
def load_data():
    df  = pd.read_csv("processed_data.csv")
    raw = pd.read_csv("European_Bank.csv")
    scores = pd.read_csv("models/churn_scores.csv")
    results = pd.read_csv("models/model_results.csv", index_col=0)
    return df, raw, scores, results

model, scaler, feature_names = load_artifacts()
df, raw_df, scores_df, results_df = load_data()

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bank.png", width=80)
st.sidebar.title("🏦 Churn Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate To:", [
    "🏠 Overview",
    "🧮 Churn Risk Calculator",
    "📊 Feature Importance",
    "🔬 What-If Simulator",
    "🏆 Model Performance"
])

# ─────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🏦 European Bank — Customer Churn Intelligence")
    st.markdown("#### Predictive Risk Scoring Dashboard")
    st.markdown("---")

    # KPI Cards
    total     = len(raw_df)
    churned   = raw_df['Exited'].sum()
    retained  = total - churned
    churn_pct = round(churned / total * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Customers", f"{total:,}")
    with c2:
        st.metric("Churned", f"{churned:,}", delta=f"{churn_pct}%", delta_color="inverse")
    with c3:
        st.metric("Retained", f"{retained:,}")
    with c4:
        st.metric("Model Accuracy", "86.5%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Churn by Geography")
        geo = raw_df.groupby('Geography')['Exited'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0f1923')
        ax.set_facecolor('#0f1923')
        colors = ['#3498db', '#e74c3c', '#f39c12']
        bars = ax.bar(geo.index, geo.values, color=colors, edgecolor='white')
        for bar, val in zip(bars, geo.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', color='white', fontweight='bold')
        ax.set_ylabel('Churn Rate (%)', color='white')
        ax.tick_params(colors='white')
        ax.spines[:].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("👥 Churn by Gender")
        gen = raw_df.groupby('Gender')['Exited'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0f1923')
        ax.set_facecolor('#0f1923')
        colors = ['#9b59b6', '#1abc9c']
        bars = ax.bar(gen.index, gen.values, color=colors, edgecolor='white')
        for bar, val in zip(bars, gen.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', color='white', fontweight='bold')
        ax.set_ylabel('Churn Rate (%)', color='white')
        ax.tick_params(colors='white')
        ax.spines[:].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("📋 Risk Score Distribution (Test Set)")
    risk_dist = scores_df['Risk_Score'].value_counts()
    c1, c2, c3 = st.columns(3)
    for col, (label, count) in zip([c1, c2, c3], risk_dist.items()):
        col.metric(label, f"{count} customers")

# ─────────────────────────────────────────
# PAGE 2: CHURN RISK CALCULATOR
# ─────────────────────────────────────────
elif page == "🧮 Churn Risk Calculator":
    st.title("🧮 Customer Churn Risk Calculator")
    st.markdown("Enter customer details to get instant churn probability!")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal Info")
        credit_score  = st.slider("Credit Score", 300, 850, 650)
        age           = st.slider("Age", 18, 80, 35)
        gender        = st.selectbox("Gender", ["Male", "Female"])
        geography     = st.selectbox("Geography", ["France", "Germany", "Spain"])

    with col2:
        st.subheader("🏦 Account Details")
        tenure        = st.slider("Tenure (Years)", 0, 10, 3)
        balance       = st.number_input("Account Balance (€)", 0, 300000, 50000, step=1000)
        num_products  = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card   = st.selectbox("Has Credit Card?", ["Yes", "No"])

    with col3:
        st.subheader("💼 Engagement")
        is_active     = st.selectbox("Is Active Member?", ["Yes", "No"])
        salary        = st.number_input("Estimated Salary (€)", 0, 200000, 80000, step=1000)

    st.markdown("---")

    if st.button("🔮 Calculate Churn Risk", use_container_width=True):
        # Build input
        input_data = {
            'CreditScore':    credit_score,
            'Age':            age,
            'Tenure':         tenure,
            'Balance':        balance,
            'NumOfProducts':  num_products,
            'HasCrCard':      1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': salary,
            'Balance_Salary_Ratio':   balance / (salary + 1),
            'Age_Tenure_Interaction': age * tenure,
            'Product_Engagement':     num_products * (1 if is_active == "Yes" else 0),
            'Zero_Balance':           1 if balance == 0 else 0,
            'Geography_France':  1 if geography == "France"  else 0,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain':   1 if geography == "Spain"   else 0,
            'Gender_Female':     1 if gender == "Female" else 0,
            'Gender_Male':       1 if gender == "Male"   else 0,
        }

        input_df  = pd.DataFrame([input_data])[feature_names]
        input_sc  = scaler.transform(input_df)
        prob      = model.predict_proba(input_sc)[0][1]
        prob_pct  = round(prob * 100, 1)

        # Risk Level
        if prob >= 0.7:
            risk_class = "🔴 HIGH RISK"
            risk_color = "#e74c3c"
            msg = "⚠️ This customer is very likely to churn. Immediate retention action needed!"
        elif prob >= 0.4:
            risk_class = "🟡 MEDIUM RISK"
            risk_color = "#f39c12"
            msg = "⚡ Moderate churn risk. Consider proactive engagement."
        else:
            risk_class = "🟢 LOW RISK"
            risk_color = "#27ae60"
            msg = "✅ Low churn risk. Customer appears satisfied."

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Churn Probability", f"{prob_pct}%")
        with c2:
            st.metric("Risk Level", risk_class)
        with c3:
            st.metric("Retention Probability", f"{100 - prob_pct}%")

        st.info(msg)

        # Gauge Chart
        fig, ax = plt.subplots(figsize=(6, 3), facecolor='#0f1923')
        ax.set_facecolor('#0f1923')
        ax.barh(["Churn Risk"], [prob_pct], color=risk_color, height=0.4)
        ax.barh(["Churn Risk"], [100 - prob_pct], left=[prob_pct],
                color='#2c3e50', height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability (%)', color='white')
        ax.tick_params(colors='white')
        ax.spines[:].set_visible(False)
        ax.text(prob_pct / 2, 0, f'{prob_pct}%', ha='center',
                va='center', color='white', fontweight='bold', fontsize=14)
        ax.set_title('Churn Probability Gauge', color='white', fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────
# PAGE 3: FEATURE IMPORTANCE
# ─────────────────────────────────────────
elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance Dashboard")
    st.markdown("What drives customer churn the most?")
    st.markdown("---")

    feat_imp = pd.Series(model.feature_importances_, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0f1923')
    ax.set_facecolor('#0f1923')
    colors = ['#e74c3c' if v > feat_imp.mean() else '#3498db' for v in feat_imp.values]
    feat_imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('XGBoost — Top Feature Importances', color='white',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', color='white')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("🔑 Key Insights")
    st.info("🔴 **Red bars** = Above-average importance (top churn drivers)")
    st.info("🔵 **Blue bars** = Below-average importance (secondary factors)")

# ─────────────────────────────────────────
# PAGE 4: WHAT-IF SIMULATOR
# ─────────────────────────────────────────
elif page == "🔬 What-If Simulator":
    st.title("🔬 What-If Scenario Simulator")
    st.markdown("See how changing factors affects churn probability!")
    st.markdown("---")

    st.subheader("📌 Base Customer Profile")
    col1, col2 = st.columns(2)

    with col1:
        age      = st.slider("Age", 18, 80, 45)
        balance  = st.slider("Balance (€)", 0, 250000, 120000, step=5000)
        products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)

    with col2:
        active   = st.selectbox("Is Active Member?", ["No", "Yes"])
        geo      = st.selectbox("Geography", ["Germany", "France", "Spain"])
        salary   = st.slider("Salary (€)", 20000, 200000, 80000, step=5000)

    def predict_prob(age, balance, products, active, geo, salary,
                     credit=600, tenure=3, has_cc=1):
        inp = {
            'CreditScore': credit, 'Age': age, 'Tenure': tenure,
            'Balance': balance, 'NumOfProducts': products,
            'HasCrCard': has_cc, 'IsActiveMember': 1 if active == "Yes" else 0,
            'EstimatedSalary': salary,
            'Balance_Salary_Ratio':   balance / (salary + 1),
            'Age_Tenure_Interaction': age * tenure,
            'Product_Engagement':     products * (1 if active == "Yes" else 0),
            'Zero_Balance':           1 if balance == 0 else 0,
            'Geography_France':  1 if geo == "France"  else 0,
            'Geography_Germany': 1 if geo == "Germany" else 0,
            'Geography_Spain':   1 if geo == "Spain"   else 0,
            'Gender_Female': 0, 'Gender_Male': 1,
        }
        df_inp = pd.DataFrame([inp])[feature_names]
        sc_inp = scaler.transform(df_inp)
        return round(model.predict_proba(sc_inp)[0][1] * 100, 1)

    base_prob = predict_prob(age, balance, products, active, geo, salary)

    st.markdown("---")
    st.subheader("🔄 What If We Change These?")

    scenarios = {
        "Base Case":              predict_prob(age, balance, products, active, geo, salary),
        "Add 1 More Product":     predict_prob(age, balance, min(products + 1, 4), active, geo, salary),
        "Make Active Member":     predict_prob(age, balance, products, "Yes", geo, salary),
        "Increase Balance €50K":  predict_prob(age, balance + 50000, products, active, geo, salary),
        "Move to France":         predict_prob(age, balance, products, active, "France", salary),
    }

    scenario_df = pd.DataFrame.from_dict(
        scenarios, orient='index', columns=['Churn Probability (%)'])
    scenario_df['Change vs Base'] = scenario_df['Churn Probability (%)'] - base_prob
    scenario_df['Change vs Base'] = scenario_df['Change vs Base'].apply(
        lambda x: f"{'▼' if x < 0 else '▲'} {abs(x):.1f}%" if x != 0 else "—")

    st.dataframe(scenario_df, use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#0f1923')
    ax.set_facecolor('#0f1923')
    vals   = list(scenarios.values())
    labels = list(scenarios.keys())
    colors = ['#e74c3c' if v > base_prob else '#27ae60' if v < base_prob else '#3498db'
              for v in vals]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', color='white', fontweight='bold')
    ax.set_ylabel('Churn Probability (%)', color='white')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    plt.xticks(rotation=20, ha='right')
    ax.set_title('Scenario Comparison', color='white', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────
# PAGE 5: MODEL PERFORMANCE
# ─────────────────────────────────────────
elif page == "🏆 Model Performance":
    st.title("🏆 Model Performance Comparison")
    st.markdown("---")

    st.dataframe(results_df.style.highlight_max(axis=0, color='#27ae60'),
                 use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Visual Comparison")

    metrics = results_df.columns.tolist()
    x = np.arange(len(metrics))
    width = 0.15
    colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0f1923')
    ax.set_facecolor('#0f1923')

    for i, (model_name, color) in enumerate(zip(results_df.index, colors_bar)):
        vals = results_df.loc[model_name].values
        ax.bar(x + i * width, vals, width, label=model_name, color=color)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics, color='white')
    ax.set_ylabel('Score (%)', color='white')
    ax.legend(facecolor='#1a2a3a', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    ax.set_title('All Models — Performance Metrics', color='white',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("🏅 Why XGBoost?")
    st.success("""
    ✅ **XGBoost** was selected as the best model because:
    - Highest ROC-AUC score → best at distinguishing churners
    - Handles imbalanced data well with SMOTE
    - Built-in feature importance for explainability
    - Robust against overfitting with regularization
    """)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**🏦 European Bank Churn**")
st.sidebar.markdown("Powered by XGBoost + SHAP")
st.sidebar.markdown("Unified Mentor Project")
