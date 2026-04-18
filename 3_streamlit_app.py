
# =============================================================
# STREAMLIT APP - Bank Customer Churn Risk Intelligence
# Self-contained: trains model on startup if not found
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

st.set_page_config(
    page_title="Bank Churn Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1923; color: white; }
    h1, h2, h3 { color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TRAIN MODEL ON STARTUP
# ─────────────────────────────────────────
@st.cache_resource
def train_and_load():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE

    df = pd.read_csv("European_Bank.csv")
    df.drop(columns=['Year', 'CustomerId', 'Surname'], inplace=True)

    # Feature Engineering
    df['Balance_Salary_Ratio']    = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['Age_Tenure_Interaction']  = df['Age'] * df['Tenure']
    df['Product_Engagement']      = df['NumOfProducts'] * df['IsActiveMember']
    df['Zero_Balance']            = (df['Balance'] == 0).astype(int)

    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=False)

    X = df.drop(columns=['Exited'])
    y = df['Exited']

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sm)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_sc, y_train_sm)

    return model, scaler, feature_names, X_test, X_test_sc, y_test

@st.cache_data
def load_raw():
    return pd.read_csv("European_Bank.csv")

# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────
with st.spinner("🔄 Loading model... Please wait!"):
    model, scaler, feature_names, X_test, X_test_sc, y_test = train_and_load()
    raw_df = load_raw()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.title("🏦 Churn Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate To:", [
    "🏠 Overview",
    "🧮 Churn Risk Calculator",
    "📊 Feature Importance",
    "🔬 What-If Simulator",
])

# ─────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🏦 European Bank — Customer Churn Intelligence")
    st.markdown("#### Predictive Risk Scoring Dashboard")
    st.markdown("---")

    total     = len(raw_df)
    churned   = raw_df['Exited'].sum()
    retained  = total - churned
    churn_pct = round(churned / total * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned", f"{churned:,}", delta=f"{churn_pct}%", delta_color="inverse")
    c3.metric("Retained", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_pct}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Churn by Geography")
        geo = raw_df.groupby('Geography')['Exited'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0f1923')
        ax.set_facecolor('#0f1923')
        bars = ax.bar(geo.index, geo.values, color=['#3498db','#e74c3c','#f39c12'], edgecolor='white')
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
        bars = ax.bar(gen.index, gen.values, color=['#9b59b6','#1abc9c'], edgecolor='white')
        for bar, val in zip(bars, gen.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', color='white', fontweight='bold')
        ax.set_ylabel('Churn Rate (%)', color='white')
        ax.tick_params(colors='white')
        ax.spines[:].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("📊 Age Distribution by Churn")
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f1923')
    ax.set_facecolor('#0f1923')
    raw_df[raw_df['Exited']==0]['Age'].hist(bins=30, alpha=0.6, color='#2ecc71', label='Retained', ax=ax)
    raw_df[raw_df['Exited']==1]['Age'].hist(bins=30, alpha=0.6, color='#e74c3c', label='Churned', ax=ax)
    ax.set_xlabel('Age', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    ax.legend(facecolor='#1a2a3a', labelcolor='white')
    st.pyplot(fig)
    plt.close()

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
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age          = st.slider("Age", 18, 80, 35)
        gender       = st.selectbox("Gender", ["Male", "Female"])
        geography    = st.selectbox("Geography", ["France", "Germany", "Spain"])
    with col2:
        st.subheader("🏦 Account Details")
        tenure       = st.slider("Tenure (Years)", 0, 10, 3)
        balance      = st.number_input("Account Balance (€)", 0, 300000, 50000, step=1000)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card  = st.selectbox("Has Credit Card?", ["Yes", "No"])
    with col3:
        st.subheader("💼 Engagement")
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        salary    = st.number_input("Estimated Salary (€)", 0, 200000, 80000, step=1000)

    st.markdown("---")
    if st.button("🔮 Calculate Churn Risk", use_container_width=True):
        input_data = {
            'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
            'Balance': balance, 'NumOfProducts': num_products,
            'HasCrCard': 1 if has_cr_card=="Yes" else 0,
            'IsActiveMember': 1 if is_active=="Yes" else 0,
            'EstimatedSalary': salary,
            'Balance_Salary_Ratio':   balance / (salary + 1),
            'Age_Tenure_Interaction': age * tenure,
            'Product_Engagement':     num_products * (1 if is_active=="Yes" else 0),
            'Zero_Balance':           1 if balance == 0 else 0,
            'Geography_France':  1 if geography=="France"  else 0,
            'Geography_Germany': 1 if geography=="Germany" else 0,
            'Geography_Spain':   1 if geography=="Spain"   else 0,
            'Gender_Female': 1 if gender=="Female" else 0,
            'Gender_Male':   1 if gender=="Male"   else 0,
        }
        input_df = pd.DataFrame([input_data])[feature_names]
        input_sc = scaler.transform(input_df)
        prob     = model.predict_proba(input_sc)[0][1]
        prob_pct = round(prob * 100, 1)

        if prob >= 0.7:
            risk = "🔴 HIGH RISK"
            msg  = "⚠️ Immediate retention action needed!"
        elif prob >= 0.4:
            risk = "🟡 MEDIUM RISK"
            msg  = "⚡ Consider proactive engagement."
        else:
            risk = "🟢 LOW RISK"
            msg  = "✅ Customer appears satisfied."

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob_pct}%")
        c2.metric("Risk Level", risk)
        c3.metric("Retention Probability", f"{100-prob_pct}%")
        st.info(msg)

# ─────────────────────────────────────────
# PAGE 3: FEATURE IMPORTANCE
# ─────────────────────────────────────────
elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance Dashboard")
    st.markdown("---")
    feat_imp = pd.Series(model.feature_importances_, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0f1923')
    ax.set_facecolor('#0f1923')
    colors = ['#e74c3c' if v > feat_imp.mean() else '#3498db' for v in feat_imp.values]
    feat_imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Top Feature Importances', color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', color='white')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    st.pyplot(fig)
    plt.close()
    st.info("🔴 Red = Above average importance | 🔵 Blue = Below average importance")

# ─────────────────────────────────────────
# PAGE 4: WHAT-IF SIMULATOR
# ─────────────────────────────────────────
elif page == "🔬 What-If Simulator":
    st.title("🔬 What-If Scenario Simulator")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age      = st.slider("Age", 18, 80, 45)
        balance  = st.slider("Balance (€)", 0, 250000, 120000, step=5000)
        products = st.selectbox("Number of Products", [1, 2, 3, 4])
    with col2:
        active = st.selectbox("Is Active Member?", ["No", "Yes"])
        geo    = st.selectbox("Geography", ["Germany", "France", "Spain"])
        salary = st.slider("Salary (€)", 20000, 200000, 80000, step=5000)

    def predict_prob(age, balance, products, active, geo, salary):
        inp = {
            'CreditScore': 600, 'Age': age, 'Tenure': 3,
            'Balance': balance, 'NumOfProducts': products,
            'HasCrCard': 1, 'IsActiveMember': 1 if active=="Yes" else 0,
            'EstimatedSalary': salary,
            'Balance_Salary_Ratio':   balance / (salary + 1),
            'Age_Tenure_Interaction': age * 3,
            'Product_Engagement':     products * (1 if active=="Yes" else 0),
            'Zero_Balance':           1 if balance==0 else 0,
            'Geography_France':  1 if geo=="France"  else 0,
            'Geography_Germany': 1 if geo=="Germany" else 0,
            'Geography_Spain':   1 if geo=="Spain"   else 0,
            'Gender_Female': 0, 'Gender_Male': 1,
        }
        df_inp = pd.DataFrame([inp])[feature_names]
        sc_inp = scaler.transform(df_inp)
        return round(model.predict_proba(sc_inp)[0][1] * 100, 1)

    base_prob = predict_prob(age, balance, products, active, geo, salary)

    scenarios = {
        "Base Case":             base_prob,
        "Add 1 More Product":    predict_prob(age, balance, min(products+1,4), active, geo, salary),
        "Make Active Member":    predict_prob(age, balance, products, "Yes", geo, salary),
        "Increase Balance €50K": predict_prob(age, balance+50000, products, active, geo, salary),
        "Move to France":        predict_prob(age, balance, products, active, "France", salary),
    }

    st.markdown("---")
    scenario_df = pd.DataFrame.from_dict(scenarios, orient='index', columns=['Churn Probability (%)'])
    st.dataframe(scenario_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#0f1923')
    ax.set_facecolor('#0f1923')
    vals   = list(scenarios.values())
    labels = list(scenarios.keys())
    colors = ['#e74c3c' if v > base_prob else '#27ae60' if v < base_prob else '#3498db' for v in vals]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', color='white', fontweight='bold')
    ax.set_ylabel('Churn Probability (%)', color='white')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.sidebar.markdown("---")
st.sidebar.markdown("**🏦 European Bank Churn**")
st.sidebar.markdown("Powered by Random Forest")
st.sidebar.markdown("Unified Mentor Project")
