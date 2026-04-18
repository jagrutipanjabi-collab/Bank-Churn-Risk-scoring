# =============================================================
# STEP 2: MODEL TRAINING & EVALUATION
# European Bank Customer Churn Project
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os, pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

os.makedirs("charts", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("   EUROPEAN BANK CHURN - MODEL TRAINING & EVALUATION")
print("=" * 60)

# ─────────────────────────────────────────
# 1. LOAD PROCESSED DATA
# ─────────────────────────────────────────
df = pd.read_csv("processed_data.csv")
print(f"\n✅ Processed data loaded: {df.shape}")

X = df.drop(columns=['Exited'])
y = df['Exited']

# ─────────────────────────────────────────
# 2. TRAIN-TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📊 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 3. HANDLE CLASS IMBALANCE WITH SMOTE
# ─────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE - Train size: {X_train_sm.shape[0]}")

# ─────────────────────────────────────────
# 4. FEATURE SCALING
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names
feature_names = X.columns.tolist()
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("✅ Scaler saved")

# ─────────────────────────────────────────
# 5. TRAIN ALL MODELS
# ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost":             xgb.XGBClassifier(n_estimators=100, random_state=42,
                                              eval_metric='logloss', verbosity=0)
}

results = {}
print("\n🤖 Training Models...")

for name, model in models.items():
    model.fit(X_train_sc, y_train_sm)
    y_pred     = model.predict(X_test_sc)
    y_prob     = model.predict_proba(X_test_sc)[:, 1]

    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred) * 100, 2),
        "F1-Score":  round(f1_score(y_test, y_pred) * 100, 2),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob) * 100, 2),
    }
    print(f"   ✅ {name} trained")

# ─────────────────────────────────────────
# 6. RESULTS TABLE
# ─────────────────────────────────────────
results_df = pd.DataFrame(results).T
print("\n📊 MODEL PERFORMANCE COMPARISON:")
print(results_df.to_string())

results_df.to_csv("models/model_results.csv")

# ─────────────────────────────────────────
# 7. BEST MODEL = XGBOOST
# ─────────────────────────────────────────
best_model = models["XGBoost"]
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("\n✅ Best model (XGBoost) saved!")

# ─────────────────────────────────────────
# 8. CONFUSION MATRIX
# ─────────────────────────────────────────
y_pred_best = best_model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'])
plt.title('XGBoost - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("charts/06_confusion_matrix.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/06_confusion_matrix.png")

# ─────────────────────────────────────────
# 9. ROC CURVE - ALL MODELS
# ─────────────────────────────────────────
plt.figure(figsize=(8, 6))
colors_roc = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for (name, model), color in zip(models.items(), colors_roc):
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color=color, label=f'{name} (AUC={auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("charts/07_roc_curves.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/07_roc_curves.png")

# ─────────────────────────────────────────
# 10. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.15
colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

fig, ax = plt.subplots(figsize=(14, 6))
for i, (model_name, color) in enumerate(zip(results_df.index, colors_bar)):
    vals = [results_df.loc[model_name, m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=model_name, color=color)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score (%)')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig("charts/08_model_comparison.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/08_model_comparison.png")

# ─────────────────────────────────────────
# 11. FEATURE IMPORTANCE
# ─────────────────────────────────────────
feat_imp = pd.Series(best_model.feature_importances_, index=feature_names)
feat_imp = feat_imp.sort_values(ascending=True).tail(15)

plt.figure(figsize=(9, 6))
colors_feat = ['#e74c3c' if v > feat_imp.mean() else '#3498db' for v in feat_imp.values]
feat_imp.plot(kind='barh', color=colors_feat)
plt.title('XGBoost - Top 15 Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig("charts/09_feature_importance.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/09_feature_importance.png")

# ─────────────────────────────────────────
# 12. SHAP VALUES
# ─────────────────────────────────────────
print("\n⚙️  Calculating SHAP values (this takes a moment)...")
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_sc)

plt.figure()
shap.summary_plot(shap_values, X_test_sc,
                  feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("charts/10_shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart saved: charts/10_shap_summary.png")

# ─────────────────────────────────────────
# 13. GENERATE CHURN PROBABILITY SCORES
# ─────────────────────────────────────────
X_test_copy = X_test.copy().reset_index(drop=True)
y_test_copy = y_test.reset_index(drop=True)

X_test_copy['Churn_Probability'] = best_model.predict_proba(X_test_sc)[:, 1]
X_test_copy['Churn_Predicted']   = best_model.predict(X_test_sc)
X_test_copy['Actual_Churn']      = y_test_copy

# Risk Scoring
def risk_label(prob):
    if prob >= 0.7:   return "🔴 HIGH RISK"
    elif prob >= 0.4: return "🟡 MEDIUM RISK"
    else:             return "🟢 LOW RISK"

X_test_copy['Risk_Score'] = X_test_copy['Churn_Probability'].apply(risk_label)
X_test_copy.to_csv("models/churn_scores.csv", index=False)

print("\n📊 Risk Score Distribution:")
print(X_test_copy['Risk_Score'].value_counts())

print("\n" + "=" * 60)
print("   ✅ MODEL TRAINING COMPLETE!")
print(f"\n   🏆 Best Model: XGBoost")
print(f"   📈 ROC-AUC: {results['XGBoost']['ROC-AUC']}%")
print(f"   🎯 Accuracy: {results['XGBoost']['Accuracy']}%")
print("\n   ▶️  Now run: streamlit run 3_streamlit_app.py")
print("=" * 60)
