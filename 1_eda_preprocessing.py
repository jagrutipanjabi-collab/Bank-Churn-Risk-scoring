# =============================================================
# STEP 1: EDA & PREPROCESSING
# European Bank Customer Churn Project
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os

# Create output folder for charts
os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("   EUROPEAN BANK CUSTOMER CHURN - EDA & PREPROCESSING")
print("=" * 60)

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("European_Bank.csv")
print(f"\n✅ Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────
# 2. BASIC INFO
# ─────────────────────────────────────────
print("\n📋 Column Info:")
print(df.dtypes)

print("\n📊 Basic Statistics:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())

# ─────────────────────────────────────────
# 3. TARGET DISTRIBUTION
# ─────────────────────────────────────────
churn_counts = df['Exited'].value_counts()
churn_pct = df['Exited'].value_counts(normalize=True) * 100
print(f"\n🎯 Churn Distribution:")
print(f"   Retained (0): {churn_counts[0]} customers ({churn_pct[0]:.1f}%)")
print(f"   Churned  (1): {churn_counts[1]} customers ({churn_pct[1]:.1f}%)")

# Plot 1: Churn Distribution
plt.figure(figsize=(6, 4))
colors = ['#2ecc71', '#e74c3c']
bars = plt.bar(['Retained', 'Churned'], churn_counts.values, color=colors, edgecolor='black')
for bar, count in zip(bars, churn_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(count), ha='center', fontweight='bold')
plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig("charts/01_churn_distribution.png", dpi=150)
plt.close()
print("\n✅ Chart saved: charts/01_churn_distribution.png")

# ─────────────────────────────────────────
# 4. DROP USELESS COLUMNS
# ─────────────────────────────────────────
df.drop(columns=['Year', 'CustomerId', 'Surname'], inplace=True)
print("\n🗑️  Dropped: Year, CustomerId, Surname")

# ─────────────────────────────────────────
# 5. CHURN BY GEOGRAPHY
# ─────────────────────────────────────────
geo_churn = df.groupby('Geography')['Exited'].mean() * 100
print(f"\n🌍 Churn Rate by Geography:")
print(geo_churn.round(2))

plt.figure(figsize=(7, 4))
colors_geo = ['#3498db', '#e74c3c', '#f39c12']
bars = plt.bar(geo_churn.index, geo_churn.values, color=colors_geo, edgecolor='black')
for bar, val in zip(bars, geo_churn.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontweight='bold')
plt.title('Churn Rate by Geography', fontsize=14, fontweight='bold')
plt.ylabel('Churn Rate (%)')
plt.tight_layout()
plt.savefig("charts/02_churn_by_geography.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/02_churn_by_geography.png")

# ─────────────────────────────────────────
# 6. CHURN BY GENDER
# ─────────────────────────────────────────
gender_churn = df.groupby('Gender')['Exited'].mean() * 100
print(f"\n👥 Churn Rate by Gender:")
print(gender_churn.round(2))

plt.figure(figsize=(5, 4))
colors_gen = ['#9b59b6', '#1abc9c']
bars = plt.bar(gender_churn.index, gender_churn.values, color=colors_gen, edgecolor='black')
for bar, val in zip(bars, gender_churn.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontweight='bold')
plt.title('Churn Rate by Gender', fontsize=14, fontweight='bold')
plt.ylabel('Churn Rate (%)')
plt.tight_layout()
plt.savefig("charts/03_churn_by_gender.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/03_churn_by_gender.png")

# ─────────────────────────────────────────
# 7. AGE DISTRIBUTION BY CHURN
# ─────────────────────────────────────────
plt.figure(figsize=(8, 4))
df[df['Exited'] == 0]['Age'].hist(bins=30, alpha=0.6, color='#2ecc71', label='Retained')
df[df['Exited'] == 1]['Age'].hist(bins=30, alpha=0.6, color='#e74c3c', label='Churned')
plt.title('Age Distribution: Churned vs Retained', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig("charts/04_age_distribution.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/04_age_distribution.png")

# ─────────────────────────────────────────
# 8. CORRELATION HEATMAP
# ─────────────────────────────────────────
df_encoded = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=False)
numeric_df = df_encoded.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("charts/05_correlation_heatmap.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/05_correlation_heatmap.png")

# ─────────────────────────────────────────
# 9. FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n⚙️  Feature Engineering...")

df['Balance_Salary_Ratio']    = df['Balance'] / (df['EstimatedSalary'] + 1)
df['Age_Tenure_Interaction']  = df['Age'] * df['Tenure']
df['Product_Engagement']      = df['NumOfProducts'] * df['IsActiveMember']
df['Zero_Balance']            = (df['Balance'] == 0).astype(int)

print("✅ New features created:")
print("   - Balance_Salary_Ratio")
print("   - Age_Tenure_Interaction")
print("   - Product_Engagement")
print("   - Zero_Balance")

# ─────────────────────────────────────────
# 10. ENCODE CATEGORICALS
# ─────────────────────────────────────────
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=False)
print(f"\n✅ After encoding, shape: {df.shape}")

# ─────────────────────────────────────────
# 11. SAVE PROCESSED DATA
# ─────────────────────────────────────────
df.to_csv("processed_data.csv", index=False)
print("\n💾 Processed data saved: processed_data.csv")

print("\n" + "=" * 60)
print("   ✅ EDA & PREPROCESSING COMPLETE!")
print("   ▶️  Now run: python 2_model_training.py")
print("=" * 60)
