# AI Tool Adoption Trend Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Load Dataset
df = pd.read_csv("../data/data.csv")

# -----------------------------
# Phase 1: Data Understanding
# -----------------------------
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -----------------------------
# Phase 2: Descriptive Analysis
# -----------------------------
print("\nDescriptive Statistics")
print(df.describe())

print("\nCompany Size Distribution")
print(df["company_size"].value_counts())

# -----------------------------
# Phase 3: Trend Investigation
# -----------------------------

# Industry vs Adoption
industry_adoption = df.groupby("industry")["ai_adoption_rate"].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
industry_adoption.plot(kind="bar")
plt.title("Industry vs AI Adoption")
plt.tight_layout()
plt.savefig("../images/industry_vs_adoption.png")
plt.close()

# Company Size vs Adoption
size_adoption = df.groupby("company_size")["ai_adoption_rate"].mean()

plt.figure(figsize=(6,4))
size_adoption.plot(kind="bar")
plt.title("Company Size vs AI Adoption")
plt.tight_layout()
plt.savefig("../images/company_size_vs_adoption.png")
plt.close()

# AI Tool vs Satisfaction
tool_satisfaction = df.groupby("ai_primary_tool")["employee_satisfaction_score"].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
tool_satisfaction.plot(kind="bar")
plt.title("AI Tool vs Satisfaction")
plt.tight_layout()
plt.savefig("../images/AI_tool_vs_satisfaction.png")
plt.close()

# -----------------------------
# Phase 4: Correlation Analysis
# -----------------------------

numeric_cols = [
    "ai_adoption_rate",
    "employee_satisfaction_score",
    "productivity_change_percent",
    "ai_investment_per_employee",
    "cost_reduction_percent"
]

corr = df[numeric_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("../images/correlation_heatmap.png")
plt.close()

# Adoption vs Productivity
plt.figure(figsize=(6,4))
plt.scatter(
    df["ai_adoption_rate"],
    df["productivity_change_percent"],
    alpha=0.3
)
plt.xlabel("AI Adoption Rate")
plt.ylabel("Productivity Change")
plt.title("AI Adoption vs Productivity")
plt.tight_layout()
plt.savefig("../images/AI_adoption_vs_productivity.png")
plt.close()

# -----------------------------
# Phase 5: ANOVA Test
# -----------------------------

startup = df[df["company_size"] == "Startup"]["ai_adoption_rate"]
sme = df[df["company_size"] == "SME"]["ai_adoption_rate"]
enterprise = df[df["company_size"] == "Enterprise"]["ai_adoption_rate"]

f_stat, p_value = f_oneway(startup, sme, enterprise)

print("\nANOVA TEST")
print("F-Statistic:", f_stat)
print("P-Value:", p_value)

if p_value < 0.05:
    print("Reject H0")
    print("Company size significantly impacts AI adoption.")
else:
    print("Fail to Reject H0")

# -----------------------------
# Phase 6: Segmentation
# -----------------------------

def segment(row):
    if row["ai_adoption_rate"] >= 80:
        return "AI Leaders"
    elif row["employee_satisfaction_score"] >= 80:
        return "High Satisfaction"
    elif row["cost_reduction_percent"] >= 20:
        return "High ROI"
    else:
        return "Slow Adopters"

df["segment"] = df.apply(segment, axis=1)

segment_counts = df["segment"].value_counts()

plt.figure(figsize=(6,4))
segment_counts.plot(kind="bar")
plt.title("Organization Segmentation")
plt.tight_layout()
plt.savefig("../images/organization_segmentation.png")
plt.close()

print("\nSegment Distribution")
print(segment_counts)

print("\nProject Completed Successfully")
