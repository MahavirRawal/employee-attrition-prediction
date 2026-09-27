"""
Exploratory Data Analysis (EDA) Script for Intelligent Employee Attrition Prediction System.
Calculates historical associations, group-level attrition rates, missing value summaries,
and outputs dataset metrics without assuming causation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join("data", "HR_Analytics.csv")
OUTPUT_DIR = os.path.join("notebooks", "eda_plots")

def load_and_validate_data(filepath=DATA_PATH):
    """Load dataset and perform initial schema checks."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure data/HR_Analytics.csv is present.")
    
    df = pd.read_csv(filepath)
    print(f"=== DATASET OVERVIEW ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print("\n=== MISSING VALUES ===")
    if missing_cols.empty:
        print("No missing values found.")
    else:
        for col, count in missing_cols.items():
            print(f" - {col}: {count} missing values ({count/len(df)*100:.2f}%)")
            
    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n=== DUPLICATE RECORDS ===")
    print(f"Total duplicate rows: {dup_count}")
    
    # Target distribution
    if "Attrition" not in df.columns:
        raise ValueError("Target column 'Attrition' not found in dataset.")
        
    print("\n=== ATTRITION DISTRIBUTION ===")
    attrition_counts = df["Attrition"].value_counts()
    attrition_pcts = df["Attrition"].value_counts(normalize=True) * 100
    for val in attrition_counts.index:
        print(f" - {val}: {attrition_counts[val]} ({attrition_pcts[val]:.2f}%)")
        
    return df

def run_exploratory_analysis(df):
    """Perform detailed categorical and numerical association analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    df_clean = df.drop_duplicates().copy()
    df_clean["Attrition_Num"] = df_clean["Attrition"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    
    print("\n=== GROUP ATTRITION RATES ===")
    
    # Department
    if "Department" in df_clean.columns:
        dept_rates = df_clean.groupby("Department")["Attrition_Num"].agg(["count", "sum", "mean"]).rename(
            columns={"count": "Total", "sum": "Attrited", "mean": "AttritionRate"}
        )
        dept_rates["AttritionRate"] = dept_rates["AttritionRate"] * 100
        print("\nAttrition by Department:")
        print(dept_rates.to_string())
        
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_clean, x="Department", y="Attrition_Num", hue="Department", legend=False, errorbar=None, palette="viridis")
        plt.title("Historical Attrition Rate by Department", fontsize=14, fontweight="bold")
        plt.ylabel("Attrition Rate", fontsize=12)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "attrition_by_department.png"), dpi=300)
        plt.close()

    # JobRole
    if "JobRole" in df_clean.columns:
        role_rates = df_clean.groupby("JobRole")["Attrition_Num"].agg(["count", "sum", "mean"]).rename(
            columns={"count": "Total", "sum": "Attrited", "mean": "AttritionRate"}
        )
        role_rates["AttritionRate"] = role_rates["AttritionRate"] * 100
        print("\nAttrition by Job Role:")
        print(role_rates.to_string())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_clean, y="JobRole", x="Attrition_Num", hue="JobRole", legend=False, errorbar=None, palette="mako")
        plt.title("Historical Attrition Rate by Job Role", fontsize=14, fontweight="bold")
        plt.xlabel("Attrition Rate", fontsize=12)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "attrition_by_job_role.png"), dpi=300)
        plt.close()

    # OverTime
    if "OverTime" in df_clean.columns:
        ot_rates = df_clean.groupby("OverTime")["Attrition_Num"].agg(["count", "sum", "mean"]).rename(
            columns={"count": "Total", "sum": "Attrited", "mean": "AttritionRate"}
        )
        ot_rates["AttritionRate"] = ot_rates["AttritionRate"] * 100
        print("\nAttrition by OverTime Status:")
        print(ot_rates.to_string())
        
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df_clean, x="OverTime", y="Attrition_Num", hue="OverTime", legend=False, errorbar=None, palette="rocket")
        plt.title("Attrition Rate by OverTime Status", fontsize=14, fontweight="bold")
        plt.ylabel("Attrition Rate", fontsize=12)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "attrition_by_overtime.png"), dpi=300)
        plt.close()

    # JobSatisfaction
    if "JobSatisfaction" in df_clean.columns:
        plt.figure(figsize=(7, 4))
        sns.barplot(data=df_clean, x="JobSatisfaction", y="Attrition_Num", hue="JobSatisfaction", legend=False, errorbar=None, palette="Blues_d")
        plt.title("Attrition Rate by Job Satisfaction (1=Low, 4=Very High)", fontsize=14, fontweight="bold")
        plt.ylabel("Attrition Rate", fontsize=12)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "attrition_by_job_satisfaction.png"), dpi=300)
        plt.close()

    # WorkLifeBalance
    if "WorkLifeBalance" in df_clean.columns:
        plt.figure(figsize=(7, 4))
        sns.barplot(data=df_clean, x="WorkLifeBalance", y="Attrition_Num", hue="WorkLifeBalance", legend=False, errorbar=None, palette="Greens_d")
        plt.title("Attrition Rate by Work-Life Balance (1=Bad, 4=Best)", fontsize=14, fontweight="bold")
        plt.ylabel("Attrition Rate", fontsize=12)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "attrition_by_worklife_balance.png"), dpi=300)
        plt.close()

    # YearsAtCompany distribution
    if "YearsAtCompany" in df_clean.columns:
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df_clean, x="Attrition", y="YearsAtCompany", hue="Attrition", legend=False, palette="Set2")
        plt.title("Years at Company by Attrition Status", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "years_at_company_by_attrition.png"), dpi=300)
        plt.close()

    print(f"\nEDA charts successfully saved to '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    df = load_and_validate_data()
    run_exploratory_analysis(df)
