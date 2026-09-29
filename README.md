# 👔 Intelligent Employee Attrition Prediction System

An end-to-end, decision-support Machine Learning application and interactive HR analytics dashboard built to analyze historical employee data and estimate attrition risk.

---

## 📌 1. Project Overview & Problem Statement

Employee turnover (attrition) imposes substantial costs on organizations through recruitment fees, lost productivity, and team disruption. The **Intelligent Employee Attrition Prediction System** leverages historical HR data to identify key association patterns linked with employee departures and provides probability-based attrition risk predictions for individual employee profiles.

### Educational, Responsible-Use & Fairness Notice
> ⚠️ **Responsible-Use & Fairness Disclaimer:** 
> - **Educational Decision-Support Tool:** This application is designed purely as a decision-support assistant. Model outputs represent statistical estimates based on historical association patterns and are **not guaranteed proof** or definitive predictions that an individual employee will leave.
> - **Exclusion of Sensitive Attributes:** Sensitive demographic features (`Gender` and `MaritalStatus`) have been explicitly removed from all predictor feature inputs and prediction forms to align with responsible AI principles.
> - **Proxy Bias Warning:** Removing `Gender` and `MaritalStatus` does **not** guarantee a completely fair or unbiased model, as other features (such as JobRole, MonthlyIncome, TotalWorkingYears, or Department) may still act as indirect proxies for demographic attributes.
> - **Prohibited Uses:** This tool must **never** be used as the sole basis for hiring, firing, promotion, compensation, performance evaluation, or disciplinary actions.

---

## 🎯 2. Project Objectives

- **Predict Attrition Risk:** Classify employee departure likelihood (Yes/No) along with estimated risk probability using 30 non-sensitive workplace features.
- **Excluded Protected Attributes:** Remove `Gender` and `MaritalStatus` from model input features and prediction forms while keeping historical CSV data intact.
- **Leakage-Free ML Pipeline:** Utilize Scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage during preprocessing, scaling, and one-hot encoding.
- **Rigorous Cross-Validation:** Benchmark 3 classifiers (Logistic Regression, Decision Tree, Random Forest) using 5-Fold Stratified Cross-Validation.
- **Model Interpretability:** Integrate SHAP (SHapley Additive exPlanations) and Logistic Regression coefficients to explain individual and global model predictions.
- **Interactive HR Dashboard:** Provide an interactive Streamlit UI for exploring department-level attrition, role-wise risk, overtime impacts, and model evaluation metrics.

---

## 🛠️ 3. Technology Stack

- **Language:** Python 3.10+
- **Data Processing:** Pandas, NumPy
- **Machine Learning & Pipeline:** Scikit-learn
- **Model Interpretability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Interactive UI:** Streamlit
- **Model Serialization:** Joblib

---

## 📂 4. Project Folder Structure

```text
employee-attrition-prediction/
├── data/
│   └── HR_Analytics.csv               # Historical HR dataset (38 columns)
├── models/
│   ├── attrition_pipeline.pkl          # Serialized scikit-learn pipeline (30 features)
│   ├── evaluation_results.json        # Detailed CV & test evaluation metrics
│   └── plots/                          # Saved diagnostic & metric plots
│       ├── cv_metrics_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       ├── precision_recall_curves.png
│       ├── feature_importance.png
│       └── logistic_regression_coefficients.png
├── notebooks/
│   ├── exploratory_analysis.py        # Standalone EDA script
│   ├── exploratory_analysis.ipynb     # Jupyter Notebook version
│   └── eda_plots/                     # Generated EDA plots
├── app.py                              # Streamlit web application
├── train_model.py                      # Model training & CV script
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Version control exclusions
└── README.md                           # Documentation
```

---

## ⚙️ 5. Installation & Setup (Windows)

### Step 1: Open PowerShell or Command Prompt
Navigate to the project root directory:
```powershell
cd "c:\Users\Mahavir Rawal\Desktop\employee-attrition-prediction"
```

### Step 2: Create & Activate Virtual Environment (Optional but Recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 6. Execution Instructions

### A. Run Exploratory Data Analysis (EDA)
Generate group-level attrition rates and descriptive association plots:
```powershell
python notebooks/exploratory_analysis.py
```
*Outputs saved to `notebooks/eda_plots/`.*

### B. Train Machine Learning Models
Run 5-Fold Stratified Cross-Validation on the 30 predictor features (excluding `Gender` and `MaritalStatus`), evaluate on held-out 80/20 test set, and save model artifacts:
```powershell
python train_model.py
```
*Artifacts saved to `models/attrition_pipeline.pkl` and `models/evaluation_results.json`.*

### C. Launch Interactive Streamlit Dashboard
Start the local Streamlit application:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 7. Model Evaluation & Benchmark Results

### A. 5-Fold Stratified Cross-Validation Comparison (30 Features)

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **0.7576 ± 0.0347** | **0.3734 ± 0.0488** | **0.7296 ± 0.0667** | **0.4936 ± 0.0571** | **0.8269 ± 0.0407** | **0.5996 ± 0.0718** |
| Random Forest | 0.8533 ± 0.0190 | 0.5753 ± 0.1091 | 0.3586 ± 0.0546 | 0.4407 ± 0.0694 | 0.7959 ± 0.0423 | 0.5238 ± 0.0608 |
| Decision Tree | 0.7257 ± 0.0250 | 0.2998 ± 0.0174 | 0.5191 ± 0.0485 | 0.3785 ± 0.0144 | 0.6204 ± 0.0404 | 0.3094 ± 0.0450 |

### B. Held-Out 80/20 Test Set Results (Logistic Regression)
- **Accuracy:** 74.92%
- **Precision:** 37.61%
- **Recall (Sensitivity):** 87.23% *(Identifies 87.2% of actual attrition cases)*
- **F1-Score:** 52.56%
- **ROC-AUC:** 0.8888
- **Average Precision (PR-AUC):** 0.7093

### C. Model Selection Rationale
`Logistic Regression` with `class_weight='balanced'` was selected as the final production model because it achieved the highest combined ROC-AUC (0.8269 CV / 0.8888 Test) and F1-score (0.4936 CV / 0.5256 Test) across 5-fold cross-validation, while achieving high **Recall (87.23%)** on the held-out test set. In employee attrition prediction, identifying potential exits (high recall) is critical for proactive retention interventions.

---

## 🌐 8. Deployment to Streamlit Community Cloud

Follow these steps to deploy the application online for free:

1. **GitHub Repository Setup:**
   - Create a public or private repository on GitHub.
   - Commit and push all application code: `app.py`, `train_model.py`, `requirements.txt`, `README.md`, `.gitignore`, `data/HR_Analytics.csv`, and `models/`.

2. **Connect Streamlit Community Cloud:**
   - Log into [Streamlit Community Cloud](https://share.streamlit.io/).
   - Click **New app**.
   - Select your GitHub repository, branch (`main`), and set Main file path to `app.py`.

3. **Deploy & Verify:**
   - Click **Deploy!**
   - Streamlit will automatically install dependencies from `requirements.txt` and launch the live web dashboard.

---

## ⚠️ 9. Limitations & Responsible Use

- **Dataset Size:** The dataset contains 1,473 deduplicated records. Larger organizational datasets can improve precision.
- **Cross-Sectional Data:** The dataset lacks longitudinal tracking (time-series retention over multiple fiscal quarters).
- **Proxy Relationships:** Excluding `Gender` and `MaritalStatus` eliminates direct model dependencies on protected demographic attributes, but indirect proxy relationships may still persist in workplace features.
- **Decision-Support Boundary:** Model outputs are experimental probability estimates to assist HR professionals in identifying early intervention opportunities, not automated determinants of employee status or career outcomes.
- **Future Enhancements:** Incorporate XGBoost/LightGBM classifiers, add real-time HR survey inputs, and implement automated stay-interview workflow integration.
