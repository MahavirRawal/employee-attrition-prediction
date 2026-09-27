# 👔 Intelligent Employee Attrition Prediction System

An end-to-end, decision-support Machine Learning application and interactive HR analytics dashboard built to analyze historical employee data and estimate attrition risk.

---

## 📌 1. Project Overview & Problem Statement

Employee turnover (attrition) imposes substantial costs on organizations through recruitment fees, lost productivity, and team disruption. The **Intelligent Employee Attrition Prediction System** leverages historical HR data to identify key association patterns linked with employee departures and provides probability-based attrition risk predictions for individual employee profiles.

### Educational & Decision-Support Notice
> ⚠️ **Responsible-Use Disclaimer:** This application is designed as an educational decision-support tool. Model outputs represent statistical estimates based on historical association patterns and are **not guaranteed proof** that an employee will leave. This tool must **not** be used as the sole basis for hiring, firing, promotion, compensation, or disciplinary decisions.

---

## 🎯 2. Project Objectives

- **Predict Attrition Risk:** Classify employee departure likelihood (Yes/No) along with estimated risk probability.
- **Leakage-Free ML Pipeline:** Utilize Scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage during preprocessing and feature scaling/encoding.
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
│   └── HR_Analytics.csv               # Historical HR dataset
├── models/
│   ├── attrition_pipeline.pkl          # Serialized scikit-learn pipeline
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
Run 5-Fold Stratified Cross-Validation, compare classifiers, evaluate on held-out test set, and save model artifacts:
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

### A. 5-Fold Stratified Cross-Validation Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **0.7651 ± 0.0397** | **0.3817 ± 0.0567** | **0.7168 ± 0.0804** | **0.4974 ± 0.0660** | **0.8313 ± 0.0447** | **0.6144 ± 0.0868** |
| Random Forest | 0.8561 ± 0.0147 | 0.5872 ± 0.0756 | 0.3714 ± 0.0441 | 0.4536 ± 0.0498 | 0.7993 ± 0.0379 | 0.5168 ± 0.0739 |
| Decision Tree | 0.7257 ± 0.0279 | 0.2977 ± 0.0279 | 0.5106 ± 0.0618 | 0.3745 ± 0.0318 | 0.6181 ± 0.0427 | 0.3009 ± 0.0383 |

### B. Held-Out 80/20 Test Set Results (Logistic Regression)
- **Accuracy:** 76.95%
- **Precision:** 39.81%
- **Recall (Sensitivity):** 87.23% *(Identifies 87% of actual attrition cases)*
- **F1-Score:** 54.67%
- **ROC-AUC:** 0.8912
- **Average Precision (PR-AUC):** 0.7127

### C. Model Selection Rationale
`Logistic Regression` with `class_weight='balanced'` was selected as the final production model because it demonstrated superior ROC-AUC (0.8313 CV / 0.8912 Test) and Average Precision (0.6144 CV / 0.7127 Test), while achieving high **Recall (87.23%)** on the held-out test set. In employee attrition prediction, identifying potential exits (high recall) is critical for proactive retention interventions.

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

## ⚠️ 9. Limitations & Future Enhancements

- **Dataset Size:** The dataset contains 1,473 deduplicated records. Larger organizational datasets can improve precision.
- **Cross-Sectional Data:** The dataset lacks longitudinal tracking (time-series retention over multiple fiscal quarters).
- **Future Enhancements:** Incorporate XGBoost/LightGBM classifiers, add real-time HR survey inputs, and implement automated stay-interview workflow integration.
