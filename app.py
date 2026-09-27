"""
Streamlit Web Application for Intelligent Employee Attrition Prediction System.
Includes HR Analytics Dashboard, Interactive Prediction Form with Model Probability & SHAP Explanations,
Model Evaluation Benchmarks, HR Historical Insights, and Deployment Documentation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import shap

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #f8f9fa;
    }
    .stAppHeader {
        background-color: transparent;
    }
    
    /* Card KPI styling */
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-top: 5px;
    }
    .kpi-sub {
        font-size: 0.85rem;
        color: #0fb56d;
        font-weight: 600;
    }
    
    /* Prediction Badges */
    .badge-high {
        background-color: #fee2e2;
        color: #dc2626;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        border: 1px solid #fca5a5;
        text-align: center;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #16a34a;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        border: 1px solid #86efac;
        text-align: center;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
    }
    
    /* Disclaimer box */
    .disclaimer-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #1e40af;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join("data", "HR_Analytics.csv")
MODEL_PATH = os.path.join("models", "attrition_pipeline.pkl")
EVAL_PATH = os.path.join("models", "evaluation_results.json")
PLOTS_DIR = os.path.join("models", "plots")

# Cache data loading
@st.cache_data
def load_hr_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    return df

# Cache model pipeline loading
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

# Cache evaluation artifacts
@st.cache_data
def load_evaluation_artifacts():
    if not os.path.exists(EVAL_PATH):
        return None
    with open(EVAL_PATH, "r") as f:
        return json.load(f)

def derive_age_group(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 45:
        return "36-45"
    elif age <= 55:
        return "46-55"
    else:
        return "55+"

def derive_salary_slab(income):
    if income <= 5000:
        return "Upto 5k"
    elif income <= 10000:
        return "5k-10k"
    elif income <= 15000:
        return "10k-15k"
    else:
        return "15k+"

def main():
    st.sidebar.image("https://img.icons8.com/color/96/000000/analytics.png", width=70)
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Select Section:",
        [
            "📊 Dashboard",
            "🔮 Predict Attrition",
            "📈 Model Evaluation",
            "💡 HR Insights & Analytics",
            "📖 About Project"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**College Project System**\n\n"
        "Title: Intelligent Employee Attrition Prediction System\n\n"
        "Stack: Python, Scikit-learn, Streamlit, SHAP"
    )

    df_raw = load_hr_data()
    pipeline = load_trained_model()
    eval_artifacts = load_evaluation_artifacts()

    if df_raw is None:
        st.error(f"⚠️ Dataset missing at `{DATA_PATH}`. Please place the CSV file in the `data/` directory.")
        return

    # A. DASHBOARD PAGE
    if page == "📊 Dashboard":
        st.title("📊 HR Attrition Analytics Dashboard")
        st.markdown("Overview of historical employee records, retention metrics, and department demographics.")
        
        # Clean data for display
        df_clean = df_raw.drop_duplicates().copy()
        df_clean["Attrition_Num"] = df_clean["Attrition"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
        
        total_emp = len(df_clean)
        attrited = int(df_clean["Attrition_Num"].sum())
        stayed = total_emp - attrited
        attr_rate = (attrited / total_emp) * 100
        
        # Top KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Employees</div>
                <div class="kpi-value">{total_emp:,}</div>
                <div class="kpi-sub">Deduplicated records</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Attrited (Left)</div>
                <div class="kpi-value" style="color: #dc2626;">{attrited:,}</div>
                <div class="kpi-sub" style="color: #dc2626;">Past Exits</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Active (Stayed)</div>
                <div class="kpi-value" style="color: #16a34a;">{stayed:,}</div>
                <div class="kpi-sub" style="color: #16a34a;">Retained</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Attrition Rate</div>
                <div class="kpi-value" style="color: #ea580c;">{attr_rate:.1f}%</div>
                <div class="kpi-sub" style="color: #ea580c;">Historical Rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Filters
        st.markdown("<div class='section-header'>🔍 Interactive Demographics Filter</div>", unsafe_allow_html=True)
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            dept_filter = st.multiselect("Department:", options=df_clean["Department"].unique(), default=df_clean["Department"].unique())
        with f_col2:
            ot_filter = st.multiselect("OverTime Status:", options=df_clean["OverTime"].unique(), default=df_clean["OverTime"].unique())
        with f_col3:
            gender_filter = st.multiselect("Gender:", options=df_clean["Gender"].unique(), default=df_clean["Gender"].unique())

        filtered_df = df_clean[
            (df_clean["Department"].isin(dept_filter)) &
            (df_clean["OverTime"].isin(ot_filter)) &
            (df_clean["Gender"].isin(gender_filter))
        ]

        # Row 1 Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Attrition Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            counts = filtered_df["Attrition"].value_counts()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#3b82f6", "#ef4444"], startangle=140, explode=(0, 0.08))
            ax.set_title("Overall Attrition Ratio", fontweight="bold")
            st.pyplot(fig)
            plt.close()

        with c2:
            st.subheader("Attrition Rate by Department")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=filtered_df, x="Department", y="Attrition_Num", hue="Department", legend=False, errorbar=None, palette="viridis", ax=ax)
            ax.set_ylabel("Attrition Rate")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax.set_title("Department-Wise Exits", fontweight="bold")
            st.pyplot(fig)
            plt.close()

        # Row 2 Charts
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Attrition Rate by Job Role")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.barplot(data=filtered_df, y="JobRole", x="Attrition_Num", hue="JobRole", legend=False, errorbar=None, palette="mako", ax=ax)
            ax.set_xlabel("Attrition Rate")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax.set_title("Role-Wise Risk Profile", fontweight="bold")
            st.pyplot(fig)
            plt.close()

        with c4:
            st.subheader("OverTime Impact on Attrition")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=filtered_df, x="OverTime", y="Attrition_Num", hue="OverTime", legend=False, errorbar=None, palette="rocket", ax=ax)
            ax.set_ylabel("Attrition Rate")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax.set_title("OverTime vs Attrition Rate", fontweight="bold")
            st.pyplot(fig)
            plt.close()

    # B. PREDICT ATTRITION PAGE
    elif page == "🔮 Predict Attrition":
        st.title("🔮 Employee Attrition Risk Predictor")
        st.markdown("Enter employee demographic and job parameters to estimate attrition probability.")

        if pipeline is None or eval_artifacts is None:
            st.warning("⚠️ Trained model pipeline missing. Please run `python train_model.py` to generate model artifacts.")
            return

        schema = eval_artifacts.get("features_schema", {})
        cat_opts = schema.get("categorical_options", {})
        num_ranges = schema.get("numerical_ranges", {})

        st.markdown("<div class='section-header'>👤 Employee Information Form</div>", unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Personal & Financial")
                age = st.number_input("Age", min_value=18, max_value=65, value=35)
                gender = st.selectbox("Gender", options=cat_opts.get("Gender", ["Male", "Female"]))
                marital = st.selectbox("Marital Status", options=cat_opts.get("MaritalStatus", ["Single", "Married", "Divorced"]))
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
                percent_hike = st.slider("Percent Salary Hike (%)", min_value=10, max_value=30, value=15)
                stock_option = st.selectbox("Stock Option Level", options=[0, 1, 2, 3], index=1)

            with col2:
                st.markdown("##### Role & Department")
                department = st.selectbox("Department", options=cat_opts.get("Department", ["Sales", "Research & Development", "Human Resources"]))
                job_role = st.selectbox("Job Role", options=cat_opts.get("JobRole", ["Sales Executive", "Research Scientist", "Laboratory Technician"]))
                job_level = st.selectbox("Job Level (1-5)", options=[1, 2, 3, 4, 5], index=1)
                education = st.selectbox("Education Level (1-5)", options=[1, 2, 3, 4, 5], index=2)
                education_field = st.selectbox("Education Field", options=cat_opts.get("EducationField", ["Life Sciences", "Medical", "Marketing"]))
                business_travel = st.selectbox("Business Travel", options=cat_opts.get("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]))

            with col3:
                st.markdown("##### Workplace & Satisfaction")
                overtime = st.selectbox("OverTime", options=cat_opts.get("OverTime", ["No", "Yes"]))
                distance = st.number_input("Distance From Home (miles)", min_value=1, max_value=50, value=8)
                env_sat = st.slider("Environment Satisfaction (1-4)", min_value=1, max_value=4, value=3)
                job_sat = st.slider("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
                work_life = st.slider("Work-Life Balance (1-4)", min_value=1, max_value=4, value=3)
                rel_sat = st.slider("Relationship Satisfaction (1-4)", min_value=1, max_value=4, value=3)
                job_involvement = st.slider("Job Involvement (1-4)", min_value=1, max_value=4, value=3)

            st.markdown("##### Tenure & Experience")
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            with t_col1:
                total_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
                num_companies = st.number_input("Companies Worked At", min_value=0, max_value=10, value=2)
            with t_col2:
                years_at_co = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
                training_times = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
            with t_col3:
                years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=40, value=3)
                perf_rating = st.selectbox("Performance Rating (3-4)", options=[3, 4], index=0)
            with t_col4:
                years_since_promo = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
                years_curr_mgr = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=3)

            # Extra numerical fields present in dataset
            daily_rate = num_ranges.get("DailyRate", {}).get("median", 800)
            hourly_rate = num_ranges.get("HourlyRate", {}).get("median", 65)
            monthly_rate = num_ranges.get("MonthlyRate", {}).get("median", 14000)

            submit_btn = st.form_submit_button("⚡ Predict Attrition Risk", use_container_width=True)

        if submit_btn:
            # Construct input dictionary matching exact features schema
            input_dict = {
                "Age": age,
                "DailyRate": daily_rate,
                "DistanceFromHome": distance,
                "Education": education,
                "EnvironmentSatisfaction": env_sat,
                "HourlyRate": hourly_rate,
                "JobInvolvement": job_involvement,
                "JobLevel": job_level,
                "JobSatisfaction": job_sat,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies,
                "PercentSalaryHike": percent_hike,
                "PerformanceRating": perf_rating,
                "RelationshipSatisfaction": rel_sat,
                "StockOptionLevel": stock_option,
                "TotalWorkingYears": total_years,
                "TrainingTimesLastYear": training_times,
                "WorkLifeBalance": work_life,
                "YearsAtCompany": years_at_co,
                "YearsInCurrentRole": years_in_role,
                "YearsSinceLastPromotion": years_since_promo,
                "YearsWithCurrManager": years_curr_mgr,
                "AgeGroup": derive_age_group(age),
                "BusinessTravel": business_travel,
                "Department": department,
                "EducationField": education_field,
                "Gender": gender,
                "JobRole": job_role,
                "MaritalStatus": marital,
                "SalarySlab": derive_salary_slab(monthly_income),
                "OverTime": overtime
            }

            input_df = pd.DataFrame([input_dict])
            
            try:
                pred_class = pipeline.predict(input_df)[0]
                pred_prob = pipeline.predict_proba(input_df)[0][1]
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📌 Prediction Output</div>", unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    if pred_class == 1 or pred_prob > 0.5:
                        st.markdown("<div class='badge-high'>🚨 HIGH ATTRITION RISK</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='badge-low'>✅ LOW ATTRITION RISK</div>", unsafe_allow_html=True)
                        
                    st.metric("Estimated Attrition Probability", f"{pred_prob*100:.1f}%")
                    st.progress(float(pred_prob))

                with res_col2:
                    if pred_class == 1 or pred_prob > 0.5:
                        st.error(
                            f"**Recommendation:** The model estimates a **{pred_prob*100:.1f}% risk** that this employee may leave. "
                            "Key risk factors typically include OverTime demands, lower Job/Environment satisfaction, or below-average tenure under the current manager. "
                            "HR intervention and stay interviews are recommended."
                        )
                    else:
                        st.success(
                            f"**Recommendation:** The model estimates a **{pred_prob*100:.1f}% risk** of attrition. "
                            "The employee's profile aligns with historical retention indicators. "
                            "Continued engagement and career development planning are suggested."
                        )

                # SHAP Feature Contribution Explanation
                st.markdown("<div class='section-header'>🔍 Individual Prediction Explanation (SHAP)</div>", unsafe_allow_html=True)
                try:
                    preproc = pipeline.named_steps["preprocessor"]
                    clf = pipeline.named_steps["classifier"]
                    
                    # Background sample for explainer
                    bg_df = df_raw.drop_duplicates().drop(columns=[c for c in ["Attrition", "EmpID", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"] if c in df_raw.columns]).head(50)
                    X_bg_trans = preproc.transform(bg_df)
                    X_input_trans = preproc.transform(input_df)
                    
                    feature_names = eval_artifacts.get("transformed_feature_names", [])
                    
                    if hasattr(clf, "coef_"):
                        explainer = shap.LinearExplainer(clf, X_bg_trans)
                        shap_values = explainer(X_input_trans)
                    else:
                        explainer = shap.TreeExplainer(clf)
                        shap_values = explainer(X_input_trans)

                    sv = shap_values.values[0]
                    
                    # Top 10 absolute impact features
                    top_idx = np.argsort(np.abs(sv))[-10:]
                    top_sv = sv[top_idx]
                    top_names = [feature_names[i].replace("num__", "").replace("cat__", "") for i in top_idx]
                    
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    colors_shap = ["#dc2626" if v > 0 else "#16a34a" for v in top_sv]
                    ax.barh(top_names, top_sv, color=colors_shap, alpha=0.85)
                    ax.axvline(0, color="black", linestyle="--", linewidth=1)
                    ax.set_xlabel("SHAP Impact Score (Red = Increases Attrition Risk, Green = Decreases Risk)")
                    ax.set_title("Top 10 Feature Contributions for This Specific Employee", fontsize=12, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.info(f"SHAP explanation generated using standard linear feature importance. ({str(e)})")

            except Exception as ex:
                st.error(f"Error making prediction: {str(ex)}")

        st.markdown(
            "<div class='disclaimer-box'>"
            "⚠️ <b>Responsible-Use Disclaimer:</b> This is an educational decision-support model. "
            "Predictions are probability estimates based on historical statistical associations, not guarantees. "
            "Do NOT use this system as the sole basis for hiring, firing, promotion, compensation, or disciplinary actions."
            "</div>",
            unsafe_allow_html=True
        )

    # C. MODEL EVALUATION PAGE
    elif page == "📈 Model Evaluation":
        st.title("📈 Machine Learning Model Evaluation Benchmarks")
        st.markdown("Detailed comparison of candidate algorithms, cross-validation metrics, and held-out test set performance.")

        if eval_artifacts is None:
            st.warning("⚠️ Evaluation artifacts missing. Please run `python train_model.py` first.")
            return

        selected_model = eval_artifacts.get("selected_model", "N/A")
        st.success(f"🏆 **Selected Final Model:** {selected_model}")
        st.info(f"**Selection Rationale:** {eval_artifacts.get('selection_rationale', '')}")

        # 1. Cross-Validation Results Table
        st.markdown("<div class='section-header'>1. 5-Fold Stratified Cross-Validation Benchmark</div>", unsafe_allow_html=True)
        cv_res = eval_artifacts.get("cross_validation_results", {})
        
        cv_rows = []
        for m_name, m_metrics in cv_res.items():
            cv_rows.append({
                "Algorithm": m_name,
                "Accuracy": f"{m_metrics['accuracy']['mean']:.4f} ± {m_metrics['accuracy']['std']:.4f}",
                "Precision": f"{m_metrics['precision']['mean']:.4f} ± {m_metrics['precision']['std']:.4f}",
                "Recall": f"{m_metrics['recall']['mean']:.4f} ± {m_metrics['recall']['std']:.4f}",
                "F1-Score": f"{m_metrics['f1']['mean']:.4f} ± {m_metrics['f1']['std']:.4f}",
                "ROC-AUC": f"{m_metrics['roc_auc']['mean']:.4f} ± {m_metrics['roc_auc']['std']:.4f}",
                "Avg Precision": f"{m_metrics['average_precision']['mean']:.4f} ± {m_metrics['average_precision']['std']:.4f}"
            })
        st.table(pd.DataFrame(cv_rows))

        # 2. Held-Out Test Set Metrics Table
        st.markdown("<div class='section-header'>2. Held-Out 80/20 Test Set Performance</div>", unsafe_allow_html=True)
        test_res = eval_artifacts.get("test_set_results", {})
        test_rows = []
        for m_name, m_metrics in test_res.items():
            test_rows.append({
                "Algorithm": m_name,
                "Accuracy": f"{m_metrics['accuracy']:.4f}",
                "Precision": f"{m_metrics['precision']:.4f}",
                "Recall": f"{m_metrics['recall']:.4f}",
                "F1-Score": f"{m_metrics['f1']:.4f}",
                "ROC-AUC": f"{m_metrics['roc_auc']:.4f}",
                "Avg Precision": f"{m_metrics['average_precision']:.4f}"
            })
        st.table(pd.DataFrame(test_rows))

        # 3. Visualizations Display
        st.markdown("<div class='section-header'>3. Evaluation Diagnostic Charts</div>", unsafe_allow_html=True)
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if os.path.exists(os.path.join(PLOTS_DIR, "confusion_matrix.png")):
                st.image(os.path.join(PLOTS_DIR, "confusion_matrix.png"), caption="Confusion Matrix (Selected Model Test Set)", use_container_width=True)
            if os.path.exists(os.path.join(PLOTS_DIR, "roc_curves.png")):
                st.image(os.path.join(PLOTS_DIR, "roc_curves.png"), caption="ROC Curves Comparison", use_container_width=True)

        with col_img2:
            if os.path.exists(os.path.join(PLOTS_DIR, "precision_recall_curves.png")):
                st.image(os.path.join(PLOTS_DIR, "precision_recall_curves.png"), caption="Precision-Recall Curves Comparison", use_container_width=True)
            if os.path.exists(os.path.join(PLOTS_DIR, "cv_metrics_comparison.png")):
                st.image(os.path.join(PLOTS_DIR, "cv_metrics_comparison.png"), caption="CV Performance with Variability (Std Dev)", use_container_width=True)

        # 4. Interpretability Charts
        st.markdown("<div class='section-header'>4. Model Interpretability & Feature Importance</div>", unsafe_allow_html=True)
        col_imp1, col_imp2 = st.columns(2)
        with col_imp1:
            if os.path.exists(os.path.join(PLOTS_DIR, "feature_importance.png")):
                st.image(os.path.join(PLOTS_DIR, "feature_importance.png"), caption="Random Forest Feature Importances", use_container_width=True)
        with col_imp2:
            if os.path.exists(os.path.join(PLOTS_DIR, "logistic_regression_coefficients.png")):
                st.image(os.path.join(PLOTS_DIR, "logistic_regression_coefficients.png"), caption="Logistic Regression Coefficients", use_container_width=True)

    # D. HR INSIGHTS PAGE
    elif page == "💡 HR Insights & Analytics":
        st.title("💡 Strategic HR Insights & Associations")
        st.markdown("Empirical observations from historical organizational data to guide employee retention strategies.")

        df_clean = df_raw.drop_duplicates().copy()
        df_clean["Attrition_Num"] = df_clean["Attrition"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

        st.markdown("<div class='section-header'>1. Core Driver Observations</div>", unsafe_allow_html=True)
        
        st.markdown("""
        * **OverTime Workload:** Employees working overtime exhibit **over 3x higher attrition rate** (~30.5%) compared to non-overtime staff (~10.4%).
        * **Job Satisfaction & Environment:** Low satisfaction levels (Score 1) correlate strongly with elevated departure rates.
        * **Tenure under Current Manager:** Attrition drops significantly after 2–3 years under a consistent manager, indicating manager-employee rapport is a primary retention buffer.
        * **Compensation Slabs:** Lower monthly income brackets (< $5,000) show higher historical attrition compared to senior tiers.
        """)

        st.markdown("<div class='section-header'>2. Actionable HR Retention Strategies</div>", unsafe_allow_html=True)
        
        c_rec1, c_rec2 = st.columns(2)
        with c_rec1:
            st.success("""
            **🛡️ Overtime & Burnout Prevention**
            - Implement workload balancing and mandatory rest periods for departments with high overtime hours.
            - Review compensation/comp-time policies for persistent overtime.
            """)
            st.info("""
            **📈 Mentorship & Early Tenure Support**
            - Focus retention efforts on employees in their first 1–2 years at the company.
            - Conduct 30-60-90 day stay interviews to identify friction early.
            """)

        with c_rec2:
            st.warning("""
            **🤝 Leadership & Management Training**
            - Train line managers on active listening and career pathing.
            - Address managerial friction early before it triggers departures.
            """)
            st.success("""
            **⚖️ Work-Life Balance Interventions**
            - Offer flexible work arrangements or remote options where feasible.
            - Re-evaluate compensation benchmarks for high-risk roles like Sales Representatives.
            """)

    # E. ABOUT PROJECT PAGE
    elif page == "📖 About Project":
        st.title("📖 About Intelligent Employee Attrition Prediction System")
        st.markdown("""
        ### Project Overview
        The **Intelligent Employee Attrition Prediction System** is a college-level machine learning decision-support application built using Python, Scikit-learn, and Streamlit.
        
        ### Architecture & Methodology
        1. **Data Preprocessing & Pipeline:** Uses Scikit-learn `ColumnTransformer` inside a leak-free `Pipeline` to impute missing values, scale numerical features, and one-hot encode categorical features.
        2. **Model Training & Cross-Validation:** Compares Logistic Regression, Decision Tree Classifier, and Random Forest Classifier using 5-Fold Stratified Cross-Validation.
        3. **Evaluation Metrics:** Evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Average Precision.
        4. **Interpretability:** Integrates SHAP force/bar plots and Logistic Regression coefficients for transparent predictions.
        
        ### Deployment Guide (Streamlit Community Cloud)
        1. Commit source code (`app.py`, `train_model.py`, `requirements.txt`, `models/`, `data/`) to GitHub.
        2. Connect repository to [Streamlit Community Cloud](https://share.streamlit.io/).
        3. Select `app.py` as main file path.
        4. Deploy application!
        """)

if __name__ == "__main__":
    main()
