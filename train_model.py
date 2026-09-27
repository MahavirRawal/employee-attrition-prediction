"""
Model Training & Evaluation Script for Intelligent Employee Attrition Prediction System.
Implements leak-free Scikit-learn Pipeline with ColumnTransformer, 5-Fold Stratified Cross-Validation,
model comparison across Logistic Regression, Decision Tree, and Random Forest, held-out 80/20 test set
evaluation, metric plots, interpretability, and serialization.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

DATA_PATH = os.path.join("data", "HR_Analytics.csv")
MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "attrition_pipeline.pkl")
EVAL_RESULTS_PATH = os.path.join(MODEL_DIR, "evaluation_results.json")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")

RANDOM_STATE = 42

def load_and_clean_dataset(filepath=DATA_PATH):
    """Load dataset, handle missing target, remove duplicates, and drop identifiers."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset missing at {filepath}.")
        
    df = pd.read_csv(filepath)
    initial_rows = len(df)
    
    # Drop exact duplicate rows
    df = df.drop_duplicates()
    dedup_rows = len(df)
    print(f"Dataset loaded: {initial_rows} initial rows -> {dedup_rows} deduplicated rows.")
    
    if "Attrition" not in df.columns:
        raise ValueError("Target column 'Attrition' absent in dataset.")
        
    # Map target column
    y = df["Attrition"].astype(str).str.strip().str.lower().apply(lambda x: 1 if x == "yes" else 0)
    
    # Identify columns to drop: target, IDs, constants
    drop_cols = ["Attrition"]
    id_cols = [c for c in ["EmpID", "EmployeeNumber"] if c in df.columns]
    constant_cols = [c for c in df.columns if c not in drop_cols and df[c].nunique(dropna=False) <= 1]
    
    exclude_cols = drop_cols + id_cols + constant_cols
    X = df.drop(columns=exclude_cols)
    
    print(f"Features count: {X.shape[1]}")
    print(f"Excluded columns: {exclude_cols}")
    print(f"Target distribution - Stayed (0): {(y == 0).sum()}, Attrited (1): {(y == 1).sum()}")
    
    return X, y, df

def build_preprocessor(X):
    """Build ColumnTransformer for numerical scaling & categorical one-hot encoding."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )
    
    return preprocessor, num_cols, cat_cols

def get_transformed_feature_names(fitted_preprocessor, num_cols, cat_cols):
    """Recover feature names after ColumnTransformer one-hot encoding."""
    feature_names = list(num_cols)
    
    # Extract categorical encoded names
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    
    feature_names.extend(encoded_cat_names)
    return feature_names

def train_and_evaluate_models():
    """Main model training, cross-validation, test evaluation, and saving function."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    X, y, df_raw = load_and_clean_dataset()
    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    
    # Model candidates
    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, class_weight="balanced")
    }
    
    # 1. 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    
    cv_results_summary = {}
    
    print("\n=================== 5-FOLD STRATIFIED CV RESULTS ===================")
    for model_name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        cv_scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        summary = {}
        for metric in scoring:
            key = f"test_{metric}"
            summary[metric] = {
                "mean": float(np.mean(cv_scores[key])),
                "std": float(np.std(cv_scores[key]))
            }
        cv_results_summary[model_name] = summary
        
        print(f"\n--- {model_name} ---")
        print(f" Accuracy:          {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
        print(f" Precision:         {summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}")
        print(f" Recall:            {summary['recall']['mean']:.4f} ± {summary['recall']['std']:.4f}")
        print(f" F1-Score:          {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}")
        print(f" ROC-AUC:           {summary['roc_auc']['mean']:.4f} ± {summary['roc_auc']['std']:.4f}")
        print(f" Avg Precision:     {summary['average_precision']['mean']:.4f} ± {summary['average_precision']['std']:.4f}")

    # 2. Held-Out 80/20 Test Set Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    
    fitted_pipelines = {}
    test_metrics = {}
    probs_test = {}
    preds_test = {}
    
    print("\n=================== HELD-OUT TEST SET (80/20) RESULTS ===================")
    for model_name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline
        
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]
        
        preds_test[model_name] = preds
        probs_test[model_name] = probs
        
        test_metrics[model_name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "average_precision": float(average_precision_score(y_test, probs)),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classification_report": classification_report(y_test, preds, output_dict=True)
        }
        
        print(f"\n--- {model_name} Test Metrics ---")
        print(f" Accuracy:      {test_metrics[model_name]['accuracy']:.4f}")
        print(f" Precision:     {test_metrics[model_name]['precision']:.4f}")
        print(f" Recall:        {test_metrics[model_name]['recall']:.4f}")
        print(f" F1-Score:      {test_metrics[model_name]['f1']:.4f}")
        print(f" ROC-AUC:       {test_metrics[model_name]['roc_auc']:.4f}")
        print(f" Avg Precision: {test_metrics[model_name]['average_precision']:.4f}")

    # 3. Model Selection
    # Compare based on ROC-AUC and F1 on CV
    best_model_name = max(
        models.keys(),
        key=lambda m: cv_results_summary[m]["roc_auc"]["mean"] + cv_results_summary[m]["f1"]["mean"]
    )
    
    selection_rationale = (
        f"{best_model_name} was selected as the final model because it achieved the highest combined "
        f"ROC-AUC ({cv_results_summary[best_model_name]['roc_auc']['mean']:.4f}) and "
        f"F1-score ({cv_results_summary[best_model_name]['f1']['mean']:.4f}) across 5-fold cross-validation, "
        f"demonstrating strong capability in identifying minority-class attrition with balanced precision and recall."
    )
    print(f"\n=================== SELECTED MODEL ===================")
    print(f"Best Model: {best_model_name}")
    print(f"Rationale: {selection_rationale}")

    # 4. Extract Feature Names and Interpretability
    best_pipeline = fitted_pipelines[best_model_name]
    fitted_preproc = best_pipeline.named_steps["preprocessor"]
    transformed_feature_names = get_transformed_feature_names(fitted_preproc, num_cols, cat_cols)
    
    # Feature Importances from Random Forest
    rf_pipeline = fitted_pipelines["Random Forest"]
    rf_importances = rf_pipeline.named_steps["classifier"].feature_importances_
    rf_importance_df = pd.DataFrame({
        "feature": transformed_feature_names,
        "importance": rf_importances
    }).sort_values(by="importance", ascending=False)
    
    # Logistic Regression Coefficients
    lr_pipeline = fitted_pipelines["Logistic Regression"]
    lr_coefs = lr_pipeline.named_steps["classifier"].coef_[0]
    lr_coef_df = pd.DataFrame({
        "feature": transformed_feature_names,
        "coefficient": lr_coefs
    }).sort_values(by="coefficient", ascending=False)

    # 5. Generate and Save Visualizations
    generate_evaluation_plots(
        cv_results_summary, test_metrics, y_test, probs_test,
        preds_test, best_model_name, rf_importance_df, lr_coef_df
    )

    # 6. Save Final Pipeline Artifact
    joblib.dump(best_pipeline, PIPELINE_PATH)
    print(f"\nSaved best model pipeline to '{PIPELINE_PATH}'.")

    # 7. Save Evaluation Results JSON
    evaluation_export = {
        "dataset_info": {
            "total_records": len(df_raw),
            "clean_records": len(X),
            "num_features": X.shape[1],
            "attrition_counts": {"Stayed (0)": int((y == 0).sum()), "Attrited (1)": int((y == 1).sum())},
            "attrition_rate_pct": float(y.mean() * 100)
        },
        "models_evaluated": list(models.keys()),
        "selected_model": best_model_name,
        "selection_rationale": selection_rationale,
        "cross_validation_results": cv_results_summary,
        "test_set_results": test_metrics,
        "transformed_feature_names": transformed_feature_names,
        "random_forest_top_features": rf_importance_df.head(15).to_dict(orient="records"),
        "logistic_regression_coefficients": lr_coef_df.to_dict(orient="records"),
        "features_schema": {
            "numerical": num_cols,
            "categorical": cat_cols,
            "categorical_options": {c: X[c].dropna().unique().tolist() for c in cat_cols},
            "numerical_ranges": {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())} for c in num_cols}
        },
        "random_state": RANDOM_STATE
    }
    
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(evaluation_export, f, indent=4)
        
    print(f"Saved evaluation metrics JSON to '{EVAL_RESULTS_PATH}'.")
    return evaluation_export

def generate_evaluation_plots(cv_summary, test_metrics, y_test, probs_test, preds_test, best_model, rf_imp_df, lr_coef_df):
    """Generate all required evaluation and interpretability charts."""
    sns.set_theme(style="whitegrid")
    
    # 1. Cross-Validation Metric Comparison with Error Bars
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    models_list = list(cv_summary.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x_indices = np.arange(len(metrics_to_plot))
    width = 0.25
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, model_name in enumerate(models_list):
        means = [cv_summary[model_name][m]["mean"] for m in metrics_to_plot]
        stds = [cv_summary[model_name][m]["std"] for m in metrics_to_plot]
        ax.bar(
            x_indices + (i - 1) * width, means, yerr=stds, width=width,
            label=model_name, capsize=4, color=colors[i % len(colors)], alpha=0.85
        )
        
    ax.set_xticks(x_indices)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in metrics_to_plot], fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("5-Fold Cross-Validation Performance Comparison (Mean ± Std)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cv_metrics_comparison.png"), dpi=300)
    plt.close()

    # 2. Confusion Matrix Heatmap for Selected Model
    cm = np.array(test_metrics[best_model]["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Stayed", "Predicted Attrited"],
                yticklabels=["Actual Stayed", "Actual Attrited"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Confusion Matrix: {best_model} (Held-out Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 3. ROC Curves Comparison
    plt.figure(figsize=(8, 6))
    for model_name, probs in probs_test.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = test_metrics[model_name]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Chance")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=300)
    plt.close()

    # 4. Precision-Recall Curves Comparison
    plt.figure(figsize=(8, 6))
    for model_name, probs in probs_test.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap_val = test_metrics[model_name]["average_precision"]
        plt.plot(rec, prec, label=f"{model_name} (AP = {ap_val:.3f})", lw=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_curves.png"), dpi=300)
    plt.close()

    # 5. Feature Importances (Random Forest)
    top_rf = rf_imp_df.head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_rf, y="feature", x="importance", hue="feature", legend=False, palette="viridis")
    plt.title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=300)
    plt.close()

    # 6. Logistic Regression Coefficients
    top_pos = lr_coef_df.head(8)
    top_neg = lr_coef_df.tail(8)
    top_coefs = pd.concat([top_pos, top_neg]).sort_values(by="coefficient")
    
    plt.figure(figsize=(10, 7))
    colors_coef = ["#d62728" if c > 0 else "#1f77b4" for c in top_coefs["coefficient"]]
    plt.barh(top_coefs["feature"], top_coefs["coefficient"], color=colors_coef, alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Logistic Regression Feature Coefficients (Positive = Higher Risk)", fontsize=14, fontweight="bold")
    plt.xlabel("Coefficient Value", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "logistic_regression_coefficients.png"), dpi=300)
    plt.close()
    
    print(f"All evaluation plots saved to '{PLOTS_DIR}'.")

if __name__ == "__main__":
    train_and_evaluate_models()
