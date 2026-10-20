import os
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

def run_pipeline():
    base_dir = r"C:\Users\PUNIT\Desktop\DAU projects\campus-placement-eda-ml"
    data_raw_path = os.path.join(base_dir, "data", "raw", "student_placement_data.csv")
    data_proc_path = os.path.join(base_dir, "data", "processed", "placement_data_encoded.csv")
    models_dir = os.path.join(base_dir, "models")
    figures_dir = os.path.join(base_dir, "reports", "figures")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("1. Loading dataset...")
    if os.path.exists(data_proc_path):
        df = pd.read_csv(data_proc_path)
        print("Loaded processed dataset.")
    else:
        df = pd.read_csv(data_raw_path)
        drop_cols = ['student_id', 'salary_lpa']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        if 'branch' in df.columns:
            df = pd.get_dummies(df, columns=['branch'], drop_first=True)
        print(f"Loaded raw dataset with standard auto-prep: {df.shape}")
        
    X = df.drop(columns=['placement_status'])
    y = df['placement_status']
    
    # 2. Train-Test Split & Scaling
    print("2. Performing 80/20 Stratified Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # 3. Model Training
    print("3. Training 4 Supervised Classifiers...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42)
    }
    
    results = []
    trained_models = {}
    preds_dict = {}
    probs_dict = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        trained_models[name] = model
        preds_dict[name] = y_pred
        probs_dict[name] = y_proba
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        })
        
    df_results = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(df_results.to_string(index=False))
    
    # 4. Save best model (Random Forest)
    best_model_path = os.path.join(models_dir, "placement_classifier.joblib")
    joblib.dump(trained_models['Random Forest'], best_model_path)
    print(f"\nBest Model (Random Forest) exported to {best_model_path}")
    
    # 5. Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot 1: Comparative ROC Curves
    print("5. Generating Comparative ROC Curves...")
    plt.figure(figsize=(9, 7))
    for name, prob in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2.2)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Comparative ROC Curves — Campus Placement Classifiers', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    roc_fig_path = os.path.join(figures_dir, "roc_curves_comparison.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to {roc_fig_path}")
    
    # Plot 2: Confusion Matrices
    print("6. Generating Comparative Confusion Matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (name, pred) in enumerate(preds_dict.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                    xticklabels=['Unplaced (0)', 'Placed (1)'],
                    yticklabels=['Unplaced (0)', 'Placed (1)'])
        axes[idx].set_title(f"{name}\nAcc: {results[idx]['Accuracy']} | F1: {results[idx]['F1-Score']}", fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
    plt.suptitle('Confusion Matrices Comparison (Test Set: 20,000 Students)', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    cm_fig_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"Confusion matrices saved to {cm_fig_path}")
    
    # Plot 3: Feature Importance
    print("7. Generating Feature Importance Plot...")
    rf = trained_models['Random Forest']
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6.5))
    palette = sns.color_palette("viridis", len(feat_imp))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette=palette, legend=False)
    plt.title('Random Forest Feature Importance: Key Determinants of Campus Placement', fontsize=13, fontweight='bold')
    plt.xlabel('Gini Importance (Relative Predictive Power)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    for i, v in enumerate(feat_imp.values):
        plt.text(v + 0.003, i, f"{v*100:.1f}%", va='center', fontsize=9.5, fontweight='bold', color='#333333')
    plt.xlim(0, max(feat_imp.values) * 1.15)
    plt.tight_layout()
    fi_fig_path = os.path.join(figures_dir, "feature_importance.png")
    plt.savefig(fi_fig_path, dpi=300)
    plt.close()
    print(f"Feature importance saved to {fi_fig_path}")
    
    # 8. Inference Demonstration on 3 Archetypes
    print("\n8. Sample Inference Demonstration (Testing Best Model):")
    sample_students = pd.DataFrame([
        # Archetype 1: Stellar academic + strong coder
        {'cpi': 8.8, 'attendance_rate': 92.0, 'backlogs': 0, 'coding_score': 8.5, 'dsa_score': 8.0,
         'dsa_problems_solved': 450, 'projects_count': 4, 'has_major_project': 1, 'internships': 2,
         'experience_months': 8, 'aptitude_score': 88.0, 'communication_skills': 8.5,
         'hackathons_attended': 2, 'certifications_count': 3,
         'branch_Civil': False, 'branch_ECE': False, 'branch_EE': False, 'branch_ICT': False,
         'branch_IT': False, 'branch_Mechanical': False, 'branch_MnC': False},
        # Archetype 2: Moderate CPI (6.8) but high coding (8.8) and projects
        {'cpi': 6.8, 'attendance_rate': 78.0, 'backlogs': 0, 'coding_score': 8.8, 'dsa_score': 8.5,
         'dsa_problems_solved': 520, 'projects_count': 5, 'has_major_project': 1, 'internships': 1,
         'experience_months': 3, 'aptitude_score': 74.0, 'communication_skills': 7.0,
         'hackathons_attended': 3, 'certifications_count': 2,
         'branch_Civil': False, 'branch_ECE': False, 'branch_EE': False, 'branch_ICT': True,
         'branch_IT': False, 'branch_Mechanical': False, 'branch_MnC': False},
        # Archetype 3: Low CPI (5.2), 3 backlogs, weak coding (2.5)
        {'cpi': 5.2, 'attendance_rate': 64.0, 'backlogs': 3, 'coding_score': 2.5, 'dsa_score': 2.0,
         'dsa_problems_solved': 25, 'projects_count': 0, 'has_major_project': 0, 'internships': 0,
         'experience_months': 0, 'aptitude_score': 42.0, 'communication_skills': 4.0,
         'hackathons_attended': 0, 'certifications_count': 0,
         'branch_Civil': False, 'branch_ECE': False, 'branch_EE': True, 'branch_ICT': False,
         'branch_IT': False, 'branch_Mechanical': False, 'branch_MnC': False}
    ])
    
    # Align columns
    for col in X.columns:
        if col not in sample_students.columns:
            sample_students[col] = 0
    sample_students = sample_students[X.columns]
    
    sample_scaled = scaler.transform(sample_students)
    rf_best = trained_models['Random Forest']
    sample_preds = rf_best.predict(sample_scaled)
    sample_probs = rf_best.predict_proba(sample_scaled)[:, 1]
    
    labels = ["High CPI + High Coder (Ideal)", "Mod CPI (6.8) + Elite Coder (Comp-Prog)", "Low CPI (5.2) + 3 Backlogs (At-Risk)"]
    for i, label in enumerate(labels):
        status = "PLACED [PASS]" if sample_preds[i] == 1 else "UNPLACED [FAIL]"
        print(f"   Student {i+1} [{label}]: {status} (Placement Probability: {sample_probs[i]*100:.2f}%)")
        
    print("\nStage 3 & 4 Pipeline completed successfully!")
    return df_results, feat_imp

if __name__ == '__main__':
    run_pipeline()
