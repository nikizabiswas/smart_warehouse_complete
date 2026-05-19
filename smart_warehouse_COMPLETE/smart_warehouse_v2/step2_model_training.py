"""
================================================================================
SMART WAREHOUSE DEMAND PREDICTION OF LPG CYLINDERS
Step 2: Model Training  (IMPROVED VERSION)
================================================================================
IMPROVEMENTS OVER v1:
  1. 5-fold cross-validation for every model
  2. AUC-ROC + Average Precision added for classifier evaluation
  3. SMOTE oversampling to handle 10:1 class imbalance
  4. RandomizedSearchCV hyperparameter tuning for best models
  5. Festival one-hot encoding fed as extra features
  6. Full confusion matrix + classification report saved to CSV
  7. Feature importance saved as CSV for app display
  8. Zone-level aggregate demand saved for app
================================================================================
"""

import pandas as pd
import numpy as np
import joblib, os, warnings, json
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model  import LinearRegression, LogisticRegression
from sklearn.tree          import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics       import (r2_score, mean_absolute_error, mean_squared_error,
                                   accuracy_score, precision_score, recall_score,
                                   f1_score, roc_auc_score, average_precision_score,
                                   confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling  import SMOTE

OUTPUT_PATH = "outputs/"
MODEL_PATH  = "models/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH,  exist_ok=True)

THRESHOLD = 0.35   # lower threshold → higher recall for stockout detection

print("=" * 70)
print("  STEP 2: MODEL TRAINING  (cross-validation + SMOTE + tuning)")
print("=" * 70)

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading preprocessed data...")
X_train        = pd.read_csv(OUTPUT_PATH + 'X_train.csv')
X_test         = pd.read_csv(OUTPUT_PATH + 'X_test.csv')
y_train_demand = pd.read_csv(OUTPUT_PATH + 'y_train_demand.csv').squeeze()
y_test_demand  = pd.read_csv(OUTPUT_PATH + 'y_test_demand.csv').squeeze()
y_train_stock  = pd.read_csv(OUTPUT_PATH + 'y_train_stock.csv').squeeze().astype(int)
y_test_stock   = pd.read_csv(OUTPUT_PATH + 'y_test_stock.csv').squeeze().astype(int)
feat_cols      = joblib.load(MODEL_PATH + 'feature_cols.pkl')

print(f"   X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"   Stockout imbalance — No: {(y_train_stock==0).sum()}  Yes: {(y_train_stock==1).sum()}")
print(f"   Stockout rate: {y_train_stock.mean()*100:.1f}% (train)  {y_test_stock.mean()*100:.1f}% (test)")

# ── 2. SMOTE — balance the stockout training data ────────────────────────────
print("\n[2/7] Applying SMOTE to balance stockout classes...")
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train_stock)
print(f"   After SMOTE — No: {(y_train_sm==0).sum()}  Yes: {(y_train_sm==1).sum()}")

# ── 3. REGRESSION — 4 models with 5-fold CV ──────────────────────────────────
print("\n[3/7] Training regression models with 5-fold cross-validation...")
reg_models = {
    'Random Forest'    : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Decision Tree'    : DecisionTreeRegressor(max_depth=5, random_state=42),
    'Linear Regression': LinearRegression(),
}

reg_results   = []
trained_regs  = {}
cv_reg_scores = {}

for name, model in reg_models.items():
    model.fit(X_train, y_train_demand)
    preds = model.predict(X_test)
    r2    = r2_score(y_test_demand, preds)
    mae   = mean_absolute_error(y_test_demand, preds)
    rmse  = np.sqrt(mean_squared_error(y_test_demand, preds))
    # 5-fold CV
    cv_scores = cross_val_score(model, X_train, y_train_demand, cv=5, scoring='r2')
    cv_reg_scores[name] = cv_scores
    reg_results.append({
        'Model': name, 'R2_Score': round(r2,4),
        'MAE': round(mae,4), 'RMSE': round(rmse,4),
        'CV_R2_Mean': round(cv_scores.mean(),4),
        'CV_R2_Std' : round(cv_scores.std(),4),
    })
    trained_regs[name] = model
    print(f"   {name:<25} R2={r2:.4f}  MAE={mae:.4f}  "
          f"CV_R2={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

reg_df      = pd.DataFrame(reg_results).sort_values('R2_Score', ascending=False)
best_reg_nm = reg_df.iloc[0]['Model']
best_reg    = trained_regs[best_reg_nm]
print(f"\n   Best regression model: {best_reg_nm}")

# ── 4. Hyperparameter tuning for best regressor ───────────────────────────────
print(f"\n[4/7] Tuning {best_reg_nm} (RandomizedSearchCV 20 iters, 3-fold)...")
if best_reg_nm == 'Gradient Boosting':
    param_dist = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [3, 4, 5, 6],
        'learning_rate'   : [0.05, 0.1, 0.15, 0.2],
        'min_samples_split': [2, 5, 10],
    }
    base = GradientBoostingRegressor(random_state=42)
elif best_reg_nm == 'Random Forest':
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth'   : [None, 5, 10, 15],
        'max_features': ['sqrt', 'log2', 0.5],
    }
    base = RandomForestRegressor(random_state=42, n_jobs=-1)
else:
    base = best_reg
    param_dist = {}

if param_dist:
    rscv = RandomizedSearchCV(base, param_dist, n_iter=20, cv=3,
                              scoring='r2', random_state=42, n_jobs=-1)
    rscv.fit(X_train, y_train_demand)
    tuned_reg   = rscv.best_estimator_
    tuned_preds = tuned_reg.predict(X_test)
    tuned_r2    = r2_score(y_test_demand, tuned_preds)
    print(f"   Best params: {rscv.best_params_}")
    print(f"   Tuned R2={tuned_r2:.4f}  (was {reg_df.iloc[0]['R2_Score']:.4f})")
    if tuned_r2 > reg_df.iloc[0]['R2_Score']:
        best_reg = tuned_reg
        print("   Using tuned model (improved)")
    else:
        print("   Keeping original (tuned did not improve)")
else:
    tuned_r2 = reg_df.iloc[0]['R2_Score']

# ── 5. CLASSIFICATION — 4 models with SMOTE + 5-fold CV ──────────────────────
print("\n[5/7] Training classification models (SMOTE + 5-fold CV)...")
clf_models = {
    'Random Forest'     : RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                                  random_state=42, n_jobs=-1),
    'Gradient Boosting' : GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Decision Tree'     : DecisionTreeClassifier(max_depth=5, class_weight='balanced',
                                                  random_state=42),
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000,
                                               random_state=42),
}

clf_results   = []
trained_clfs  = {}
skf           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in clf_models.items():
    model.fit(X_train_sm, y_train_sm)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else None
    preds = (probs >= THRESHOLD).astype(int) if probs is not None else model.predict(X_test)

    acc  = accuracy_score(y_test_stock, preds)
    prec = precision_score(y_test_stock, preds, zero_division=0)
    rec  = recall_score(y_test_stock, preds, zero_division=0)
    f1   = f1_score(y_test_stock, preds, zero_division=0)
    auc  = roc_auc_score(y_test_stock, probs) if probs is not None else 0.5
    ap   = average_precision_score(y_test_stock, probs) if probs is not None else 0.0

    cv_f1 = cross_val_score(model, X_train_sm, y_train_sm, cv=5, scoring='f1')

    clf_results.append({
        'Model': name, 'Accuracy': round(acc,4),
        'Precision': round(prec,4), 'Recall': round(rec,4),
        'F1_Score': round(f1,4),
        'AUC_ROC': round(auc,4), 'Avg_Precision': round(ap,4),
        'CV_F1_Mean': round(cv_f1.mean(),4), 'CV_F1_Std': round(cv_f1.std(),4),
    })
    trained_clfs[name] = model
    print(f"   {name:<25} F1={f1:.4f}  Rec={rec:.4f}  "
          f"AUC={auc:.4f}  CV_F1={cv_f1.mean():.4f}±{cv_f1.std():.4f}")

clf_df      = pd.DataFrame(clf_results).sort_values('F1_Score', ascending=False)
best_clf_nm = clf_df.iloc[0]['Model']
best_clf    = trained_clfs[best_clf_nm]
print(f"\n   Best classification model: {best_clf_nm}")

# ── 6. Final metrics + all outputs ───────────────────────────────────────────
print("\n[6/7] Computing final metrics and saving all outputs...")

reg_preds  = best_reg.predict(X_test)
clf_probs  = best_clf.predict_proba(X_test)[:,1]
clf_preds  = (clf_probs >= THRESHOLD).astype(int)

cm = confusion_matrix(y_test_stock, clf_preds)
print(f"\n  Confusion Matrix (threshold={THRESHOLD}):")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print()
print(classification_report(y_test_stock, clf_preds,
                             target_names=['No Stockout','Stockout']))

# Feature importance
feat_imp_df = None
if hasattr(best_reg, 'feature_importances_'):
    feat_imp_df = pd.DataFrame({
        'Feature'   : feat_cols,
        'Importance': best_reg.feature_importances_.round(6),
    }).sort_values('Importance', ascending=False)
    feat_imp_df.to_csv(OUTPUT_PATH + 'feature_importance.csv', index=False)
    print("  Top 5 features:", feat_imp_df['Feature'].head(5).tolist())

# Zone-level aggregate demand from raw train data
raw = pd.read_excel("data/Warehouse_Demand_Realistic_v2.xlsx",
                    sheet_name="Train_80pct", header=1)
for c in ['Actual_Demand (cylinders)','Stockout_Occurred','Month_Number','Year']:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')
zone_agg = raw.groupby(['Warehouse_Zone','Year','Month_Number']).agg(
    Total_Demand =('Actual_Demand (cylinders)','sum'),
    Families     =('Family_ID','count'),
    Stockouts    =('Stockout_Occurred','sum'),
).reset_index()
zone_agg.to_csv(OUTPUT_PATH + 'zone_aggregate_demand.csv', index=False)

# Confusion matrix as CSV
cm_df = pd.DataFrame(
    cm,
    index  =['Actual_No_Stockout','Actual_Stockout'],
    columns=['Pred_No_Stockout',  'Pred_Stockout']
)
cm_df.to_csv(OUTPUT_PATH + 'confusion_matrix.csv')

# Predictions CSV
pred_out = pd.DataFrame({
    'Actual_Demand'    : y_test_demand.values,
    'Predicted_Demand' : np.round(reg_preds, 4),
    'Actual_Stockout'  : y_test_stock.values,
    'Predicted_Stockout': clf_preds,
    'Stockout_Probability': np.round(clf_probs, 4),
})
pred_out.to_csv(OUTPUT_PATH + 'test_predictions.csv', index=False)

# Cross-validation summary CSV
cv_rows = []
for r in reg_results:
    cv_rows.append({'Task':'Regression','Model':r['Model'],
                    'CV_Mean':r['CV_R2_Mean'],'CV_Std':r['CV_R2_Std'],'Metric':'R2'})
for r in clf_results:
    cv_rows.append({'Task':'Classification','Model':r['Model'],
                    'CV_Mean':r['CV_F1_Mean'],'CV_Std':r['CV_F1_Std'],'Metric':'F1'})
pd.DataFrame(cv_rows).to_csv(OUTPUT_PATH + 'cross_validation_results.csv', index=False)

# Model comparison CSVs
reg_df.to_csv(OUTPUT_PATH + 'regression_model_comparison.csv',    index=False)
clf_df.to_csv(OUTPUT_PATH + 'classification_model_comparison.csv', index=False)

# ── 7. Save models + model_info ──────────────────────────────────────────────
print("[7/7] Saving models...")
joblib.dump(best_reg, MODEL_PATH + 'best_demand_model.pkl')
joblib.dump(best_clf, MODEL_PATH + 'best_stockout_model.pkl')

final_reg_r2   = r2_score(y_test_demand, reg_preds)
final_reg_mae  = mean_absolute_error(y_test_demand, reg_preds)
final_reg_rmse = np.sqrt(mean_squared_error(y_test_demand, reg_preds))

model_info = {
    'best_regression_model'    : best_reg_nm,
    'regression_r2'            : round(final_reg_r2,  4),
    'regression_mae'           : round(final_reg_mae,  4),
    'regression_rmse'          : round(final_reg_rmse, 4),
    'regression_cv_r2_mean'    : round(float(np.mean([r['CV_R2_Mean'] for r in reg_results if r['Model']==best_reg_nm])), 4),
    'regression_cv_r2_std'     : round(float(np.mean([r['CV_R2_Std']  for r in reg_results if r['Model']==best_reg_nm])), 4),
    'best_classification_model': best_clf_nm,
    'classification_accuracy'  : round(float(accuracy_score(y_test_stock, clf_preds)), 4),
    'classification_precision' : round(float(precision_score(y_test_stock, clf_preds, zero_division=0)), 4),
    'classification_recall'    : round(float(recall_score(y_test_stock, clf_preds, zero_division=0)), 4),
    'classification_f1'        : round(float(f1_score(y_test_stock, clf_preds, zero_division=0)), 4),
    'classification_auc_roc'   : round(float(roc_auc_score(y_test_stock, clf_probs)), 4),
    'classification_avg_prec'  : round(float(average_precision_score(y_test_stock, clf_probs)), 4),
    'clf_cv_f1_mean'           : round(float(np.mean([r['CV_F1_Mean'] for r in clf_results if r['Model']==best_clf_nm])), 4),
    'clf_cv_f1_std'            : round(float(np.mean([r['CV_F1_Std']  for r in clf_results if r['Model']==best_clf_nm])), 4),
    'stockout_threshold'       : THRESHOLD,
    'class_imbalance_method'   : 'SMOTE + class_weight=balanced',
    'confusion_matrix'         : cm.tolist(),
}
joblib.dump(model_info, MODEL_PATH + 'model_info.pkl')

print("\n" + "=" * 70)
print("  TRAINING COMPLETE!")
print(f"\n  REGRESSION  (best: {best_reg_nm})")
for r in reg_df.to_dict('records'):
    m = " ← BEST" if r['Model']==best_reg_nm else ""
    print(f"    {r['Model']:<25} R2={r['R2_Score']:.4f}  CV={r['CV_R2_Mean']:.4f}±{r['CV_R2_Std']:.4f}{m}")
print(f"\n  CLASSIFICATION  (best: {best_clf_nm}  threshold={THRESHOLD})")
for r in clf_df.to_dict('records'):
    m = " ← BEST" if r['Model']==best_clf_nm else ""
    print(f"    {r['Model']:<25} F1={r['F1_Score']:.4f}  Rec={r['Recall']:.4f}  AUC={r['AUC_ROC']:.4f}  CV={r['CV_F1_Mean']:.4f}±{r['CV_F1_Std']:.4f}{m}")
print("\n  New output files:")
print("    outputs/feature_importance.csv")
print("    outputs/cross_validation_results.csv")
print("    outputs/confusion_matrix.csv")
print("    outputs/zone_aggregate_demand.csv")
print("=" * 70)
print("  Next: Run step3_visualization.py")
print("=" * 70)
