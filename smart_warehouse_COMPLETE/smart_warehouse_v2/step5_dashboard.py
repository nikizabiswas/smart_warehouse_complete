"""
================================================================================
SMART WAREHOUSE DEMAND PREDICTION OF LPG CYLINDERS
Step 5: Dashboard Report
================================================================================
WHAT THIS FILE DOES:
- Builds a master dashboard image combining all key panels
- Prints a complete performance summary table
- Exports a final Excel report with model metrics, predictions, and future forecasts

UPDATED FOR v2 DATASET:
- Reads Warehouse_Demand_Realistic_v2.xlsx (Train_80pct sheet, header=1)
- Stockout Zone chart pulls live zone data from the raw training sheet
- Zone_Summary sheet used for zone-level stats panel
- Dashboard footer reflects v2 dataset stats (5000 records, 3 years)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH = "outputs/"
MODEL_PATH  = "models/"
CHART_PATH  = "charts/"
os.makedirs(CHART_PATH, exist_ok=True)

print("=" * 65)
print("  STEP 5: DASHBOARD REPORT GENERATOR  (v2 Dataset)")
print("=" * 65)

# -- Load all saved artefacts -------------------------------------------------
model_info = joblib.load(MODEL_PATH + 'model_info.pkl')
reg_df     = pd.read_csv(OUTPUT_PATH + 'regression_model_comparison.csv')
clf_df     = pd.read_csv(OUTPUT_PATH + 'classification_model_comparison.csv')
pred_df    = pd.read_csv(OUTPUT_PATH + 'test_predictions.csv')

# Load raw v2 training data for zone-level stockout chart
raw_train = pd.read_excel(
    "data/Warehouse_Demand_Realistic_v2.xlsx",
    sheet_name="Train_80pct",
    header=1
)
raw_train['Stockout_Occurred'] = pd.to_numeric(raw_train['Stockout_Occurred'], errors='coerce')

# Load Zone_Summary for the stats panel
zone_summary = pd.read_excel(
    "data/Warehouse_Demand_Realistic_v2.xlsx",
    sheet_name="Zone_Summary",
    header=0
)

COLORS = ['#1A5276', '#1E8449', '#B7950B', '#C0392B', '#7D3C98']

# =============================================================================
# BUILD MASTER DASHBOARD
# =============================================================================
print("\n[1/2] Building master dashboard...")

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor('#F0F3F4')

gs = gridspec.GridSpec(5, 3, figure=fig,
                       hspace=0.50, wspace=0.35,
                       top=0.93, bottom=0.03,
                       left=0.06, right=0.97)

# -- Title Banner -------------------------------------------------------------
ax_title = fig.add_axes([0, 0.94, 1, 0.06])
ax_title.set_facecolor('#1F3864')
ax_title.axis('off')
ax_title.text(0.5, 0.68,
              'Smart Warehouse Demand Prediction — LPG Cylinders',
              ha='center', va='center', color='white',
              fontsize=20, fontweight='bold')
ax_title.text(0.5, 0.22,
              'Mitra Bharatgas Agency  |  Murshidabad, West Bengal  |  '
              'Dataset v2  |  ML-Based Approach',
              ha='center', va='center', color='#AED6F1', fontsize=11)

# -- Panel 1: Regression R2 Comparison ----------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('white')
bars = ax1.barh(reg_df['Model'], reg_df['R2_Score'],
                color=COLORS[:len(reg_df)], height=0.5)
ax1.set_xlim(0, 1.12)
ax1.set_title('Regression Model R2 Comparison', fontweight='bold', fontsize=11)
ax1.set_xlabel('R2 Score')
for bar, val in zip(bars, reg_df['R2_Score']):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=8, fontweight='bold')
ax1.axvline(0.9, color='red', linestyle='--', alpha=0.5, linewidth=1,
            label='Target 0.90')
ax1.legend(fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# -- Panel 2: Classification F1 Comparison ------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('white')
bars2 = ax2.barh(clf_df['Model'], clf_df['F1_Score'],
                 color=COLORS[:len(clf_df)], height=0.5)
ax2.set_xlim(0, 1.12)
ax2.set_title('Classification Model F1 Comparison', fontweight='bold', fontsize=11)
ax2.set_xlabel('F1 Score')
for bar, val in zip(bars2, clf_df['F1_Score']):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=8, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# -- Panel 3: KPI Summary Card ------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#1F3864')
ax3.axis('off')
kpis = [
    ('Best Demand Model',   model_info['best_regression_model'].split()[0]),
    ('R2 Score',            f"{model_info['regression_r2']:.4f}"),
    ('MAE',                 f"{model_info['regression_mae']:.4f} cyl"),
    ('RMSE',                f"{model_info['regression_rmse']:.4f} cyl"),
    ('Best Stockout Model', model_info['best_classification_model'].split()[0]),
    ('Accuracy',            f"{model_info['classification_accuracy']*100:.2f}%"),
    ('F1 Score',            f"{model_info['classification_f1']:.4f}"),
]
ax3.text(0.5, 0.97, 'Model Performance Summary',
         ha='center', va='top', color='white',
         fontsize=11, fontweight='bold', transform=ax3.transAxes)
for i, (lbl, val) in enumerate(kpis):
    y = 0.85 - i * 0.11
    ax3.text(0.05, y, lbl + ':', color='#AED6F1', fontsize=9, transform=ax3.transAxes)
    ax3.text(0.95, y, val, color='white', fontsize=9,
             fontweight='bold', ha='right', transform=ax3.transAxes)

# -- Panel 4: Actual vs Predicted Demand (wide) --------------------------------
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.set_facecolor('white')
ax4.scatter(pred_df['Actual_Demand'], pred_df['Predicted_Demand'],
            alpha=0.35, color='#1A5276', s=12)
lims = [pred_df['Actual_Demand'].min() - 0.1, pred_df['Actual_Demand'].max() + 0.1]
ax4.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
ax4.set_xlabel('Actual Demand (cylinders)', fontsize=10)
ax4.set_ylabel('Predicted Demand (cylinders)', fontsize=10)
ax4.set_title('Actual vs Predicted Demand (Test Set)', fontweight='bold', fontsize=11)
ax4.legend(fontsize=9)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# -- Panel 5: Error Distribution -----------------------------------------------
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('white')
errors = pred_df['Actual_Demand'] - pred_df['Predicted_Demand']
ax5.hist(errors, bins=35, color='#1A5276', alpha=0.78, edgecolor='white')
ax5.axvline(0, color='red', linestyle='--', linewidth=1.3, label='Zero Error')
ax5.axvline(errors.mean(), color='orange', linestyle='--', linewidth=1.3,
            label=f'Mean: {errors.mean():.3f}')
ax5.set_xlabel('Error (Actual - Predicted)', fontsize=9)
ax5.set_title('Prediction Error Distribution', fontweight='bold', fontsize=11)
ax5.legend(fontsize=8)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# -- Panel 6: Confusion Matrix -------------------------------------------------
ax6 = fig.add_subplot(gs[2, 0])
ax6.set_facecolor('white')
cm = confusion_matrix(pred_df['Actual_Stockout'], pred_df['Predicted_Stockout'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
            xticklabels=['No Stockout', 'Stockout'],
            yticklabels=['No Stockout', 'Stockout'],
            annot_kws={'size': 13, 'weight': 'bold'}, cbar=False)
ax6.set_title('Stockout Confusion Matrix', fontweight='bold', fontsize=11)
ax6.set_xlabel('Predicted')
ax6.set_ylabel('Actual')

# -- Panel 7: Stockout Rate by Zone (from v2 raw training data) ----------------
ax7 = fig.add_subplot(gs[2, 1:3])
ax7.set_facecolor('white')
zone_so = (raw_train.groupby('Warehouse_Zone')['Stockout_Occurred']
           .mean()
           .sort_values(ascending=False) * 100)
bar_colors_zone = ['#C0392B' if v > zone_so.mean() else '#1E8449'
                   for v in zone_so.values]
bars7 = ax7.bar(zone_so.index, zone_so.values,
                color=bar_colors_zone, edgecolor='white', width=0.6)
ax7.axhline(zone_so.mean(), color='orange', linestyle='--', linewidth=1.5,
            label=f'Avg: {zone_so.mean():.1f}%')
ax7.set_ylabel('Stockout Rate (%)', fontsize=10)
ax7.set_title('Stockout Rate by Warehouse Zone (v2 Training Data)\n'
              'Red = above average risk', fontweight='bold', fontsize=11)
ax7.set_xticklabels(zone_so.index, rotation=28, ha='right', fontsize=9)
ax7.legend(fontsize=9)
for bar, val in zip(bars7, zone_so.values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val:.1f}%', ha='center', va='bottom',
             fontsize=8, fontweight='bold')
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)

# -- Panel 8: Future Predictions (from step4 output) --------------------------
ax8 = fig.add_subplot(gs[3, :])
ax8.set_facecolor('white')
try:
    fut = pd.read_csv(OUTPUT_PATH + 'future_predictions.csv')
    short_labels = []
    for s in fut['Scenario']:
        parts = s.split('|')
        short_labels.append(parts[0].strip() + '\n' + parts[1].strip()
                            if len(parts) > 1 else s[:22])
    risk_colors = ['#C0392B' if 'HIGH' in r
                   else '#B7950B' if 'MEDIUM' in r
                   else '#1E8449'
                   for r in fut['Risk_Level']]
    bars8 = ax8.bar(short_labels, fut['Predicted_Demand'],
                    color=risk_colors, edgecolor='white', width=0.55)
    ax8.set_ylabel('Predicted Demand (cylinders)', fontsize=10)
    ax8.set_title('Future Demand Predictions by Scenario (2025)',
                  fontweight='bold', fontsize=12)
    for bar, val, order in zip(bars8, fut['Predicted_Demand'],
                               fut['Recommended_Order_Qty']):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}\nOrder:{order}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax8.legend(handles=[
        Patch(color='#C0392B', label='High Risk'),
        Patch(color='#B7950B', label='Medium Risk'),
        Patch(color='#1E8449', label='Low Risk'),
    ], fontsize=9)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
except FileNotFoundError:
    ax8.text(0.5, 0.5,
             'Run step4_predict_new.py first\nto generate future predictions',
             ha='center', va='center', transform=ax8.transAxes,
             fontsize=12, color='gray')
    ax8.axis('off')

# -- Panel 9: Summary Footer --------------------------------------------------
ax9 = fig.add_subplot(gs[4, :])
ax9.set_facecolor('#EAF2FB')
ax9.axis('off')
summary = (
    f"PROJECT SUMMARY  |  Smart Warehouse Demand Prediction (LPG Cylinders)  |  "
    f"Mitra Bharatgas Agency, Murshidabad, West Bengal\n"
    f"Dataset v2: 5,000 records  |  10 Warehouse Zones  |  Jan 2022 - Dec 2024  |  "
    f"80% Train / 20% Test  |  18 Feature Columns\n"
    f"Demand Model : {model_info['best_regression_model']}  ->  "
    f"R2={model_info['regression_r2']:.4f},  "
    f"MAE={model_info['regression_mae']:.4f},  "
    f"RMSE={model_info['regression_rmse']:.4f}     |     "
    f"Stockout Model : {model_info['best_classification_model']}  ->  "
    f"Accuracy={model_info['classification_accuracy']*100:.2f}%,  "
    f"F1={model_info['classification_f1']:.4f}"
)
ax9.text(0.5, 0.5, summary,
         ha='center', va='center', transform=ax9.transAxes,
         fontsize=9, color='#1F3864', linespacing=1.9, fontweight='bold')

plt.savefig(CHART_PATH + 'dashboard_master.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   Saved: charts/dashboard_master.png")

# =============================================================================
# EXPORT EXCEL REPORT
# =============================================================================
print("\n[2/2] Exporting final Excel report...")

excel_path = OUTPUT_PATH + 'smart_warehouse_v2_report.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # Sheet 1: Model comparison (regression)
    reg_df.to_excel(writer, sheet_name='Regression_Models', index=False)

    # Sheet 2: Model comparison (classification)
    clf_df.to_excel(writer, sheet_name='Classification_Models', index=False)

    # Sheet 3: Test set predictions
    pred_df.to_excel(writer, sheet_name='Test_Predictions', index=False)

    # Sheet 4: Future predictions (if available)
    try:
        fut = pd.read_csv(OUTPUT_PATH + 'future_predictions.csv')
        fut.to_excel(writer, sheet_name='Future_Predictions', index=False)
    except FileNotFoundError:
        pass

    # Sheet 5: Zone summary (from v2 dataset)
    zone_summary.to_excel(writer, sheet_name='Zone_Summary', index=False)

    # Sheet 6: KPI summary
    kpi_data = {
        'Metric': [
            'Best Demand Model', 'Regression R2', 'Regression MAE', 'Regression RMSE',
            'Best Stockout Model', 'Classification Accuracy', 'Classification F1',
            'Training Records', 'Test Records', 'Feature Columns', 'Warehouse Zones',
        ],
        'Value': [
            model_info['best_regression_model'],
            round(model_info['regression_r2'], 4),
            round(model_info['regression_mae'], 4),
            round(model_info['regression_rmse'], 4),
            model_info['best_classification_model'],
            f"{model_info['classification_accuracy']*100:.2f}%",
            round(model_info['classification_f1'], 4),
            4000, 1000, 18, 10,
        ]
    }
    pd.DataFrame(kpi_data).to_excel(writer, sheet_name='KPI_Summary', index=False)

print(f"   Saved: {excel_path}")

# =============================================================================
# PRINT FINAL CONSOLE SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  SMART WAREHOUSE v2 — FINAL RESULTS")
print("=" * 65)
print(f"\n  DEMAND FORECASTING (Regression)")
print(f"  {'-'*42}")
for _, row in reg_df.iterrows():
    marker = " <- BEST" if row['Model'] == model_info['best_regression_model'] else ""
    print(f"  {row['Model']:<25}  R2={row['R2_Score']:.4f}  "
          f"MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}{marker}")

print(f"\n  STOCKOUT PREDICTION (Classification)")
print(f"  {'-'*42}")
for _, row in clf_df.iterrows():
    marker = " <- BEST" if row['Model'] == model_info['best_classification_model'] else ""
    print(f"  {row['Model']:<25}  F1={row['F1_Score']:.4f}  "
          f"Acc={row['Accuracy']:.4f}{marker}")

print("\n" + "=" * 65)
print("  ALL STEPS COMPLETE!  Output files:")
print("  models/                              -> trained .pkl model files")
print("  outputs/X_train.csv, X_test.csv      -> scaled feature matrices")
print("  outputs/test_predictions.csv         -> model predictions on test set")
print("  outputs/future_predictions.csv       -> 2025 scenario forecasts")
print("  outputs/smart_warehouse_v2_report.xlsx -> full Excel report")
print("  charts/dashboard_master.png          -> master dashboard image")
print("  charts/chart1-8_*.png                -> individual charts")
print("=" * 65)
