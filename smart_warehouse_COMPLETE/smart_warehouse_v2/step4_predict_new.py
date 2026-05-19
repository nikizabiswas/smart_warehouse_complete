"""
================================================================================
SMART WAREHOUSE DEMAND PREDICTION OF LPG CYLINDERS
Step 4: Predict on New / Future Data
================================================================================
WHAT THIS FILE DOES:
- Loads the trained models from Step 2
- Predicts demand and stockout risk for 6 future scenarios across different
  warehouse zones and seasons (updated for v2 dataset zones/prices)
- Outputs: predicted demand, stockout probability, recommended order quantity

HOW TO USE:
  Edit the FUTURE_SCENARIOS list below with your actual planning data,
  then run: python step4_predict_new.py

ZONE ENCODING (all 10 zones in v2 dataset):
  Bali=0, Berhampore=1, Domkal=2, Farakka=3, Jangipur=4,
  Kandi=5, Lalbagh=6, Raghunathganj=7, Samserganj=8, Suti=9

SUBSIDY ENCODING:
  Non-Subsidized=0, PMUY=1

UPDATED FOR v2 DATASET:
  - LPG prices reflect v2 realistic range (Rs 876 - Rs 940)
  - 6 diverse scenarios covering all key season/festival combinations
  - Avg_Daily_Demand excluded from features (leakage column)
================================================================================
"""

import pandas as pd
import numpy as np
import joblib, warnings
warnings.filterwarnings('ignore')

MODEL_PATH  = "models/"
OUTPUT_PATH = "outputs/"

print("=" * 65)
print("  STEP 4: PREDICT FUTURE DEMAND  (v2 Dataset)")
print("=" * 65)

# -- Load saved models and preprocessing objects ------------------------------
print("\n[1/3] Loading trained models...")
best_reg     = joblib.load(MODEL_PATH + 'best_demand_model.pkl')
best_clf     = joblib.load(MODEL_PATH + 'best_stockout_model.pkl')
scaler       = joblib.load(MODEL_PATH + 'scaler.pkl')
encoders     = joblib.load(MODEL_PATH + 'label_encoders.pkl')
feature_cols = joblib.load(MODEL_PATH + 'feature_cols.pkl')
model_info   = joblib.load(MODEL_PATH + 'model_info.pkl')

print(f"   Demand Model  : {model_info['best_regression_model']}")
print(f"   Stockout Model: {model_info['best_classification_model']}")
print(f"   Features used : {feature_cols}")

# -----------------------------------------------------------------------------
# DEFINE FUTURE SCENARIOS
# Each dict = one zone x one month prediction scenario.
#
# Feature columns (18 total, in exact order from training):
#   Warehouse_Zone, Year, Month_Number, Is_Winter, Is_Festival_Month,
#   Days_in_Month, No_of_Family_Members, No_of_Adults, No_of_Children,
#   Monthly_Income (Rs), Subsidy_Type, LPG_Price_per_Cylinder (Rs),
#   Zone_Opening_Stock, Zone_Cylinders_Ordered, Lead_Time_Days,
#   Zone_Cylinders_Delivered, Damaged_Cylinders, Zone_Closing_Stock
# -----------------------------------------------------------------------------
FUTURE_SCENARIOS = [
    {
        # Berhampore -- January 2025: Winter + Makar Sankranti festival
        # High demand expected; longer lead time (6.64 avg per Zone_Summary)
        'label'                      : 'Berhampore | Jan 2025 (Winter + Festival)',
        'Warehouse_Zone'             : 1,    # Berhampore
        'Year'                       : 2025,
        'Month_Number'               : 1,
        'Is_Winter'                  : 1,
        'Is_Festival_Month'          : 1,
        'Days_in_Month'              : 31,
        'No_of_Family_Members'       : 4,
        'No_of_Adults'               : 2,
        'No_of_Children'             : 2,
        'Monthly_Income (Rs)'        : 23649,   # zone avg income
        'Subsidy_Type'               : 1,       # PMUY
        'LPG_Price_per_Cylinder (Rs)': 921,
        'Zone_Opening_Stock'         : 550,
        'Zone_Cylinders_Ordered'     : 480,
        'Lead_Time_Days'             : 7,
        'Zone_Cylinders_Delivered'   : 450,
        'Damaged_Cylinders'          : 3,
        'Zone_Closing_Stock'         : 610,
    },
    {
        # Farakka -- June 2025: Monsoon (lower demand, longer lead time)
        'label'                      : 'Farakka | Jun 2025 (Monsoon)',
        'Warehouse_Zone'             : 3,    # Farakka
        'Year'                       : 2025,
        'Month_Number'               : 6,
        'Is_Winter'                  : 0,
        'Is_Festival_Month'          : 0,
        'Days_in_Month'              : 30,
        'No_of_Family_Members'       : 4,
        'No_of_Adults'               : 2,
        'No_of_Children'             : 2,
        'Monthly_Income (Rs)'        : 23477,
        'Subsidy_Type'               : 1,
        'LPG_Price_per_Cylinder (Rs)': 891,
        'Zone_Opening_Stock'         : 580,
        'Zone_Cylinders_Ordered'     : 320,
        'Lead_Time_Days'             : 7,
        'Zone_Cylinders_Delivered'   : 300,
        'Damaged_Cylinders'          : 1,
        'Zone_Closing_Stock'         : 625,
    },
    {
        # Kandi -- October 2025: Durga Puja (festival peak, shortest lead time)
        'label'                      : 'Kandi | Oct 2025 (Durga Puja Peak)',
        'Warehouse_Zone'             : 5,    # Kandi
        'Year'                       : 2025,
        'Month_Number'               : 10,
        'Is_Winter'                  : 0,
        'Is_Festival_Month'          : 1,
        'Days_in_Month'              : 31,
        'No_of_Family_Members'       : 5,
        'No_of_Adults'               : 3,
        'No_of_Children'             : 2,
        'Monthly_Income (Rs)'        : 23039,
        'Subsidy_Type'               : 0,   # Non-Subsidized
        'LPG_Price_per_Cylinder (Rs)': 915,
        'Zone_Opening_Stock'         : 820,
        'Zone_Cylinders_Ordered'     : 460,
        'Lead_Time_Days'             : 3,
        'Zone_Cylinders_Delivered'   : 440,
        'Damaged_Cylinders'          : 2,
        'Zone_Closing_Stock'         : 849,
    },
    {
        # Raghunathganj -- December 2025: Peak Winter
        'label'                      : 'Raghunathganj | Dec 2025 (Peak Winter)',
        'Warehouse_Zone'             : 7,    # Raghunathganj
        'Year'                       : 2025,
        'Month_Number'               : 12,
        'Is_Winter'                  : 1,
        'Is_Festival_Month'          : 0,
        'Days_in_Month'              : 31,
        'No_of_Family_Members'       : 4,
        'No_of_Adults'               : 2,
        'No_of_Children'             : 2,
        'Monthly_Income (Rs)'        : 23234,
        'Subsidy_Type'               : 1,
        'LPG_Price_per_Cylinder (Rs)': 917,
        'Zone_Opening_Stock'         : 780,
        'Zone_Cylinders_Ordered'     : 500,
        'Lead_Time_Days'             : 6,
        'Zone_Cylinders_Delivered'   : 470,
        'Damaged_Cylinders'          : 2,
        'Zone_Closing_Stock'         : 810,
    },
    {
        # Samserganj -- March 2025: Summer (moderate demand, low lead time)
        'label'                      : 'Samserganj | Mar 2025 (Summer)',
        'Warehouse_Zone'             : 8,    # Samserganj
        'Year'                       : 2025,
        'Month_Number'               : 3,
        'Is_Winter'                  : 0,
        'Is_Festival_Month'          : 0,
        'Days_in_Month'              : 31,
        'No_of_Family_Members'       : 4,
        'No_of_Adults'               : 2,
        'No_of_Children'             : 2,
        'Monthly_Income (Rs)'        : 22520,
        'Subsidy_Type'               : 1,
        'LPG_Price_per_Cylinder (Rs)': 903,
        'Zone_Opening_Stock'         : 940,
        'Zone_Cylinders_Ordered'     : 420,
        'Lead_Time_Days'             : 4,
        'Zone_Cylinders_Delivered'   : 400,
        'Damaged_Cylinders'          : 1,
        'Zone_Closing_Stock'         : 980,
    },
    {
        # Suti -- November 2025: Diwali/Kali Puja festival
        'label'                      : 'Suti | Nov 2025 (Diwali/Kali Puja)',
        'Warehouse_Zone'             : 9,    # Suti
        'Year'                       : 2025,
        'Month_Number'               : 11,
        'Is_Winter'                  : 0,
        'Is_Festival_Month'          : 1,
        'Days_in_Month'              : 30,
        'No_of_Family_Members'       : 5,
        'No_of_Adults'               : 3,
        'No_of_Children'             : 2,
        'Monthly_Income (Rs)'        : 24167,
        'Subsidy_Type'               : 0,
        'LPG_Price_per_Cylinder (Rs)': 910,
        'Zone_Opening_Stock'         : 700,
        'Zone_Cylinders_Ordered'     : 490,
        'Lead_Time_Days'             : 5,
        'Zone_Cylinders_Delivered'   : 460,
        'Damaged_Cylinders'          : 3,
        'Zone_Closing_Stock'         : 732,
    },
]

# Build column name mapping: scenarios use 'Monthly_Income (Rs)' (no rupee symbol)
# but the actual feature column name uses the Unicode rupee sign.
# We auto-map to match whatever feature_cols says.
RUPEE_MAP = {
    'Monthly_Income (Rs)'        : 'Monthly_Income (\u20b9)',
    'LPG_Price_per_Cylinder (Rs)': 'LPG_Price_per_Cylinder (\u20b9)',
}

# -----------------------------------------------------------------------------
# RUN PREDICTIONS
# -----------------------------------------------------------------------------
print("\n[2/3] Running predictions...\n")
print("-" * 65)

results = []

for scenario in FUTURE_SCENARIOS:
    label = scenario.pop('label')

    # Remap plain 'Rs' keys to Unicode rupee symbol if needed
    for plain, fancy in RUPEE_MAP.items():
        if plain in scenario and fancy in feature_cols:
            scenario[fancy] = scenario.pop(plain)

    # Build input DataFrame in exact feature column order
    input_df = pd.DataFrame([{col: scenario.get(col, 0) for col in feature_cols}])

    # Scale using the saved scaler (fitted on training data)
    input_scaled = scaler.transform(input_df)

    # Predict demand (regression)
    demand_pred = best_reg.predict(input_scaled)[0]
    demand_pred = max(1, round(demand_pred, 2))

    # Predict stockout (classification)
    stockout_pred = best_clf.predict(input_scaled)[0]
    stockout_prob = best_clf.predict_proba(input_scaled)[0][1] * 100

    # Warehouse recommendations
    safety_stock      = round(demand_pred * 0.15)
    reorder_point     = round(demand_pred * 0.25)
    recommended_order = round(demand_pred * 1.10 + safety_stock)

    if stockout_prob > 60:
        risk_label = "HIGH RISK"
        risk_icon  = "[HIGH]"
    elif stockout_prob > 30:
        risk_label = "MEDIUM RISK"
        risk_icon  = "[MED] "
    else:
        risk_label = "LOW RISK"
        risk_icon  = "[LOW] "

    print(f"  Scenario : {label}")
    print(f"  +------------------------------------------+")
    print(f"  | Predicted Demand     : {demand_pred:>6.2f} cylinders  |")
    print(f"  | Stockout Probability : {stockout_prob:>6.1f}%             |")
    print(f"  | Risk Level           : {risk_icon} {risk_label:<14}|")
    print(f"  | Recommended Order    : {recommended_order:>6} cylinders  |")
    print(f"  | Safety Stock Level   : {safety_stock:>6} cylinders  |")
    print(f"  | Reorder Point        : {reorder_point:>6} cylinders  |")
    print(f"  +------------------------------------------+")
    print()

    results.append({
        'Scenario'              : label,
        'Predicted_Demand'      : demand_pred,
        'Stockout_Probability_%': round(stockout_prob, 2),
        'Risk_Level'            : risk_label,
        'Recommended_Order_Qty' : recommended_order,
        'Safety_Stock'          : safety_stock,
        'Reorder_Point'         : reorder_point,
    })

# -----------------------------------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------------------------------
print("[3/3] Saving prediction results...")
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH + 'future_predictions.csv', index=False)
print(f"   Saved: {OUTPUT_PATH}future_predictions.csv")
print()
print(results_df.to_string(index=False))

print("\n" + "=" * 65)
print("  PREDICTION COMPLETE!")
print("  Open outputs/future_predictions.csv to review all results.")
print("=" * 65)
print("  Next: Run step5_dashboard.py for the full visual dashboard")
print("=" * 65)
