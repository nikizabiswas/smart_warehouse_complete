"""
================================================================================
SMART WAREHOUSE DEMAND PREDICTION OF LPG CYLINDERS
Step 1: Data Loading & Preprocessing
================================================================================
WHAT THIS FILE DOES:
- Loads the NEW dataset (Warehouse_Demand_Realistic_v2.xlsx)
  which has pre-split Train_80pct and Test_20pct sheets
- Cleans the data (handles missing values, wrong types)
- Encodes text columns into numbers (ML models only understand numbers)
- Scales/normalizes the numerical columns
- Saves the cleaned data for use in Step 2

DATASET CHANGES vs v1:
- New file: Warehouse_Demand_Realistic_v2.xlsx
- Sheet names: 'Train_80pct' and 'Test_20pct'
- Row 1 = category group labels (skip), Row 2 = column headers, Row 3+ = data
- Avg_Daily_Demand is now a float (e.g. 0.0323) -- handled in numeric conversion
- 10 Warehouse Zones: Bali, Berhampore, Domkal, Farakka, Jangipur,
                      Kandi, Lalbagh, Raghunathganj, Samserganj, Suti
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# -- Folder paths --------------------------------------------------------------
DATA_PATH   = "data/Warehouse_Demand_Realistic_v2.xlsx"
OUTPUT_PATH = "outputs/"
MODEL_PATH  = "models/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH,  exist_ok=True)

print("=" * 65)
print("  STEP 1: DATA LOADING & PREPROCESSING  (v2 Dataset)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# The new Excel file has:
#   Row 1 (index 0) : category group labels  -> skip
#   Row 2 (index 1) : actual column names    -> use as header
#   Row 3 onwards   : data rows
#
# header=1 reads row 2 as column names; data starts from row 3.
# No need to drop an extra header row (unlike v1).
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading data from Excel (Warehouse_Demand_Realistic_v2.xlsx)...")

train_raw = pd.read_excel(DATA_PATH, sheet_name="Train_80pct", header=1)
test_raw  = pd.read_excel(DATA_PATH, sheet_name="Test_20pct",  header=1)

print(f"   Training rows : {len(train_raw):,}")
print(f"   Testing rows  : {len(test_raw):,}")
print(f"   Columns       : {len(train_raw.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DROP COLUMNS NOT USEFUL FOR ML
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Dropping non-useful columns...")

DROP_COLS = ['Record_ID', 'Family_ID', 'Month', 'Season', 'Festival']
train = train_raw.drop(columns=DROP_COLS, errors='ignore')
test  = test_raw.drop(columns=DROP_COLS,  errors='ignore')

print(f"   Remaining columns: {list(train.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONVERT COLUMNS TO CORRECT DATA TYPES
# Note: Avg_Daily_Demand is now a float like 0.0323 -- correct & expected.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Converting column data types...")

NUMERIC_COLS = [
    'Year', 'Month_Number', 'Is_Winter', 'Is_Festival_Month', 'Days_in_Month',
    'No_of_Family_Members', 'No_of_Adults', 'No_of_Children',
    'Monthly_Income (Rs)', 'LPG_Price_per_Cylinder (Rs)',
    'Monthly_Income (₹)', 'LPG_Price_per_Cylinder (₹)',
    'Zone_Opening_Stock', 'Zone_Cylinders_Ordered', 'Lead_Time_Days',
    'Zone_Cylinders_Delivered', 'Zone_Safety_Stock', 'Zone_Reorder_Point',
    'Gas_Consumption_kg', 'Actual_Demand (cylinders)',
    'Fulfilled_Demand (cylinders)', 'Units_Short',
    'Stockout_Occurred', 'Damaged_Cylinders', 'Zone_Closing_Stock',
    'Avg_Daily_Demand'
]

for col in NUMERIC_COLS:
    if col in train.columns:
        train[col] = pd.to_numeric(train[col], errors='coerce')
        test[col]  = pd.to_numeric(test[col],  errors='coerce')

print("   Done.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Handling missing values...")

print(f"   Missing in train (before): {train.isnull().sum().sum()}")

for col in train.select_dtypes(include=[np.number]).columns:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col]  = test[col].fillna(median_val)

print(f"   Missing in train (after) : {train.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ENCODE CATEGORICAL COLUMNS
# Zone encoding (alphabetical order from LabelEncoder):
#   Bali=0, Berhampore=1, Domkal=2, Farakka=3, Jangipur=4,
#   Kandi=5, Lalbagh=6, Raghunathganj=7, Samserganj=8, Suti=9
# Subsidy: Non-Subsidized=0, PMUY=1
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Encoding categorical columns...")

CAT_COLS = ['Warehouse_Zone', 'Subsidy_Type']
encoders = {}

for col in CAT_COLS:
    if col in train.columns:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))
        encoders[col] = le
        print(f"   {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

joblib.dump(encoders, MODEL_PATH + 'label_encoders.pkl')
print("   Encoders saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. DEFINE FEATURES AND TARGETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Defining features and targets...")

EXCLUDE = [
    'Actual_Demand (cylinders)',    # target 1
    'Stockout_Occurred',            # target 2
    'Fulfilled_Demand (cylinders)', # derived from demand
    'Units_Short',                  # derived from demand
    'Avg_Daily_Demand',             # derived from demand
    'Gas_Consumption_kg',           # too correlated with target
    'Zone_Safety_Stock',            # computed from demand
    'Zone_Reorder_Point',           # computed from demand
]

FEATURE_COLS = [c for c in train.columns if c not in EXCLUDE]

print(f"   Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

X_train = train[FEATURE_COLS]
X_test  = test[FEATURE_COLS]

y_train_demand = train['Actual_Demand (cylinders)']
y_test_demand  = test['Actual_Demand (cylinders)']

y_train_stock  = train['Stockout_Occurred'].astype(int)
y_test_stock   = test['Stockout_Occurred'].astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 7. SCALE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Scaling features (StandardScaler)...")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_COLS)

joblib.dump(scaler,       MODEL_PATH + 'scaler.pkl')
joblib.dump(FEATURE_COLS, MODEL_PATH + 'feature_cols.pkl')
print("   Scaler saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE PREPROCESSED DATA
# ─────────────────────────────────────────────────────────────────────────────
X_train_scaled.to_csv(OUTPUT_PATH + 'X_train.csv', index=False)
X_test_scaled.to_csv(OUTPUT_PATH  + 'X_test.csv',  index=False)
y_train_demand.to_csv(OUTPUT_PATH + 'y_train_demand.csv', index=False)
y_test_demand.to_csv(OUTPUT_PATH  + 'y_test_demand.csv',  index=False)
y_train_stock.to_csv(OUTPUT_PATH  + 'y_train_stock.csv',  index=False)
y_test_stock.to_csv(OUTPUT_PATH   + 'y_test_stock.csv',   index=False)

print("\n" + "=" * 65)
print("  PREPROCESSING COMPLETE!")
print(f"  X_train shape : {X_train_scaled.shape}")
print(f"  X_test shape  : {X_test_scaled.shape}")
print(f"  Demand target : {y_train_demand.min():.1f} - {y_train_demand.max():.1f} cylinders")
print(f"  Stockout rate : {y_train_stock.mean()*100:.1f}% of training records")
print("=" * 65)
print("  Next: Run step2_model_training.py")
print("=" * 65)
