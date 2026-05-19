# Smart Warehouse Streamlit App
## Mitra Bharatgas Agency — LPG Demand Prediction

### Prerequisites
Make sure you have run all 5 pipeline steps first:
  python step1_data_preprocessing.py
  python step2_model_training.py
  python step3_visualization.py
  python step4_predict_new.py
  python step5_dashboard.py

### Install dependencies
  pip install -r app/requirements.txt

### Run the app
From inside the smart_warehouse_v2/ directory:
  streamlit run app/app.py

The app will open at: http://localhost:8501

### Pages
  1. Dashboard      — KPIs, zone stockout chart, monthly trend, zone stats table
  2. Predict Demand — Live prediction form with gauge chart output
  3. Model Performance — Regression & classification model comparison + confusion matrix
  4. Zone Analysis  — Per-zone deep dive + risk bubble matrix
  5. Data Explorer  — Filter and visualize the raw training data interactively
