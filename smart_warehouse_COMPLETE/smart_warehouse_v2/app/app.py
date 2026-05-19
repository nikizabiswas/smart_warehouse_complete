"""
================================================================================
SMART WAREHOUSE DEMAND PREDICTION — STREAMLIT APP  (COMPLETE VERSION)
Mitra Bharatgas Agency | Murshidabad, West Bengal
================================================================================
Pages:
  1. Dashboard         — KPIs, zone stockout, monthly trend, zone stats
  2. Predict Demand    — single-family + batch CSV upload + download
  3. Model Performance — regression, classification, CV, AUC-ROC, feature imp
  4. Zone Analysis     — per-zone deep dive + risk bubble matrix
  5. Data Explorer     — filter & visualise raw training data
  6. About             — methodology, limitations, future work
================================================================================
Run:  streamlit run app/app.py   (from smart_warehouse_v2/ root)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Warehouse — LPG Demand Prediction",
    page_icon="🏭", layout="wide", initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ZONES       = ['Bali','Berhampore','Domkal','Farakka','Jangipur',
               'Kandi','Lalbagh','Raghunathganj','Samserganj','Suti']
ZONE_ENC    = {z:i for i,z in enumerate(ZONES)}
SUBSIDY_ENC = {'Non-Subsidized':0,'PMUY':1}
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
SEASON_MAP  = {1:'Winter',2:'Winter',3:'Summer',4:'Summer',5:'Summer',
               6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',
               10:'Autumn',11:'Autumn',12:'Winter'}
FESTIVAL_MONTHS = {1,8,9,10,11}
WINTER_MONTHS   = {11,12,1,2}
COLORS = ['#1A5276','#1E8449','#B7950B','#C0392B','#7D3C98',
          '#117A65','#784212','#1F618D','#7B241C','#196F3D']

ZONE_SUMMARY = {
    'Bali':          {'records':504,'stockouts':44,'avg_lead':4.11,'avg_closing':751.74,'avg_family':4.36,'avg_income':23219},
    'Berhampore':    {'records':504,'stockouts':59,'avg_lead':6.64,'avg_closing':610.70,'avg_family':4.46,'avg_income':23649},
    'Domkal':        {'records':504,'stockouts':42,'avg_lead':4.17,'avg_closing':918.08,'avg_family':4.52,'avg_income':23040},
    'Farakka':       {'records':504,'stockouts':56,'avg_lead':6.72,'avg_closing':625.06,'avg_family':4.48,'avg_income':23477},
    'Jangipur':      {'records':464,'stockouts':45,'avg_lead':5.43,'avg_closing':700.08,'avg_family':4.41,'avg_income':23850},
    'Kandi':         {'records':504,'stockouts':33,'avg_lead':3.25,'avg_closing':849.39,'avg_family':4.49,'avg_income':23039},
    'Lalbagh':       {'records':504,'stockouts':38,'avg_lead':4.33,'avg_closing':848.25,'avg_family':4.55,'avg_income':24032},
    'Raghunathganj': {'records':504,'stockouts':58,'avg_lead':5.58,'avg_closing':810.02,'avg_family':4.36,'avg_income':23234},
    'Samserganj':    {'records':504,'stockouts':41,'avg_lead':3.50,'avg_closing':980.46,'avg_family':4.27,'avg_income':22520},
    'Suti':          {'records':504,'stockouts':45,'avg_lead':5.47,'avg_closing':732.42,'avg_family':4.64,'avg_income':24167},
}

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR,'models')
OUT_DIR   = os.path.join(BASE_DIR,'outputs')
DATA_DIR  = os.path.join(BASE_DIR,'data')

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS & DATA  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return (
        joblib.load(os.path.join(MODEL_DIR,'best_demand_model.pkl')),
        joblib.load(os.path.join(MODEL_DIR,'best_stockout_model.pkl')),
        joblib.load(os.path.join(MODEL_DIR,'scaler.pkl')),
        joblib.load(os.path.join(MODEL_DIR,'label_encoders.pkl')),
        joblib.load(os.path.join(MODEL_DIR,'feature_cols.pkl')),
        joblib.load(os.path.join(MODEL_DIR,'model_info.pkl')),
    )

@st.cache_data
def load_outputs():
    pred = pd.read_csv(os.path.join(OUT_DIR,'test_predictions.csv'))
    reg  = pd.read_csv(os.path.join(OUT_DIR,'regression_model_comparison.csv'))
    clf  = pd.read_csv(os.path.join(OUT_DIR,'classification_model_comparison.csv'))
    fut  = pd.read_csv(os.path.join(OUT_DIR,'future_predictions.csv'))
    fi   = pd.read_csv(os.path.join(OUT_DIR,'feature_importance.csv'))
    cv   = pd.read_csv(os.path.join(OUT_DIR,'cross_validation_results.csv'))
    agg  = pd.read_csv(os.path.join(OUT_DIR,'zone_aggregate_demand.csv'))
    return pred, reg, clf, fut, fi, cv, agg

@st.cache_data
def load_raw():
    df = pd.read_excel(os.path.join(DATA_DIR,'Warehouse_Demand_Realistic_v2.xlsx'),
                       sheet_name='Train_80pct', header=1)
    for c in ['Month_Number','Year','Actual_Demand (cylinders)','Stockout_Occurred',
              'Monthly_Income (₹)','LPG_Price_per_Cylinder (₹)','No_of_Family_Members',
              'Zone_Opening_Stock','Zone_Closing_Stock','Lead_Time_Days']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

reg_model, clf_model, scaler, encoders, feat_cols, model_info = load_models()
pred_df, reg_df, clf_df, fut_df, fi_df, cv_df, agg_df = load_outputs()
raw_train = load_raw()
THRESHOLD = model_info.get('stockout_threshold', 0.35)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — full dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],.main,.block-container{
    background-color:#0E1117!important;color:#FAFAFA!important}
[data-testid="stSidebar"]{background-color:#1A1D24!important;border-right:1px solid #2C2F36}
[data-testid="stSidebar"] *{color:#FAFAFA!important}
[data-testid="stSidebar"] hr{border-color:#2C2F36!important}
.main-header{background:linear-gradient(135deg,#1F3864 0%,#2874A6 100%);
    padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.2rem;text-align:center}
.main-header h1{color:#fff;font-size:1.8rem;margin:0;font-weight:700}
.main-header p{color:#AED6F1;font-size:.9rem;margin:.3rem 0 0}
.kpi-card{background:#1A1D24;border-radius:10px;padding:1rem 1.2rem;
    border-left:5px solid #2874A6;box-shadow:0 2px 12px rgba(0,0,0,.4);margin-bottom:.5rem}
.kpi-card .kpi-value{font-size:1.6rem;font-weight:800;color:#5DADE2}
.kpi-card .kpi-label{font-size:.75rem;color:#9BAAB8;font-weight:500;
    text-transform:uppercase;letter-spacing:.05em}
.risk-high{background:#C0392B;color:#fff;padding:.3rem .8rem;border-radius:20px;font-weight:700;font-size:.85rem}
.risk-medium{background:#B7950B;color:#fff;padding:.3rem .8rem;border-radius:20px;font-weight:700;font-size:.85rem}
.risk-low{background:#1E8449;color:#fff;padding:.3rem .8rem;border-radius:20px;font-weight:700;font-size:.85rem}
.pred-box{background:#1A1D24;border:2px solid #2874A6;border-radius:12px;padding:1.5rem;margin-top:1rem}
.pred-box h2{color:#AED6F1;font-size:1.05rem;margin:0 0 .8rem}
.section-title{font-size:1.1rem;font-weight:700;color:#AED6F1;
    border-bottom:2px solid #2874A6;padding-bottom:.4rem;margin:1rem 0 .8rem}
.stTabs [data-baseweb="tab"]{background:#1A1D24!important;color:#9BAAB8!important;border-radius:6px 6px 0 0}
.stTabs [aria-selected="true"]{background:#2874A6!important;color:#fff!important}
.stFormSubmitButton button{background:linear-gradient(135deg,#1F3864,#2874A6)!important;
    color:white!important;font-weight:700!important;border:none!important;border-radius:8px!important}
hr{border-color:#2C2F36!important}
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(
    plot_bgcolor='#1A1D24', paper_bgcolor='#1A1D24', font_color='#FAFAFA',
    xaxis=dict(gridcolor='#2C2F36',linecolor='#3D4048'),
    yaxis=dict(gridcolor='#2C2F36',linecolor='#3D4048'),
    legend=dict(bgcolor='#1A1D24',bordercolor='#3D4048'),
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 Smart Warehouse")
    st.markdown("**Mitra Bharatgas Agency**")
    st.markdown("Murshidabad, West Bengal")
    st.divider()
    page = st.radio("Navigate to",[
        "🏠 Dashboard","🔮 Predict Demand","📊 Model Performance",
        "🗺️ Zone Analysis","📈 Data Explorer","ℹ️ About"
    ], label_visibility="collapsed")
    st.divider()
    st.caption(f"**Demand Model:** {model_info['best_regression_model']}")
    st.caption(f"**R²:** {model_info['regression_r2']:.4f}  |  **CV R²:** {model_info['regression_cv_r2_mean']:.4f}±{model_info['regression_cv_r2_std']:.4f}")
    st.caption(f"**Stockout Model:** {model_info['best_classification_model']}")
    st.caption(f"**AUC-ROC:** {model_info['classification_auc_roc']:.4f}  |  **Recall:** {model_info['classification_recall']:.4f}")
    st.caption(f"**Imbalance fix:** {model_info['class_imbalance_method']}")
    st.caption(f"**Decision threshold:** {THRESHOLD}")
    st.divider()
    st.caption("Dataset v2 · 5,000 records · Jan 2022–Dec 2024 · 10 Zones")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — run single prediction
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(zone,year,month,income,subsidy,lpg_price,
                   family_members,adults,children,
                   opening_stock,ordered,lead_time,delivered,damaged,closing_stock):
    errors = []
    if adults + children > family_members:
        errors.append("Adults + Children cannot exceed Total Family Members.")
    if delivered > ordered:
        errors.append("Cylinders Delivered cannot exceed Cylinders Ordered.")
    if closing_stock > opening_stock + delivered:
        errors.append("Closing Stock seems too high vs Opening Stock + Delivered.")
    if errors:
        return None, errors

    is_winter   = 1 if month in WINTER_MONTHS   else 0
    is_festival = 1 if month in FESTIVAL_MONTHS else 0
    days        = [31,28,31,30,31,30,31,31,30,31,30,31][month-1]
    row = {
        'Warehouse_Zone'            : ZONE_ENC[zone],
        'Year'                      : year,
        'Month_Number'              : month,
        'Is_Winter'                 : is_winter,
        'Is_Festival_Month'         : is_festival,
        'Days_in_Month'             : days,
        'No_of_Family_Members'      : family_members,
        'No_of_Adults'              : adults,
        'No_of_Children'            : children,
        'Monthly_Income (\u20b9)'        : income,
        'Subsidy_Type'              : SUBSIDY_ENC[subsidy],
        'LPG_Price_per_Cylinder (\u20b9)': lpg_price,
        'Zone_Opening_Stock'        : opening_stock,
        'Zone_Cylinders_Ordered'    : ordered,
        'Lead_Time_Days'            : lead_time,
        'Zone_Cylinders_Delivered'  : delivered,
        'Damaged_Cylinders'         : damaged,
        'Zone_Closing_Stock'        : closing_stock,
    }
    inp       = pd.DataFrame([{c: row[c] for c in feat_cols}])
    inp_sc    = scaler.transform(inp)
    demand    = max(1, round(float(reg_model.predict(inp_sc)[0]), 2))
    prob      = float(clf_model.predict_proba(inp_sc)[0][1]) * 100
    stockout  = 1 if prob/100 >= THRESHOLD else 0
    risk      = "HIGH" if prob>60 else "MEDIUM" if prob>30 else "LOW"
    safety_stock      = max(1, round(demand * 0.15))
    reorder_point     = max(1, round(demand * 0.25))
    recommended_order = round(demand * 1.10 + safety_stock)
    return {
        'demand':demand,'stockout_prob':prob,'stockout':stockout,'risk':risk,
        'safety_stock':safety_stock,'reorder_point':reorder_point,
        'recommended_order':recommended_order,
        'is_winter':is_winter,'is_festival':is_festival,'season':SEASON_MAP[month],
        'zone':zone,'year':year,'month':month,
    }, []

def predict_batch(df_in):
    """Run predictions on a DataFrame matching the template."""
    rows = []
    for _, r in df_in.iterrows():
        result, errs = run_prediction(
            r.get('Warehouse_Zone','Bali'), int(r.get('Year',2025)),
            int(r.get('Month_Number',1)), int(r.get('Monthly_Income',23000)),
            r.get('Subsidy_Type','PMUY'), int(r.get('LPG_Price',916)),
            int(r.get('No_of_Family_Members',4)), int(r.get('No_of_Adults',2)),
            int(r.get('No_of_Children',2)), int(r.get('Zone_Opening_Stock',700)),
            int(r.get('Zone_Cylinders_Ordered',450)), int(r.get('Lead_Time_Days',5)),
            int(r.get('Zone_Cylinders_Delivered',420)), int(r.get('Damaged_Cylinders',2)),
            int(r.get('Zone_Closing_Stock',700)),
        )
        if result:
            rows.append({'Zone':result['zone'],'Month':MONTH_NAMES[result['month']-1],
                'Year':result['year'],'Predicted_Demand':result['demand'],
                'Stockout_Probability_%':round(result['stockout_prob'],1),
                'Risk_Level':result['risk'],
                'Recommended_Order':result['recommended_order'],
                'Safety_Stock':result['safety_stock']})
    return pd.DataFrame(rows)

def risk_badge(risk):
    cls = {'HIGH':'risk-high','MEDIUM':'risk-medium','LOW':'risk-low'}[risk]
    icons = {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🟢'}
    return f'<span class="{cls}">{icons[risk]} {risk} RISK</span>'

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("""<div class="main-header">
        <h1>🏭 Smart Warehouse — LPG Demand Prediction</h1>
        <p>Mitra Bharatgas Agency &nbsp;|&nbsp; Murshidabad, West Bengal &nbsp;|&nbsp;
        Dataset v2 · 5,000 records · 10 Zones · Jan 2022–Dec 2024</p></div>""",
        unsafe_allow_html=True)

    total_stockouts = sum(z['stockouts'] for z in ZONE_SUMMARY.values())
    total_records   = sum(z['records']   for z in ZONE_SUMMARY.values())
    so_rate         = round(total_stockouts / total_records * 100, 1)
    avg_income      = round(np.mean([z['avg_income'] for z in ZONE_SUMMARY.values()]))

    k1,k2,k3,k4,k5 = st.columns(5)
    for col,val,lbl in zip([k1,k2,k3,k4,k5],
        ['5,000','10',f'{total_stockouts}',f'{so_rate}%',f'₹{avg_income:,}'],
        ['Total Records','Warehouse Zones','Total Stockout Events','Stockout Rate','Avg Monthly Income']):
        col.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                     f'<div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">📍 Stockout Rate by Zone</div>', unsafe_allow_html=True)
        zone_df = pd.DataFrame([{'Zone':z,'Rate':round(d['stockouts']/d['records']*100,1)}
                                  for z,d in ZONE_SUMMARY.items()]).sort_values('Rate',ascending=False)
        avg_so  = zone_df['Rate'].mean()
        fig = go.Figure(go.Bar(x=zone_df['Zone'],y=zone_df['Rate'],
            marker_color=['#C0392B' if v>avg_so else '#1E8449' for v in zone_df['Rate']],
            text=zone_df['Rate'].apply(lambda x:f'{x:.1f}%'),textposition='outside'))
        fig.add_hline(y=avg_so,line_dash='dash',line_color='orange',
                      annotation_text=f'Avg {avg_so:.1f}%')
        fig.update_layout(height=320,xaxis_tickangle=-30,margin=dict(t=20,b=10,l=10,r=10),**DARK_LAYOUT)
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">📅 Monthly Demand Trend</div>', unsafe_allow_html=True)
        monthly = raw_train.groupby('Month_Number')['Actual_Demand (cylinders)'].mean().reset_index()
        sc_map  = {'Winter':'#5DADE2','Summer':'#F39C12','Monsoon':'#27AE60','Autumn':'#E67E22'}
        monthly['Color']  = monthly['Month_Number'].map(SEASON_MAP).map(sc_map)
        monthly['Month']  = monthly['Month_Number'].apply(lambda x: MONTH_NAMES[x-1])
        fig2 = go.Figure(go.Bar(x=monthly['Month'],y=monthly['Actual_Demand (cylinders)'],
            marker_color=monthly['Color'],
            text=monthly['Actual_Demand (cylinders)'].apply(lambda x:f'{x:.2f}'),textposition='outside'))
        for s,c in sc_map.items():
            fig2.add_trace(go.Scatter(x=[None],y=[None],mode='markers',
                marker=dict(color=c,size=10,symbol='square'),name=s,showlegend=True))
        fig2.update_layout(height=320,margin=dict(t=20,b=10,l=10,r=10),
                           legend=dict(orientation='h',y=1.15),**DARK_LAYOUT)
        st.plotly_chart(fig2,use_container_width=True)

    c3,c4 = st.columns([1.1,0.9])
    with c3:
        st.markdown('<div class="section-title">🗂️ Zone Summary</div>', unsafe_allow_html=True)
        zone_tbl = pd.DataFrame([{
            'Zone':z,'Stockouts':d['stockouts'],
            'SO Rate':f"{d['stockouts']/d['records']*100:.1f}%",
            'Avg Lead':f"{d['avg_lead']:.1f}d",
            'Avg Closing':f"{d['avg_closing']:.0f}",
            'Avg Income':f"₹{d['avg_income']:,}"}
            for z,d in ZONE_SUMMARY.items()])
        st.dataframe(zone_tbl,use_container_width=True,hide_index=True,height=290)

    with c4:
        st.markdown('<div class="section-title">🤖 Model Summary</div>', unsafe_allow_html=True)
        mi = model_info
        st.metric("Best Demand Model", mi['best_regression_model'])
        m1,m2 = st.columns(2)
        m1.metric("R² Score",       f"{mi['regression_r2']:.4f}")
        m2.metric("CV R²",          f"{mi['regression_cv_r2_mean']:.4f}±{mi['regression_cv_r2_std']:.4f}")
        st.metric("Best Stockout Model", mi['best_classification_model'])
        m3,m4 = st.columns(2)
        m3.metric("AUC-ROC",        f"{mi['classification_auc_roc']:.4f}")
        m4.metric("Recall",         f"{mi['classification_recall']:.4f}")
        st.caption(f"Imbalance: {mi['class_imbalance_method']}")
        st.caption(f"Decision threshold: {THRESHOLD} (tuned for high recall)")

    st.markdown('<div class="section-title">🔮 2025 Scenario Forecasts</div>', unsafe_allow_html=True)
    fut_show = fut_df.copy()
    fut_show['Zone'] = fut_show['Scenario'].apply(lambda s: s.split('|')[0].strip())
    fut_show['Period'] = fut_show['Scenario'].apply(lambda s: s.split('|')[1].strip() if '|' in s else s)
    st.dataframe(fut_show[['Zone','Period','Predicted_Demand',
        'Stockout_Probability_%','Risk_Level','Recommended_Order_Qty']],
        use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PREDICT DEMAND
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Demand":
    st.markdown("""<div class="main-header">
        <h1>🔮 Predict LPG Demand</h1>
        <p>Single-family prediction or upload a CSV for batch predictions</p>
    </div>""", unsafe_allow_html=True)

    tab_single, tab_batch = st.tabs(["👤 Single Prediction", "📂 Batch CSV Upload"])

    # ── Single prediction ────────────────────────────────────────────────────
    with tab_single:
        with st.form("pred_form"):
            st.markdown('<div class="section-title">📍 Zone & Time</div>', unsafe_allow_html=True)
            fc1,fc2,fc3 = st.columns(3)
            with fc1: zone = st.selectbox("Warehouse Zone", ZONES)
            with fc2: year = st.number_input("Year", min_value=2022, max_value=2035, value=2025, step=1)
            with fc3: month= st.selectbox("Month", range(1,13), format_func=lambda x: MONTH_NAMES[x-1])

            zs = ZONE_SUMMARY[zone]
            st.markdown('<div class="section-title">👨‍👩‍👧 Family Profile</div>', unsafe_allow_html=True)
            ff1,ff2,ff3,ff4 = st.columns(4)
            with ff1: family_members = st.number_input("Total Family Members",1,15,int(zs['avg_family']))
            with ff2: adults         = st.number_input("Adults",1,10,2)
            with ff3: children       = st.number_input("Children",0,10,max(0,int(zs['avg_family'])-2))
            with ff4: subsidy        = st.selectbox("Subsidy Type",['PMUY','Non-Subsidized'])
            income = st.number_input("Monthly Income (₹)",min_value=5000,max_value=80000,
                                     value=int(zs['avg_income']),step=500,
                                     help="Type exact value or use +/− (₹5,000–₹80,000)")

            st.markdown('<div class="section-title">🏪 Market & Operations</div>', unsafe_allow_html=True)
            fm1,fm2 = st.columns(2)
            with fm1: lpg_price = st.number_input("LPG Price per Cylinder (₹)",700,1200,916,1,
                                                   help="Typical range ₹876–₹940")
            with fm2: lead_time = st.number_input("Lead Time (Days)",1,30,int(round(zs['avg_lead'])))

            fo1,fo2,fo3,fo4,fo5 = st.columns(5)
            avg_cs = int(zs['avg_closing'])
            with fo1: opening_stock = st.number_input("Opening Stock",0,5000,avg_cs,10)
            with fo2: ordered       = st.number_input("Cylinders Ordered",0,2000,450,10)
            with fo3: delivered     = st.number_input("Cylinders Delivered",0,2000,420,10)
            with fo4: closing_stock = st.number_input("Closing Stock",0,5000,avg_cs,10)
            with fo5: damaged       = st.number_input("Damaged",0,50,2)

            submitted = st.form_submit_button("🔮 Predict Now",use_container_width=True,type="primary")

        if submitted:
            result, errors = run_prediction(zone,year,month,income,subsidy,lpg_price,
                                            family_members,adults,children,
                                            opening_stock,ordered,lead_time,delivered,
                                            damaged,closing_stock)
            if errors:
                for e in errors: st.error(e)
            else:
                st.markdown("---")
                tags = []
                if result['is_winter']:   tags.append("❄️ Winter")
                if result['is_festival']: tags.append("🎉 Festival")
                tags.append(f"🌤️ {result['season']}")
                st.markdown(f"""<div class="pred-box">
                    <h2>Zone: {zone} &nbsp;|&nbsp; {MONTH_NAMES[month-1]} {year}
                    &nbsp;|&nbsp; {" &nbsp; ".join(tags)}</h2></div>""",
                    unsafe_allow_html=True)

                r1,r2,r3,r4 = st.columns(4)
                r1.metric("📦 Predicted Demand",   f"{result['demand']:.2f} cyl")
                r2.metric("⚠️ Stockout Probability",f"{result['stockout_prob']:.1f}%")
                r3.metric("🛒 Recommended Order",   f"{result['recommended_order']} cyl")
                r4.metric("🔒 Safety Stock",        f"{result['safety_stock']} cyl")

                ra,rb = st.columns(2)
                ra.markdown(f"**Risk Level:** {risk_badge(result['risk'])}", unsafe_allow_html=True)
                rb.metric("📍 Reorder Point", f"{result['reorder_point']} cyl")

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=result['stockout_prob'],
                    number={'suffix':'%','font':{'size':28}},
                    title={'text':f"Stockout Risk (threshold={THRESHOLD})",'font':{'size':13}},
                    gauge={'axis':{'range':[0,100]},
                           'bar':{'color':'#2874A6'},
                           'steps':[{'range':[0,30],'color':'#1A3A2A'},
                                    {'range':[30,60],'color':'#3D2B00'},
                                    {'range':[60,100],'color':'#3D0000'}],
                           'threshold':{'line':{'color':'red','width':3},
                                        'thickness':0.75,'value':result['stockout_prob']}}))
                fig_g.update_layout(height=260,margin=dict(t=40,b=10,l=20,r=20),**DARK_LAYOUT)
                st.plotly_chart(fig_g, use_container_width=True)

                # Download single result
                single_df = pd.DataFrame([{
                    'Zone':zone,'Month':MONTH_NAMES[month-1],'Year':year,
                    'Predicted_Demand':result['demand'],
                    'Stockout_Probability_%':round(result['stockout_prob'],1),
                    'Risk_Level':result['risk'],
                    'Recommended_Order':result['recommended_order'],
                    'Safety_Stock':result['safety_stock'],
                    'Reorder_Point':result['reorder_point'],
                }])
                csv_bytes = single_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download this prediction as CSV",
                                   data=csv_bytes, file_name=f"prediction_{zone}_{MONTH_NAMES[month-1]}_{year}.csv",
                                   mime="text/csv")

    # ── Batch prediction ─────────────────────────────────────────────────────
    with tab_batch:
        st.markdown('<div class="section-title">📂 Batch Prediction via CSV Upload</div>', unsafe_allow_html=True)
        st.markdown("Upload a CSV with one row per family. Required columns:")
        template_cols = ['Warehouse_Zone','Year','Month_Number','Monthly_Income',
                         'Subsidy_Type','LPG_Price','No_of_Family_Members',
                         'No_of_Adults','No_of_Children','Zone_Opening_Stock',
                         'Zone_Cylinders_Ordered','Lead_Time_Days',
                         'Zone_Cylinders_Delivered','Damaged_Cylinders','Zone_Closing_Stock']
        template_data = {
            'Warehouse_Zone':['Berhampore','Kandi','Suti'],
            'Year':[2025,2025,2025],'Month_Number':[1,6,10],
            'Monthly_Income':[23649,23039,24167],
            'Subsidy_Type':['PMUY','Non-Subsidized','PMUY'],
            'LPG_Price':[921,903,910],
            'No_of_Family_Members':[4,4,5],'No_of_Adults':[2,2,3],
            'No_of_Children':[2,2,2],
            'Zone_Opening_Stock':[550,940,700],'Zone_Cylinders_Ordered':[480,420,490],
            'Lead_Time_Days':[7,4,5],'Zone_Cylinders_Delivered':[450,400,460],
            'Damaged_Cylinders':[3,1,3],'Zone_Closing_Stock':[610,980,732],
        }
        template_df = pd.DataFrame(template_data)
        tmpl_csv = template_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV Template", data=tmpl_csv,
                           file_name="batch_prediction_template.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload your filled CSV", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                st.success(f"Loaded {len(df_up)} rows")
                st.dataframe(df_up.head(), use_container_width=True, hide_index=True)
                if st.button("▶️ Run Batch Predictions", type="primary"):
                    with st.spinner("Running predictions..."):
                        batch_results = predict_batch(df_up)
                    st.markdown("### Results")
                    st.dataframe(batch_results, use_container_width=True, hide_index=True)

                    fig_b = px.bar(batch_results, x='Zone', y='Predicted_Demand',
                                   color='Risk_Level',
                                   color_discrete_map={'HIGH':'#C0392B','MEDIUM':'#B7950B','LOW':'#1E8449'},
                                   text='Predicted_Demand', title='Batch Predictions — Demand by Zone')
                    fig_b.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10), **DARK_LAYOUT)
                    st.plotly_chart(fig_b, use_container_width=True)

                    csv_out = batch_results.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Batch Results CSV",
                                       data=csv_out, file_name="batch_predictions.csv", mime="text/csv")
            except Exception as ex:
                st.error(f"Error reading file: {ex}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown("""<div class="main-header">
        <h1>📊 Model Performance</h1>
        <p>Regression · Classification · Cross-validation · AUC-ROC · Feature Importance</p>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.tabs(["📉 Regression","🎯 Classification",
                                "🔁 Cross-Validation","📈 ROC & PR Curves","🌟 Feature Importance"])

    with t1:
        st.caption("R² closer to 1.0 = better. MAE and RMSE lower = less error. CV R² proves the model generalises, not just memorises.")
        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(reg_df.sort_values('R2_Score'), x='R2_Score', y='Model',
                         orientation='h', text=reg_df.sort_values('R2_Score')['R2_Score'].apply(lambda x:f'{x:.4f}'),
                         color='R2_Score', color_continuous_scale='Blues', title='R² Score')
            fig.add_vline(x=0.9,line_dash='dash',line_color='red',annotation_text='Target 0.90')
            fig.update_layout(height=300,coloraxis_showscale=False,
                              margin=dict(t=40,b=10,l=10,r=10),**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(reg_df.sort_values('CV_R2_Mean'), x='CV_R2_Mean', y='Model',
                          error_x='CV_R2_Std', orientation='h',
                          text=reg_df.sort_values('CV_R2_Mean')['CV_R2_Mean'].apply(lambda x:f'{x:.4f}'),
                          title='5-Fold CV R² (Mean ± Std)', color_continuous_scale='Blues',
                          color='CV_R2_Mean')
            fig2.update_layout(height=300, coloraxis_showscale=False,
                               margin=dict(t=40,b=10,l=10,r=10),**DARK_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(reg_df.style.highlight_max(subset=['R2_Score','CV_R2_Mean'],color='#1E3A2A')
                              .highlight_min(subset=['MAE','RMSE'],color='#1E3A2A')
                              .format({'R2_Score':'{:.4f}','MAE':'{:.4f}','RMSE':'{:.4f}',
                                       'CV_R2_Mean':'{:.4f}','CV_R2_Std':'{:.4f}'}),
                     use_container_width=True, hide_index=True)
        # Actual vs Predicted
        st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
        p1,p2 = st.columns(2)
        with p1:
            fig_sc = px.scatter(pred_df,x='Actual_Demand',y='Predicted_Demand',opacity=0.45,
                                color_discrete_sequence=['#5DADE2'],title='Actual vs Predicted Demand')
            mn = pred_df['Actual_Demand'].min()-0.1; mx = pred_df['Actual_Demand'].max()+0.1
            fig_sc.add_shape(type='line',x0=mn,y0=mn,x1=mx,y1=mx,
                             line=dict(color='red',dash='dash',width=2))
            fig_sc.update_layout(height=300,margin=dict(t=40,b=10,l=10,r=10),**DARK_LAYOUT)
            st.plotly_chart(fig_sc, use_container_width=True)
        with p2:
            errs = pred_df['Actual_Demand']-pred_df['Predicted_Demand']
            fig_err = px.histogram(errs,nbins=40,title='Error Distribution',
                                   color_discrete_sequence=['#5DADE2'])
            fig_err.add_vline(x=0,line_dash='dash',line_color='red',annotation_text='Zero Error')
            fig_err.add_vline(x=float(errs.mean()),line_dash='dot',line_color='orange',
                              annotation_text=f'Mean={errs.mean():.4f}')
            fig_err.update_layout(height=300,showlegend=False,margin=dict(t=40,b=10,l=10,r=10),**DARK_LAYOUT)
            st.plotly_chart(fig_err, use_container_width=True)

    with t2:
        st.caption(f"Stockout class imbalance: 90.8% No / 9.2% Yes. Fix used: **{model_info['class_imbalance_method']}**. Decision threshold: **{THRESHOLD}** (tuned for high recall — catching real stockouts matters more than precision).")
        c1,c2 = st.columns(2)
        with c1:
            metrics_long = clf_df.melt(id_vars='Model',value_vars=['Recall','Precision','F1_Score','AUC_ROC'],
                                        var_name='Metric',value_name='Score')
            fig_m = px.bar(metrics_long,x='Model',y='Score',color='Metric',barmode='group',
                           title='All Metrics by Model',color_discrete_sequence=COLORS)
            fig_m.update_layout(height=330,xaxis_tickangle=-20,margin=dict(t=40,b=10,l=10,r=10),**DARK_LAYOUT)
            st.plotly_chart(fig_m, use_container_width=True)
        with c2:
            cm_vals = model_info['confusion_matrix']
            cm_df2  = pd.DataFrame(cm_vals,
                index=['Actual No Stockout','Actual Stockout'],
                columns=['Pred No Stockout','Pred Stockout'])
            fig_cm = px.imshow(cm_df2,text_auto=True,color_continuous_scale='Blues',
                               title=f'Confusion Matrix — {model_info["best_classification_model"]} (threshold={THRESHOLD})')
            fig_cm.update_layout(height=330,margin=dict(t=50,b=10,l=10,r=10),**DARK_LAYOUT)
            st.plotly_chart(fig_cm, use_container_width=True)

        tn,fp,fn,tp = cm_vals[0][0],cm_vals[0][1],cm_vals[1][0],cm_vals[1][1]
        total_so = fn+tp
        e1,e2,e3,e4 = st.columns(4)
        e1.metric("Stockouts Caught (TP)",  f"{tp} / {total_so}", f"{tp/total_so*100:.1f}% recall")
        e2.metric("Missed Stockouts (FN)",  str(fn), delta=f"-{fn/total_so*100:.1f}%", delta_color="inverse")
        e3.metric("False Alarms (FP)",      str(fp))
        e4.metric("Correct Non-Stockouts",  str(tn))

        st.dataframe(clf_df.style.highlight_max(subset=['Recall','F1_Score','AUC_ROC'],color='#1E3A2A')
                              .format({'Accuracy':'{:.4f}','Precision':'{:.4f}','Recall':'{:.4f}',
                                       'F1_Score':'{:.4f}','AUC_ROC':'{:.4f}','Avg_Precision':'{:.4f}',
                                       'CV_F1_Mean':'{:.4f}','CV_F1_Std':'{:.4f}'}),
                     use_container_width=True, hide_index=True)

    with t3:
        st.caption("Cross-validation splits training data into 5 folds, trains on 4, tests on 1, repeats 5 times. The mean and std show whether the model is consistent across different data subsets.")
        c1,c2 = st.columns(2)
        reg_cv = cv_df[cv_df['Task']=='Regression'].sort_values('CV_Mean')
        clf_cv = cv_df[cv_df['Task']=='Classification'].sort_values('CV_Mean')
        for col,df2,title,color in [(c1,reg_cv,'5-Fold CV R² — Regression','#1A5276'),
                                     (c2,clf_cv,'5-Fold CV F1 — Classification','#1E8449')]:
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(x=df2['CV_Mean'],y=df2['Model'],orientation='h',
                error_x=dict(type='data',array=df2['CV_Std'],visible=True),
                marker_color=color,text=df2.apply(lambda r:f"{r['CV_Mean']:.4f}±{r['CV_Std']:.4f}",axis=1),
                textposition='outside'))
            fig_cv.update_layout(title=title,height=300,xaxis=dict(range=[0,1.2]),
                                  margin=dict(t=40,b=10,l=10,r=150),**DARK_LAYOUT)
            col.plotly_chart(fig_cv, use_container_width=True)

        st.markdown('<div class="section-title">What cross-validation tells us</div>', unsafe_allow_html=True)
        st.info(f"""
**Regression (Gradient Boosting):** CV R² = {model_info['regression_cv_r2_mean']:.4f} ± {model_info['regression_cv_r2_std']:.4f}.
The std of ±{model_info['regression_cv_r2_std']:.4f} is acceptable — the model is consistent across folds.

**Classification (Decision Tree):** CV F1 = {model_info['clf_cv_f1_mean']:.4f} ± {model_info['clf_cv_f1_std']:.4f}.
The F1 score is stable. Note: CV was run on SMOTE-resampled data so folds are balanced.
        """)

    with t4:
        st.caption("ROC curve shows the tradeoff between True Positive Rate and False Positive Rate across all thresholds. Precision-Recall curve is more informative for imbalanced datasets like this one (9.2% stockout rate).")
        X_test_df = pd.read_csv(os.path.join(OUT_DIR,'X_test.csv'))
        y_test_s  = pd.read_csv(os.path.join(OUT_DIR,'y_test_stock.csv')).squeeze().astype(int)
        probs_all = clf_model.predict_proba(X_test_df)[:,1]

        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr,tpr,thresh_roc = roc_curve(y_test_s, probs_all)
        prec_c,rec_c,_     = precision_recall_curve(y_test_s, probs_all)

        c1,c2 = st.columns(2)
        with c1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'AUC={model_info["classification_auc_roc"]:.4f}',line=dict(color='#5DADE2',width=2)))
            fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random',line=dict(color='red',dash='dash',width=1)))
            # Mark current threshold
            curr_idx = np.argmin(np.abs(probs_all - THRESHOLD)) if len(probs_all)>0 else 0
            fig_roc.update_layout(title=f'ROC Curve — AUC={model_info["classification_auc_roc"]:.4f}',
                                   xaxis_title='False Positive Rate',yaxis_title='True Positive Rate',
                                   height=340,margin=dict(t=50,b=10,l=10,r=10),**DARK_LAYOUT)
            st.plotly_chart(fig_roc, use_container_width=True)
        with c2:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec_c,y=prec_c,mode='lines',
                name=f'Avg Precision={model_info["classification_avg_prec"]:.4f}',
                line=dict(color='#1E8449',width=2)))
            fig_pr.add_hline(y=y_test_s.mean(),line_dash='dash',line_color='red',
                              annotation_text=f'Baseline={y_test_s.mean():.3f}')
            fig_pr.update_layout(title='Precision-Recall Curve',
                                  xaxis_title='Recall',yaxis_title='Precision',
                                  height=340,margin=dict(t=50,b=10,l=10,r=10),**DARK_LAYOUT)
            st.plotly_chart(fig_pr, use_container_width=True)

    with t5:
        st.caption("Shows which input features have the most influence on demand prediction. Computed from the Gradient Boosting regression model using mean decrease in impurity.")
        top_n = st.slider("Show top N features", 5, 18, 12)
        fi_show = fi_df.head(top_n).sort_values('Importance')
        fig_fi = go.Figure(go.Bar(
            x=fi_show['Importance'], y=fi_show['Feature'], orientation='h',
            marker_color=['#C0392B' if v==fi_show['Importance'].max() else '#1A5276'
                          for v in fi_show['Importance']],
            text=fi_show['Importance'].apply(lambda x:f'{x:.4f}'), textposition='outside'))
        fig_fi.update_layout(title=f'Top {top_n} Feature Importances — {model_info["best_regression_model"]}',
                              height=max(300, top_n*35),
                              margin=dict(t=50,b=10,l=10,r=80),**DARK_LAYOUT)
        st.plotly_chart(fig_fi, use_container_width=True)
        st.info(f"""
**Top driver: {fi_df.iloc[0]['Feature']}** ({fi_df.iloc[0]['Importance']*100:.1f}% importance) — Family size is the single strongest predictor of demand.
**2nd: {fi_df.iloc[1]['Feature']}** ({fi_df.iloc[1]['Importance']*100:.1f}%) — How many cylinders were actually delivered this month.
**3rd: {fi_df.iloc[2]['Feature']}** ({fi_df.iloc[2]['Importance']*100:.1f}%) — Ordering patterns signal expected demand.
        """)
        st.dataframe(fi_df, use_container_width=True, hide_index=True)
        fi_csv = fi_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Feature Importance CSV", data=fi_csv,
                           file_name="feature_importance.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ZONE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Zone Analysis":
    st.markdown("""<div class="main-header">
        <h1>🗺️ Zone-Level Analysis</h1>
        <p>Per-zone demand trends, stockout risk, and aggregate monthly totals</p>
    </div>""", unsafe_allow_html=True)

    sel_zone = st.selectbox("Select Zone for Deep Dive", ZONES)
    zd = ZONE_SUMMARY[sel_zone]

    z1,z2,z3,z4 = st.columns(4)
    z1.metric("Records",      f"{zd['records']:,}")
    z2.metric("Stockouts",    f"{zd['stockouts']}",
              delta=f"{zd['stockouts']/zd['records']*100:.1f}%", delta_color="inverse")
    z3.metric("Avg Lead Time",f"{zd['avg_lead']:.1f} days")
    z4.metric("Avg Closing",  f"{zd['avg_closing']:.0f} cyl")
    st.divider()

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="section-title">Monthly Demand — {sel_zone}</div>', unsafe_allow_html=True)
        zone_m = raw_train[raw_train['Warehouse_Zone']==sel_zone].groupby('Month_Number')['Actual_Demand (cylinders)'].mean().reset_index()
        zone_m['Month']  = zone_m['Month_Number'].apply(lambda x: MONTH_NAMES[x-1])
        zone_m['Season'] = zone_m['Month_Number'].map(SEASON_MAP)
        sc2 = {'Winter':'#5DADE2','Summer':'#F39C12','Monsoon':'#27AE60','Autumn':'#E67E22'}
        zone_m['Color'] = zone_m['Season'].map(sc2)
        fig_zm = go.Figure(go.Bar(x=zone_m['Month'],y=zone_m['Actual_Demand (cylinders)'],
            marker_color=zone_m['Color'],
            text=zone_m['Actual_Demand (cylinders)'].apply(lambda x:f'{x:.2f}'),textposition='outside'))
        fig_zm.update_layout(height=300,margin=dict(t=20,b=10,l=10,r=10),**DARK_LAYOUT)
        st.plotly_chart(fig_zm, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Zone Comparison</div>', unsafe_allow_html=True)
        metric_ch = st.selectbox("Compare by",
            ['Stockout Rate %','Avg Lead Time (days)','Avg Closing Stock','Avg Income ₹'])
        compare_data = {'Zone':list(ZONE_SUMMARY.keys())}
        compare_data['Stockout Rate %']      = [round(d['stockouts']/d['records']*100,1) for d in ZONE_SUMMARY.values()]
        compare_data['Avg Lead Time (days)'] = [d['avg_lead'] for d in ZONE_SUMMARY.values()]
        compare_data['Avg Closing Stock']    = [d['avg_closing'] for d in ZONE_SUMMARY.values()]
        compare_data['Avg Income ₹']         = [d['avg_income'] for d in ZONE_SUMMARY.values()]
        cmp_df = pd.DataFrame(compare_data)
        colors_cmp = ['#C0392B' if z==sel_zone else '#1A5276' for z in cmp_df['Zone']]
        fig_cmp = go.Figure(go.Bar(x=cmp_df['Zone'],y=cmp_df[metric_ch],
            marker_color=colors_cmp,text=cmp_df[metric_ch].apply(lambda x:f'{x:.1f}'),
            textposition='outside'))
        fig_cmp.update_layout(height=300,xaxis_tickangle=-30,margin=dict(t=20,b=10,l=10,r=10),**DARK_LAYOUT)
        st.plotly_chart(fig_cmp, use_container_width=True)

    # Zone aggregate demand trend (NEW)
    st.markdown(f'<div class="section-title">Zone-Level Aggregate Demand — {sel_zone} (Total cylinders/month)</div>', unsafe_allow_html=True)
    zone_agg_sel = agg_df[agg_df['Warehouse_Zone']==sel_zone].copy()
    zone_agg_sel['Period'] = zone_agg_sel['Year'].astype(str)+'-'+zone_agg_sel['Month_Number'].astype(str).str.zfill(2)
    fig_agg = go.Figure()
    fig_agg.add_trace(go.Scatter(x=zone_agg_sel['Period'],y=zone_agg_sel['Total_Demand'],
        mode='lines+markers',name='Total Demand',line=dict(color='#5DADE2',width=2)))
    fig_agg.add_trace(go.Bar(x=zone_agg_sel['Period'],y=zone_agg_sel['Stockouts'],
        name='Stockout Events',marker_color='#C0392B',opacity=0.6,yaxis='y2'))
    fig_agg.update_layout(height=320,
        yaxis=dict(title='Total Demand (cylinders)',**{k:v for k,v in DARK_LAYOUT['yaxis'].items()}),
        yaxis2=dict(title='Stockouts',overlaying='y',side='right',
                    gridcolor='#2C2F36',linecolor='#3D4048'),
        xaxis_tickangle=-45,legend=dict(bgcolor='#1A1D24'),
        margin=dict(t=20,b=10,l=10,r=60),**{k:v for k,v in DARK_LAYOUT.items() if k not in ['yaxis']})
    st.plotly_chart(fig_agg, use_container_width=True)

    # Risk bubble matrix
    st.markdown('<div class="section-title">Zone Risk Matrix — Stockout Rate vs Lead Time</div>', unsafe_allow_html=True)
    risk_matrix = pd.DataFrame([{
        'Zone':z,'Stockout Rate %':round(d['stockouts']/d['records']*100,1),
        'Avg Lead Time':d['avg_lead'],'Records':d['records'],'Avg Income':d['avg_income']}
        for z,d in ZONE_SUMMARY.items()])
    fig_bub = px.scatter(risk_matrix,x='Avg Lead Time',y='Stockout Rate %',
        size='Records',color='Zone',text='Zone',hover_data=['Avg Income'],
        title='Zone Risk Matrix (top-right = highest risk)',
        color_discrete_sequence=COLORS)
    fig_bub.update_traces(textposition='top center')
    fig_bub.update_layout(height=400,margin=dict(t=50,b=10,l=10,r=10),**DARK_LAYOUT)
    st.plotly_chart(fig_bub, use_container_width=True)

    # Download zone aggregate
    agg_csv = agg_df[agg_df['Warehouse_Zone']==sel_zone].to_csv(index=False).encode()
    st.download_button(f"⬇️ Download {sel_zone} Aggregate Demand CSV",
                       data=agg_csv, file_name=f"zone_demand_{sel_zone}.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Data Explorer":
    st.markdown("""<div class="main-header">
        <h1>📈 Data Explorer</h1>
        <p>Filter and visualise the raw training dataset interactively</p>
    </div>""", unsafe_allow_html=True)

    fe1,fe2,fe3 = st.columns(3)
    with fe1: sel_zones  = st.multiselect("Zone",   ZONES, default=ZONES[:3])
    with fe2: sel_years  = st.multiselect("Year",   sorted(raw_train['Year'].dropna().astype(int).unique()), default=[2022])
    with fe3: sel_months = st.multiselect("Month",  list(range(1,13)),
                                           format_func=lambda x: MONTH_NAMES[x-1], default=list(range(1,4)))

    filtered = raw_train.copy()
    if sel_zones:  filtered = filtered[filtered['Warehouse_Zone'].isin(sel_zones)]
    if sel_years:  filtered = filtered[filtered['Year'].isin(sel_years)]
    if sel_months: filtered = filtered[filtered['Month_Number'].isin(sel_months)]

    st.markdown(f"**Showing {len(filtered):,} records**")
    e1,e2 = st.columns(2)

    with e1:
        st.markdown('<div class="section-title">Income Distribution</div>', unsafe_allow_html=True)
        fig_inc = px.histogram(filtered,x='Monthly_Income (₹)',color='Warehouse_Zone',
                               nbins=30,opacity=0.75,color_discrete_sequence=COLORS)
        fig_inc.update_layout(height=300,margin=dict(t=20,b=10,l=10,r=10),**DARK_LAYOUT)
        st.plotly_chart(fig_inc, use_container_width=True)

    with e2:
        st.markdown('<div class="section-title">LPG Price Trend</div>', unsafe_allow_html=True)
        pt = filtered.groupby(['Year','Month_Number'])['LPG_Price_per_Cylinder (₹)'].mean().reset_index()
        pt['Period'] = pt['Year'].astype(str)+'-'+pt['Month_Number'].astype(str).str.zfill(2)
        fig_price = px.line(pt,x='Period',y='LPG_Price_per_Cylinder (₹)',
                            color_discrete_sequence=['#C0392B'])
        fig_price.update_layout(height=300,xaxis_tickangle=-45,margin=dict(t=20,b=10,l=10,r=10),**DARK_LAYOUT)
        st.plotly_chart(fig_price, use_container_width=True)

    st.markdown('<div class="section-title">Demand by Family Size</div>', unsafe_allow_html=True)
    fig_fam = px.box(filtered,x='No_of_Family_Members',y='Actual_Demand (cylinders)',
                     color='Warehouse_Zone',color_discrete_sequence=COLORS)
    fig_fam.update_layout(height=340,margin=dict(t=20,b=10,l=10,r=10),**DARK_LAYOUT)
    st.plotly_chart(fig_fam, use_container_width=True)

    with st.expander("📄 Raw Data Table (up to 500 rows)"):
        show_cols = ['Warehouse_Zone','Year','Month_Number','Actual_Demand (cylinders)',
                     'Stockout_Occurred','Monthly_Income (₹)','LPG_Price_per_Cylinder (₹)',
                     'No_of_Family_Members','Subsidy_Type','Zone_Opening_Stock','Zone_Closing_Stock','Lead_Time_Days']
        avail = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[avail].head(500), use_container_width=True, hide_index=True)
        dl_csv = filtered[avail].to_csv(index=False).encode()
        st.download_button("⬇️ Download Filtered Data CSV", data=dl_csv,
                           file_name="filtered_data.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("""<div class="main-header">
        <h1>ℹ️ About This Project</h1>
        <p>Methodology · Dataset · Limitations · Future Work</p>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["📋 Overview","🔬 Methodology","⚠️ Limitations","🚀 Future Work"])

    with t1:
        st.markdown("""
### Smart Warehouse Demand Prediction of Consumer Items
**A Machine Learning Based Approach**

**Organisation:** Mitra Bharatgas Agency, Murshidabad, West Bengal
**Scope:** Monthly LPG cylinder demand forecasting and stockout risk prediction
across 10 distribution zones using supervised machine learning.

#### Problem Statement
The agency manually decides how many LPG cylinders to order each month per zone.
Under-ordering causes stockouts (families without gas, trust loss).
Over-ordering locks capital and increases damage risk.
This system replaces intuition with a data-driven ML prediction pipeline.

#### Two Core Predictions
| Task | Type | Model | Key Metric |
|------|------|-------|-----------|
| Monthly demand (cylinders) | Regression | Gradient Boosting | R²=0.34, CV R²=0.35±0.08 |
| Stockout risk (yes/no) | Classification | Decision Tree (SMOTE) | AUC-ROC=0.60, Recall=47% |

#### Dataset
- **File:** Warehouse_Demand_Realistic_v2.xlsx
- **Records:** 5,000 (4,000 train / 1,000 test)
- **Period:** January 2022 – December 2024
- **Zones:** Bali, Berhampore, Domkal, Farakka, Jangipur, Kandi, Lalbagh, Raghunathganj, Samserganj, Suti
- **Features used:** 18 of 31 columns (leakage columns excluded)

#### Before vs After ML
| Metric | Without ML | With ML |
|--------|-----------|---------|
| Stockout detection | Reactive (after it happens) | Predictive (before it happens) |
| Order quantity | Manager's estimate | Model recommendation |
| Zone risk ranking | Experience-based | Data-driven (76% stockout recall) |
| Festival planning | Manual calendar check | Is_Festival_Month flag in model |
        """)

    with t2:
        st.markdown("""
### Methodology

#### Pipeline Steps
1. **Data Preprocessing** (step1) — load from Excel, drop non-predictive columns,
   encode categorical features (zone → 0–9, subsidy → 0/1), apply StandardScaler
2. **Model Training** (step2) — 4 regression + 4 classification models, SMOTE for
   class balancing, 5-fold cross-validation, RandomizedSearchCV hyperparameter tuning
3. **Visualization** (step3) — 10 performance charts including ROC/PR curves and CV comparison
4. **Future Prediction** (step4) — 6 planning scenarios for 2025 across different zones/seasons
5. **Dashboard** (step5) — master PNG report + Excel export

#### Why Each Model Choice
- **Gradient Boosting (regression):** Best R² (0.34) among 4 models. Ensemble of decision
  trees that correct each other's errors. Handles non-linear relationships between family size,
  income, season, and demand.
- **Decision Tree (classification):** Best F1 among 4 models after SMOTE. Interpretable
  (you can print the tree). Uses threshold=0.35 (not 0.5) to bias toward catching stockouts.

#### Class Imbalance Treatment
Dataset has 90.8% no-stockout vs 9.2% stockout. Two methods applied together:
1. **SMOTE** (Synthetic Minority Over-sampling Technique) — creates synthetic stockout
   examples in feature space to balance training data to 50/50
2. **class_weight='balanced'** — weights the loss function to penalise misclassifying
   stockouts more heavily than non-stockouts

#### Leakage Prevention
8 columns excluded from features because they contain future information:
Fulfilled_Demand, Units_Short, Avg_Daily_Demand, Gas_Consumption_kg,
Zone_Safety_Stock, Zone_Reorder_Point (all derived from actual demand)

#### Decision Threshold = 0.35
Default threshold for classification is 0.50. We lower it to 0.35 because:
- A false alarm (flagged as stockout but wasn't) costs only an extra order
- A missed stockout means families have no gas — much worse
- At 0.35: recall = 47%, catching nearly half of real stockouts
        """)

    with t3:
        st.markdown("""
### Known Limitations

#### Data Limitations
- **Synthetic dataset:** The data is generated based on realistic Murshidabad parameters,
  not collected from an actual Bharatgas depot. Performance on real data may differ.
- **Demand is nearly binary:** 97.5% of records have demand = 1 cylinder. This makes the
  regression trivial and explains the modest R² of 0.34. A real dataset would show more variance.
- **3 years of data only:** Jan 2022–Dec 2024. More years would improve seasonal pattern detection.

#### Model Limitations
- **Stockout recall = 47%:** Even with SMOTE + balanced weights + threshold tuning,
  the model still misses ~53% of real stockouts. This is primarily a data limitation —
  the features available (stock levels, orders) don't fully explain why stockouts occur.
- **AUC-ROC = 0.60:** Slightly above random (0.50). The stockout signal in the current
  features is weak. Adding external data (weather, regional events, supply chain disruptions)
  could improve this significantly.
- **No temporal modelling:** The model treats each month as independent. Time-series methods
  (LSTM, ARIMA, Prophet) would capture month-to-month dependencies.

#### Deployment Limitations
- **No real-time data ingestion:** The app uses static pre-loaded data. In production,
  it would need a live database connection.
- **No retraining automation:** Models must be retrained manually when new data arrives.
        """)

    with t4:
        st.markdown("""
### Future Work

#### Short-term (1–3 months)
- [ ] **Collect real depot data** from Mitra Bharatgas — replace synthetic dataset
- [ ] **Add weather data** — temperature strongly predicts cooking gas usage
- [ ] **Add SHAP explainability** — show per-prediction feature contributions
- [ ] **Encode individual festivals** — one-hot encode Durga Puja, Diwali, Eid separately

#### Medium-term (3–6 months)
- [ ] **Time-series model** — implement Facebook Prophet or LSTM for sequential demand
- [ ] **Zone-level regression** — predict total zone demand (not per-family) directly
- [ ] **API deployment** — wrap models in a FastAPI endpoint for integration with depot ERP
- [ ] **Automated retraining** — monthly cron job to retrain when >500 new records available

#### Long-term
- [ ] **Multi-product extension** — extend beyond LPG to kerosene, coal, other consumer items
- [ ] **Supply chain optimisation** — recommend optimal lead times per zone using reinforcement learning
- [ ] **Mobile app** — lightweight Flutter app for field officers to log deliveries and get predictions
        """)

    st.divider()
    st.caption("Built with: Streamlit · Plotly · Scikit-learn · Pandas · imbalanced-learn")
    st.caption(f"Models: Gradient Boosting (demand) · Decision Tree SMOTE (stockout) · Threshold={THRESHOLD}")
