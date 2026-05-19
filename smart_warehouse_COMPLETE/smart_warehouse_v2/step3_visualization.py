"""
Step 3: Visualization — updated for improved model outputs
Generates 10 charts (added CV comparison + feature importance + AUC curves)
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches, seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

OUTPUT_PATH = "outputs/"; MODEL_PATH = "models/"; CHART_PATH = "charts/"
os.makedirs(CHART_PATH, exist_ok=True)

plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.3,'figure.dpi':150})
COLORS = ['#1A5276','#1E8449','#B7950B','#C0392B','#7D3C98','#117A65']
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
SEASON_MAP  = {1:'Winter',2:'Winter',3:'Summer',4:'Summer',5:'Summer',
               6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',10:'Autumn',11:'Autumn',12:'Winter'}

print("="*65); print("  STEP 3: VISUALIZATION (10 charts)"); print("="*65)

# Load
pred_df  = pd.read_csv(OUTPUT_PATH+'test_predictions.csv')
reg_df   = pd.read_csv(OUTPUT_PATH+'regression_model_comparison.csv')
clf_df   = pd.read_csv(OUTPUT_PATH+'classification_model_comparison.csv')
cv_df    = pd.read_csv(OUTPUT_PATH+'cross_validation_results.csv')
fi_df    = pd.read_csv(OUTPUT_PATH+'feature_importance.csv')
mi       = joblib.load(MODEL_PATH+'model_info.pkl')
best_clf = joblib.load(MODEL_PATH+'best_stockout_model.pkl')
raw_train= pd.read_excel("data/Warehouse_Demand_Realistic_v2.xlsx",sheet_name="Train_80pct",header=1)
for c in ['Month_Number','Year','Actual_Demand (cylinders)','Stockout_Occurred']:
    raw_train[c] = pd.to_numeric(raw_train[c], errors='coerce')
y_test  = pd.read_csv(OUTPUT_PATH+'y_test_stock.csv').squeeze().astype(int)
X_test  = pd.read_csv(OUTPUT_PATH+'X_test.csv')

# Chart 1: Regression R2
fig,ax=plt.subplots(figsize=(9,5))
bars=ax.barh(reg_df['Model'],reg_df['R2_Score'],color=COLORS[:len(reg_df)],height=0.5)
ax.set_xlim(0,1.12); ax.set_xlabel('R² Score'); ax.set_title('Regression Model Comparison — R² Score',fontsize=13,fontweight='bold')
for bar,val in zip(bars,reg_df['R2_Score']): ax.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=10,fontweight='bold')
ax.axvline(0.9,color='red',linestyle='--',alpha=0.5,label='Target 0.90'); ax.legend()
plt.tight_layout(); plt.savefig(CHART_PATH+'chart1_regression_comparison.png',bbox_inches='tight'); plt.close()
print("  Saved chart1_regression_comparison.png")

# Chart 2: Classification F1 + AUC
fig,axes=plt.subplots(1,2,figsize=(13,5))
ax=axes[0]; bars=ax.barh(clf_df['Model'],clf_df['F1_Score'],color=COLORS[:len(clf_df)],height=0.5)
ax.set_xlim(0,1.1); ax.set_title('Classification — F1 Score',fontweight='bold')
for bar,val in zip(bars,clf_df['F1_Score']): ax.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=9)
ax=axes[1]; bars2=ax.barh(clf_df['Model'],clf_df['AUC_ROC'],color=COLORS[:len(clf_df)],height=0.5)
ax.set_xlim(0,1.1); ax.set_title('Classification — AUC-ROC',fontweight='bold')
for bar,val in zip(bars2,clf_df['AUC_ROC']): ax.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=9)
ax.axvline(0.5,color='red',linestyle='--',alpha=0.5,label='Random baseline'); ax.legend(fontsize=9)
plt.suptitle('Classification Model Comparison',fontsize=13,fontweight='bold'); plt.tight_layout()
plt.savefig(CHART_PATH+'chart2_classification_comparison.png',bbox_inches='tight'); plt.close()
print("  Saved chart2_classification_comparison.png")

# Chart 3: Actual vs Predicted
fig,ax=plt.subplots(figsize=(7,7))
ax.scatter(pred_df['Actual_Demand'],pred_df['Predicted_Demand'],alpha=0.4,color='#1A5276',s=20)
lims=[pred_df['Actual_Demand'].min()-0.1,pred_df['Actual_Demand'].max()+0.1]
ax.plot(lims,lims,'r--',linewidth=1.5,label='Perfect Prediction')
ax.set_xlabel('Actual Demand (cylinders)'); ax.set_ylabel('Predicted Demand (cylinders)')
ax.set_title(f'Actual vs Predicted Demand\n{mi["best_regression_model"]} R²={mi["regression_r2"]:.4f}',fontsize=12,fontweight='bold')
ax.legend(); plt.tight_layout(); plt.savefig(CHART_PATH+'chart3_actual_vs_predicted.png',bbox_inches='tight'); plt.close()
print("  Saved chart3_actual_vs_predicted.png")

# Chart 4: Error distribution
errors=pred_df['Actual_Demand']-pred_df['Predicted_Demand']
fig,ax=plt.subplots(figsize=(9,5))
ax.hist(errors,bins=40,color='#1A5276',alpha=0.75,edgecolor='white')
ax.axvline(0,color='red',linestyle='--',linewidth=1.5,label='Zero Error')
ax.axvline(errors.mean(),color='orange',linestyle='--',linewidth=1.5,label=f'Mean={errors.mean():.4f}')
ax.set_xlabel('Error (Actual − Predicted)'); ax.set_ylabel('Frequency')
ax.set_title('Prediction Error Distribution',fontsize=12,fontweight='bold'); ax.legend()
plt.tight_layout(); plt.savefig(CHART_PATH+'chart4_error_distribution.png',bbox_inches='tight'); plt.close()
print("  Saved chart4_error_distribution.png")

# Chart 5: Monthly demand trend
monthly=raw_train.groupby('Month_Number')['Actual_Demand (cylinders)'].mean().reset_index()
sc={'Winter':'#5DADE2','Summer':'#F39C12','Monsoon':'#27AE60','Autumn':'#E67E22'}
colors=[sc[SEASON_MAP[m]] for m in monthly['Month_Number']]
fig,ax=plt.subplots(figsize=(11,5))
bars=ax.bar([MONTH_NAMES[m-1] for m in monthly['Month_Number']],monthly['Actual_Demand (cylinders)'],color=colors,edgecolor='white',width=0.6)
for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f'{bar.get_height():.2f}',ha='center',va='bottom',fontsize=8)
ax.set_title('Average Monthly LPG Demand by Month',fontsize=13,fontweight='bold')
ax.set_ylabel('Avg Demand (cylinders)')
legend_els=[mpatches.Patch(color=c,label=s) for s,c in sc.items()]; ax.legend(handles=legend_els,fontsize=9)
plt.tight_layout(); plt.savefig(CHART_PATH+'chart5_monthly_demand_trend.png',bbox_inches='tight'); plt.close()
print("  Saved chart5_monthly_demand_trend.png")

# Chart 6: Feature importance
top_fi=fi_df.head(12).sort_values('Importance')
fig,ax=plt.subplots(figsize=(9,6))
bars=ax.barh(top_fi['Feature'],top_fi['Importance'],color='#1A5276',height=0.6)
for bar,val in zip(bars,top_fi['Importance']): ax.text(bar.get_width()+0.002,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=9)
ax.set_title(f'Feature Importance — {mi["best_regression_model"]}',fontsize=12,fontweight='bold'); ax.set_xlabel('Importance Score')
plt.tight_layout(); plt.savefig(CHART_PATH+'chart6_feature_importance.png',bbox_inches='tight'); plt.close()
print("  Saved chart6_feature_importance.png")

# Chart 7: Stockout by zone
zone_so=raw_train.groupby('Warehouse_Zone')['Stockout_Occurred'].apply(lambda x: pd.to_numeric(x,errors='coerce').mean()*100).sort_values(ascending=False)
avg=zone_so.mean()
fig,ax=plt.subplots(figsize=(11,5))
bar_colors=['#C0392B' if v>avg else '#1E8449' for v in zone_so.values]
bars=ax.bar(zone_so.index,zone_so.values,color=bar_colors,edgecolor='white',width=0.6)
ax.axhline(avg,color='orange',linestyle='--',linewidth=1.5,label=f'Avg {avg:.1f}%')
for bar,val in zip(bars,zone_so.values): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.15,f'{val:.1f}%',ha='center',va='bottom',fontsize=9,fontweight='bold')
ax.set_title('Stockout Rate by Warehouse Zone',fontsize=12,fontweight='bold'); ax.set_ylabel('Stockout Rate (%)')
ax.set_xticklabels(zone_so.index,rotation=28,ha='right'); ax.legend()
plt.tight_layout(); plt.savefig(CHART_PATH+'chart7_stockout_by_zone.png',bbox_inches='tight'); plt.close()
print("  Saved chart7_stockout_by_zone.png")

# Chart 8: Confusion matrix
cm=confusion_matrix(y_test,pred_df['Predicted_Stockout'])
fig,ax=plt.subplots(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,xticklabels=['No Stockout','Stockout'],yticklabels=['No Stockout','Stockout'],annot_kws={'size':14,'weight':'bold'})
ax.set_title(f'Confusion Matrix — {mi["best_classification_model"]}\n(threshold={mi["stockout_threshold"]})',fontsize=12,fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout(); plt.savefig(CHART_PATH+'chart8_confusion_matrix.png',bbox_inches='tight'); plt.close()
print("  Saved chart8_confusion_matrix.png")

# Chart 9: Cross-validation comparison
fig,axes=plt.subplots(1,2,figsize=(13,5))
reg_cv=cv_df[cv_df['Task']=='Regression'].sort_values('CV_Mean',ascending=True)
clf_cv=cv_df[cv_df['Task']=='Classification'].sort_values('CV_Mean',ascending=True)
for ax2,df2,title,color in [(axes[0],reg_cv,'Regression CV R²','#1A5276'),(axes[1],clf_cv,'Classification CV F1','#1E8449')]:
    bars=ax2.barh(df2['Model'],df2['CV_Mean'],xerr=df2['CV_Std'],color=color,height=0.5,capsize=4,alpha=0.85)
    ax2.set_xlim(0,1.1); ax2.set_title(f'5-Fold Cross-Validation\n{title}',fontweight='bold')
    for bar,val,std in zip(bars,df2['CV_Mean'],df2['CV_Std']):
        ax2.text(bar.get_width()+0.02,bar.get_y()+bar.get_height()/2,f'{val:.4f}±{std:.4f}',va='center',fontsize=8)
plt.suptitle('Cross-Validation Results (Mean ± Std)',fontsize=13,fontweight='bold'); plt.tight_layout()
plt.savefig(CHART_PATH+'chart9_cross_validation.png',bbox_inches='tight'); plt.close()
print("  Saved chart9_cross_validation.png")

# Chart 10: ROC curve + Precision-Recall curve
probs=best_clf.predict_proba(X_test)[:,1]
fpr,tpr,_=roc_curve(y_test,probs)
prec_c,rec_c,_=precision_recall_curve(y_test,probs)
fig,axes=plt.subplots(1,2,figsize=(12,5))
ax=axes[0]; ax.plot(fpr,tpr,color='#1A5276',lw=2,label=f'AUC={mi["classification_auc_roc"]:.4f}')
ax.plot([0,1],[0,1],'r--',lw=1,label='Random'); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Stockout Prediction',fontweight='bold'); ax.legend()
ax=axes[1]; ax.plot(rec_c,prec_c,color='#1E8449',lw=2,label=f'Avg Precision={mi["classification_avg_prec"]:.4f}')
ax.axhline(y_test.mean(),color='r',linestyle='--',lw=1,label=f'Baseline={y_test.mean():.3f}')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-Recall Curve',fontweight='bold'); ax.legend()
plt.suptitle(f'{mi["best_classification_model"]} — Stockout Detection',fontsize=13,fontweight='bold'); plt.tight_layout()
plt.savefig(CHART_PATH+'chart10_roc_pr_curves.png',bbox_inches='tight'); plt.close()
print("  Saved chart10_roc_pr_curves.png")

print("\n"+"="*65)
print("  ALL 10 CHARTS SAVED!"); print("  Next: Run step4_predict_new.py")
print("="*65)
