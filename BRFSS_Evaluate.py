# ============================================================
# BRFSS 2024 -- STEP 8: EVALUATION OF MODEL PERFORMANCE
# ============================================================
# Loads saved models and the held-out test set, then produces:
#   - Accuracy, ROC-AUC, F1 comparison table
#   - ROC curves (all models, one plot)
#   - Precision-Recall curves (all models, one plot)
#   - Confusion matrix heatmaps (one per model)
#   - Metric bar-chart summary
# Outputs saved to BRFSS_Evaluation/.
# ============================================================

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay,
)

# -- Paths ---------------------------------------------------
CLEANED_PATH  = "BRFSS_Cleaned/BRFSS_2024_cleaned.parquet"
MODELS_DIR    = "BRFSS_Models"
OUTPUT_DIR    = "BRFSS_Evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    'PHYSHLTH', 'EXERANY2', '_TOTINDA', '_SMOKER3', '_DRNKWK3',
    '_BMI5', '_BMI5CAT', '_AGE80', 'SEXVAR', 'EDUCA', 'INCOME3',
    'EMPLOY1', '_HLTHPL2', 'PERSDOC3', 'CHECKUP1',
]
TARGET = 'mental_risk'

MODEL_NAMES = [
    'Logistic_Regression',
    'Decision_Tree',
    'Random_Forest',
    'XGBoost',
]
COLORS = ['royalblue', 'darkorange', 'forestgreen', 'crimson']

# -- 1. Reconstruct test set ---------------------------------
print("=" * 70)
print("BRFSS 2024 -- STEP 8: EVALUATION OF MODEL PERFORMANCE")
print("=" * 70)

df = pd.read_parquet(CLEANED_PATH)

with open(f"{MODELS_DIR}/train_test_split.pkl", 'rb') as f:
    split = pickle.load(f)

test_idx = split['test_index']
X_test   = df.loc[test_idx, FEATURES]
y_test   = df.loc[test_idx, TARGET]
print(f"\nTest set : {len(X_test):,} rows  "
      f"({y_test.sum():,} at-risk, {(y_test==0).sum():,} no-risk)")

# -- 2. Load models and collect predictions ------------------
pipelines  = {}
y_preds    = {}
y_probs    = {}

for name in MODEL_NAMES:
    path = f"{MODELS_DIR}/{name}.pkl"
    with open(path, 'rb') as f:
        pipelines[name] = pickle.load(f)
    y_preds[name] = pipelines[name].predict(X_test)
    y_probs[name] = pipelines[name].predict_proba(X_test)[:, 1]
    print(f"  Loaded : {name}")

# -- 3. Metrics table ----------------------------------------
print("\n" + "=" * 70)
print("PERFORMANCE METRICS ON TEST SET")
print("=" * 70)

rows = []
for name in MODEL_NAMES:
    yp  = y_preds[name]
    ypr = y_probs[name]
    cm  = confusion_matrix(y_test, yp)
    tn, fp, fn, tp = cm.ravel()
    rows.append({
        'Model':       name.replace('_', ' '),
        'Accuracy':    round(accuracy_score(y_test, yp),       4),
        'ROC-AUC':     round(roc_auc_score(y_test, ypr),       4),
        'Avg-PR-AUC':  round(average_precision_score(y_test, ypr), 4),
        'F1 (At Risk)':round(f1_score(y_test, yp),             4),
        'Recall (AR)': round(tp / (tp + fn),                   4),
        'Precision(AR)':round(tp / (tp + fp),                  4),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    })

metrics_df = pd.DataFrame(rows).set_index('Model')
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics_table.csv")

display_cols = ['Accuracy', 'ROC-AUC', 'Avg-PR-AUC',
                'F1 (At Risk)', 'Recall (AR)', 'Precision(AR)']
print(metrics_df[display_cols].to_string())

best_auc = metrics_df['ROC-AUC'].idxmax()
best_f1  = metrics_df['F1 (At Risk)'].idxmax()
print(f"\n  Best ROC-AUC      : {best_auc}  ({metrics_df.loc[best_auc, 'ROC-AUC']})")
print(f"  Best F1 (At Risk) : {best_f1}   ({metrics_df.loc[best_f1, 'F1 (At Risk)']})")

# -- 4. ROC curves -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name, color in zip(MODEL_NAMES, COLORS):
    fpr, tpr, _ = roc_curve(y_test, y_probs[name])
    auc_val      = roc_auc_score(y_test, y_probs[name])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name.replace('_',' ')}  (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves — All Models', fontsize=13)
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_roc_curves.png", dpi=150)
plt.close()
print("\n  Saved: 01_roc_curves.png")

# -- 5. Precision-Recall curves ------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
baseline = y_test.mean()
ax.axhline(baseline, color='k', linestyle='--', lw=1,
           label=f'Random baseline ({baseline:.3f})')

for name, color in zip(MODEL_NAMES, COLORS):
    prec, rec, _ = precision_recall_curve(y_test, y_probs[name])
    ap_val        = average_precision_score(y_test, y_probs[name])
    ax.plot(rec, prec, color=color, lw=2,
            label=f"{name.replace('_',' ')}  (AP = {ap_val:.3f})")

ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curves — All Models', fontsize=13)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_precision_recall_curves.png", dpi=150)
plt.close()
print("  Saved: 02_precision_recall_curves.png")

# -- 6. Confusion matrices -----------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, color) in enumerate(zip(MODEL_NAMES, COLORS)):
    cm  = confusion_matrix(y_test, y_preds[name])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_preds[name])
    auc = roc_auc_score(y_test, y_probs[name])

    disp = ConfusionMatrixDisplay(cm, display_labels=['No Risk', 'At Risk'])
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(
        f"{name.replace('_', ' ')}\nAcc={acc:.3f}  AUC={auc:.3f}",
        fontsize=10
    )

plt.suptitle('BRFSS 2024 — Confusion Matrices (Test Set)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_confusion_matrices.png")

# -- 7. Metric comparison bar chart --------------------------
plot_metrics = ['Accuracy', 'ROC-AUC', 'F1 (At Risk)', 'Recall (AR)']
n_metrics    = len(plot_metrics)
n_models     = len(MODEL_NAMES)
x            = np.arange(n_metrics)
width        = 0.18

fig, ax = plt.subplots(figsize=(11, 5))
for i, (name, color) in enumerate(zip(MODEL_NAMES, COLORS)):
    vals = [metrics_df.loc[name.replace('_', ' '), m] for m in plot_metrics]
    bars = ax.bar(x + i * width, vals, width, label=name.replace('_', ' '),
                  color=color, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xticks(x + width * (n_models - 1) / 2)
ax.set_xticklabels(plot_metrics, fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title('BRFSS 2024 — Model Performance Comparison (Test Set)', fontsize=13)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_metric_comparison.png", dpi=150)
plt.close()
print("  Saved: 04_metric_comparison.png")

# -- 8. Detailed classification reports ----------------------
report_path = f"{OUTPUT_DIR}/classification_reports.txt"
with open(report_path, 'w') as f:
    f.write("BRFSS 2024 — Classification Reports (Test Set)\n")
    f.write("=" * 70 + "\n\n")
    for name in MODEL_NAMES:
        f.write(f"[{name}]\n")
        f.write(classification_report(
            y_test, y_preds[name],
            target_names=['No Risk', 'At Risk']
        ))
        f.write("\n")
print(f"  Saved: classification_reports.txt")

# -- 9. Final summary ----------------------------------------
print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"\n  Output directory : {OUTPUT_DIR}/")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"    {fname:<45}  {size/1e3:7.1f} KB")

print("\nDONE — proceed to Step 9 (BRFSS_Interpret.py) for feature importance.\n")
