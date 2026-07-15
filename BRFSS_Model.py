# ============================================================
# BRFSS 2024 -- STEP 7: MODELING
# ============================================================
# Trains four classifiers on the cleaned BRFSS data:
#   1. Logistic Regression  (baseline)
#   2. Decision Tree
#   3. Random Forest        (highly recommended)
#   4. XGBoost
#
# Outputs saved to BRFSS_Models/:
#   - Trained model objects (.pkl)
#   - Train/test indices
#   - Per-model training summary
# ============================================================

import os
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)

# -- Paths ---------------------------------------------------
CLEANED_PATH = "BRFSS_Cleaned/BRFSS_2024_cleaned.parquet"
OUTPUT_DIR   = "BRFSS_Models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Feature sets --------------------------------------------
# MENTHLTH is excluded: it directly encodes the target (>=14 -> mental_risk=1)
# and including it would be data leakage.
FEATURES = [
    'PHYSHLTH',   # poor physical health days
    'EXERANY2',   # any physical activity (binary)
    '_TOTINDA',   # meets activity guidelines (binary)
    '_SMOKER3',   # smoking status (ordinal 1-4)
    '_DRNKWK3',   # drinks per week (continuous)
    '_BMI5',      # BMI x100 (continuous)
    '_BMI5CAT',   # BMI category (ordinal 1-4)
    '_AGE80',     # age (continuous)
    'SEXVAR',     # sex (binary 1-2)
    'EDUCA',      # education (ordinal 1-6)
    'INCOME3',    # income (ordinal 1-11)
    'EMPLOY1',    # employment (categorical 1-8)
    '_HLTHPL2',   # health insurance (binary 1-2)
    'PERSDOC3',   # personal doctor (categorical 1-3)
    'CHECKUP1',   # last checkup (categorical 1-8)
]
TARGET = 'mental_risk'

# Continuous features get StandardScaler; the rest pass through.
SCALE_COLS = ['PHYSHLTH', '_DRNKWK3', '_BMI5', '_AGE80']
PASS_COLS  = [c for c in FEATURES if c not in SCALE_COLS]

# -- 1. Load -------------------------------------------------
print("=" * 70)
print("BRFSS 2024 -- STEP 7: MODELING")
print("=" * 70)

df = pd.read_parquet(CLEANED_PATH)
n  = len(df)
print(f"\nLoaded : {n:,} rows x {df.shape[1]} columns")

X = df[FEATURES]
y = df[TARGET]

# Class balance info
n1 = y.sum()
n0 = n - n1
ratio = n0 / n1
print(f"\nClass balance  : {n0:,} no-risk  ({n0/n*100:.1f}%) "
      f"| {n1:,} at-risk ({n1/n*100:.1f}%)")
print(f"scale_pos_weight for XGBoost : {ratio:.2f}")

# -- 2. Train / test split (80 / 20, stratified) -------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train):,} rows")
print(f"Test  : {len(X_test):,}  rows")

# Save split indices for reproducibility
split_info = {
    'train_index': X_train.index.tolist(),
    'test_index':  X_test.index.tolist(),
}
with open(f"{OUTPUT_DIR}/train_test_split.pkl", 'wb') as f:
    pickle.dump(split_info, f)
print(f"\nSplit indices saved: {OUTPUT_DIR}/train_test_split.pkl")

# -- 3. Preprocessor -----------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ('scale', StandardScaler(), SCALE_COLS),
    ('pass',  'passthrough',    PASS_COLS),
], remainder='drop')

# -- 4. Model definitions ------------------------------------
# class_weight='balanced' compensates for 86/14 imbalance in LR, DT, RF.
# XGBoost uses scale_pos_weight instead.

models = {
    'Logistic_Regression': Pipeline([
        ('prep',  preprocessor),
        ('model', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1,
        )),
    ]),
    'Decision_Tree': Pipeline([
        ('prep',  preprocessor),
        ('model', DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=8,
            min_samples_leaf=50,
            random_state=42,
        )),
    ]),
    'Random_Forest': Pipeline([
        ('prep',  preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            max_depth=12,
            min_samples_leaf=20,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
        )),
    ]),
    'XGBoost': Pipeline([
        ('prep',  preprocessor),
        ('model', XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}

# -- 5. Train & evaluate each model --------------------------
print("\n" + "=" * 70)
print("TRAINING  (train metrics shown; test metrics are for reporting only)")
print("=" * 70)

results = []

for name, pipeline in models.items():
    print(f"\n[{name}]")
    t0 = time.time()

    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Fit time : {elapsed:.1f}s")

    # --- Train metrics
    y_train_pred = pipeline.predict(X_train)
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]
    train_acc    = accuracy_score(y_train, y_train_pred)
    train_auc    = roc_auc_score(y_train, y_train_prob)
    print(f"  Train Acc : {train_acc:.4f}   Train AUC : {train_auc:.4f}")

    # --- Test metrics
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_acc    = accuracy_score(y_test, y_test_pred)
    test_auc    = roc_auc_score(y_test, y_test_prob)
    print(f"  Test  Acc : {test_acc:.4f}   Test  AUC : {test_auc:.4f}")

    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  Confusion matrix (test):")
    print(f"    TN={cm[0,0]:>6,}  FP={cm[0,1]:>6,}")
    print(f"    FN={cm[1,0]:>6,}  TP={cm[1,1]:>6,}")

    report = classification_report(y_test, y_test_pred,
                                   target_names=['No Risk', 'At Risk'])
    print(f"  Classification report (test):")
    for line in report.strip().split('\n'):
        print(f"    {line}")

    results.append({
        'model':      name,
        'fit_time_s': round(elapsed, 2),
        'train_acc':  round(train_acc, 4),
        'train_auc':  round(train_auc, 4),
        'test_acc':   round(test_acc, 4),
        'test_auc':   round(test_auc, 4),
        'test_tn':    int(cm[0, 0]),
        'test_fp':    int(cm[0, 1]),
        'test_fn':    int(cm[1, 0]),
        'test_tp':    int(cm[1, 1]),
    })

    # Save model
    model_path = f"{OUTPUT_DIR}/{name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  Saved : {model_path}")

# -- 6. Comparison summary -----------------------------------
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results).set_index('model')
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")

print(results_df[['train_acc', 'train_auc', 'test_acc', 'test_auc',
                   'fit_time_s']].to_string())

best_auc = results_df['test_auc'].idxmax()
best_acc = results_df['test_acc'].idxmax()
print(f"\n  Best Test AUC : {best_auc}  ({results_df.loc[best_auc, 'test_auc']:.4f})")
print(f"  Best Test Acc : {best_acc}  ({results_df.loc[best_acc, 'test_acc']:.4f})")

# -- 7. Output listing ---------------------------------------
print("\n" + "-" * 70)
print(f"Files saved to {OUTPUT_DIR}/")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"  {fname:<40}  {size/1e6:6.1f} MB")

print("\nDONE — proceed to Step 8 (BRFSS_Evaluate.py) for full evaluation.\n")
