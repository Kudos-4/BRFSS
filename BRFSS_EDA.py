# ============================================================
# BRFSS 2024 -- STEP 6: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# Loads cleaned data, examines distributions and correlations,
# and saves all plots / summaries to BRFSS_EDA/.
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# -- Paths ---------------------------------------------------
CLEANED_PATH = "BRFSS_Cleaned/BRFSS_2024_cleaned.parquet"
OUTPUT_DIR   = "BRFSS_EDA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Variable metadata ---------------------------------------
CONTINUOUS  = ['MENTHLTH', 'PHYSHLTH', '_BMI5', '_AGE80', '_DRNKWK3']
CATEGORICAL = ['EXERANY2', '_TOTINDA', '_SMOKER3', '_BMI5CAT',
               'SEXVAR', 'EDUCA', 'INCOME3', 'EMPLOY1',
               '_HLTHPL2', 'PERSDOC3', 'CHECKUP1']
TARGET = 'mental_risk'

VAR_LABELS = {
    'MENTHLTH' : 'Poor Mental Health Days (0-30)',
    'PHYSHLTH' : 'Poor Physical Health Days (0-30)',
    'EXERANY2' : 'Any Physical Activity',
    '_TOTINDA'  : 'Meets Activity Guidelines',
    '_SMOKER3'  : 'Smoking Status',
    '_DRNKWK3'  : 'Alcoholic Drinks / Week',
    '_BMI5'     : 'BMI (x100)',
    '_BMI5CAT'  : 'BMI Category',
    '_AGE80'    : 'Age',
    'SEXVAR'   : 'Sex (1=M, 2=F)',
    'EDUCA'    : 'Education Level',
    'INCOME3'  : 'Income Category',
    'EMPLOY1'  : 'Employment Status',
    '_HLTHPL2'  : 'Health Insurance',
    'PERSDOC3' : 'Personal Doctor',
    'CHECKUP1' : 'Last Medical Checkup',
    'mental_risk': 'Mental Health Risk',
}

# -- 1. Load -------------------------------------------------
print("=" * 70)
print("BRFSS 2024 -- STEP 6: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

df = pd.read_parquet(CLEANED_PATH)
n = len(df)
print(f"\nLoaded : {n:,} rows x {df.shape[1]} columns")
print(f"Columns: {', '.join(df.columns.tolist())}")

# -- 2. Summary statistics -----------------------------------
print("\n" + "-" * 70)
print("SUMMARY STATISTICS")
stats = df.describe(include='all').round(3)
stats.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv")
print(stats.to_string())

# -- 3. Target distribution ----------------------------------
print("\n" + "-" * 70)
print("TARGET DISTRIBUTION  (mental_risk)")
vc = df[TARGET].value_counts()
for label, code in [('No Risk  (0)', 0), ('At Risk  (1)', 1)]:
    cnt = vc.get(code, 0)
    print(f"  {label}: {cnt:,}  ({cnt/n*100:.1f}%)")

fig, ax = plt.subplots(figsize=(5, 4))
counts = [vc.get(0, 0), vc.get(1, 0)]
bars = ax.bar(['No Risk (0)', 'At Risk (1)'], counts,
              color=['steelblue', 'tomato'], edgecolor='white', width=0.5)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, cnt + n * 0.004,
            f'{cnt:,}\n({cnt/n*100:.1f}%)', ha='center', fontsize=9)
ax.set_title('Target: Mental Health Risk (≥14 poor days)')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png", dpi=150)
plt.close()
print("  Saved: 01_target_distribution.png")

# -- 4. Continuous variable distributions --------------------
print("\n" + "-" * 70)
print("CONTINUOUS VARIABLE DISTRIBUTIONS")

ncols = 3
nrows = (len(CONTINUOUS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
axes = axes.flatten()

for i, col in enumerate(CONTINUOUS):
    ax   = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(VAR_LABELS.get(col, col))
    ax.set_xlabel(VAR_LABELS.get(col, col))
    ax.set_ylabel('Count')
    ax.text(0.97, 0.95,
            f'n={len(data):,}\nmean={data.mean():.1f}\nmedian={data.median():.0f}\nstd={data.std():.1f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('BRFSS 2024 — Continuous Variable Distributions', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_continuous_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_continuous_distributions.png")

# -- 5. Categorical variable distributions -------------------
print("\n" + "-" * 70)
print("CATEGORICAL VARIABLE DISTRIBUTIONS")

ncols = 3
nrows = (len(CATEGORICAL) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
axes = axes.flatten()

for i, col in enumerate(CATEGORICAL):
    ax  = axes[i]
    vc2 = df[col].value_counts().sort_index()
    ax.bar(vc2.index.astype(str), vc2.values, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(VAR_LABELS.get(col, col))
    ax.set_xlabel('Code value')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', labelsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('BRFSS 2024 — Categorical Variable Distributions', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_categorical_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_categorical_distributions.png")

# -- 6. Pearson correlation matrix ---------------------------
print("\n" + "-" * 70)
print("CORRELATION MATRIX")

corr = df.corr(numeric_only=True)
corr.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={'size': 8})
ax.set_title('BRFSS 2024 — Pearson Correlation Matrix', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_correlation_matrix.png")

target_corr = (corr[TARGET]
               .drop(TARGET)
               .abs()
               .sort_values(ascending=False))
print(f"\n  Correlations with '{TARGET}' (absolute, descending):")
for feat, val in target_corr.items():
    sign = '+' if corr.loc[feat, TARGET] > 0 else '-'
    print(f"    {feat:<15}  r = {sign}{val:.4f}")

# -- 7. Feature distributions split by target ----------------
print("\n" + "-" * 70)
print("FEATURE DISTRIBUTIONS BY MENTAL HEALTH RISK")

features_compare = CONTINUOUS + ['_SMOKER3', 'EXERANY2', '_TOTINDA',
                                  '_BMI5CAT', 'SEXVAR', 'EDUCA', 'INCOME3',
                                  'EMPLOY1', '_HLTHPL2']

group0 = df[df[TARGET] == 0]
group1 = df[df[TARGET] == 1]

ncols = 3
nrows = (len(features_compare) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.8))
axes = axes.flatten()

for i, col in enumerate(features_compare):
    ax    = axes[i]
    label = VAR_LABELS.get(col, col)
    if col in CONTINUOUS:
        ax.hist(group0[col].dropna(), bins=30, alpha=0.65,
                color='steelblue', label='No Risk', density=True)
        ax.hist(group1[col].dropna(), bins=30, alpha=0.65,
                color='tomato',    label='At Risk',  density=True)
        ax.set_ylabel('Density')
    else:
        vc0  = group0[col].value_counts(normalize=True).sort_index()
        vc1  = group1[col].value_counts(normalize=True).sort_index()
        idx  = sorted(set(vc0.index) | set(vc1.index))
        x    = np.arange(len(idx))
        w    = 0.38
        ax.bar(x - w/2, [vc0.get(k, 0) for k in idx], w,
               color='steelblue', label='No Risk', alpha=0.85)
        ax.bar(x + w/2, [vc1.get(k, 0) for k in idx], w,
               color='tomato',    label='At Risk',  alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in idx], fontsize=8)
        ax.set_ylabel('Proportion')
    ax.set_title(label, fontsize=9)
    ax.legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('BRFSS 2024 — Feature Distributions by Mental Health Risk',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_features_by_target.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_features_by_target.png")

# -- 8. Pairplot of continuous features ----------------------
print("\n" + "-" * 70)
print("PAIRPLOT (continuous features)")

pair_cols = CONTINUOUS + [TARGET]
pair_sample = df[pair_cols].sample(n=min(5000, n), random_state=42)

pairfig = sns.pairplot(pair_sample, hue=TARGET, corner=True,
                       plot_kws={'alpha': 0.3, 's': 10},
                       palette={0: 'steelblue', 1: 'tomato'})
pairfig.figure.suptitle('BRFSS 2024 — Continuous Features Pairplot (n=5,000 sample)',
                         y=1.01, fontsize=12)
pairfig.savefig(f"{OUTPUT_DIR}/06_pairplot_continuous.png", dpi=120, bbox_inches='tight')
plt.close('all')
print("  Saved: 06_pairplot_continuous.png")

# -- 9. EDA summary ------------------------------------------
print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
print(f"  Output directory : {OUTPUT_DIR}/")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"    {fname:<48}  {size/1e3:7.1f} KB")
print("\nDONE\n")
