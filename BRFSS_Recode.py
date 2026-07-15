# ============================================================
# BRFSS 2024 -- STEP 5: INSPECT UNIQUE VALUES + RECODE TO BINARY
# ============================================================
# Implements the five prep steps requested by the PI:
#   1. Count unique values in each variable.
#   2. List what those unique values are.
#   3. Check for remaining NaN; drop any that are found.
#   4. Create binary (0/1) variables by recoding, using the
#      BRFSS 2024 codebook labels for the "yes" side.
#   5. Collapse multi-category variables into binary where it
#      makes analytical sense (smoker, obese, income, etc.).
#
# Input : BRFSS_Cleaned/BRFSS_2024_cleaned.parquet  (from BRFSS.py)
# Output: BRFSS_Cleaned/BRFSS_2024_recoded.parquet + .csv
#
# All recodes below are documented against the codebook file
# USCODE24_LLCP_082125.HTML so the labels can be verified.
# ============================================================

import os
import pandas as pd

# -- Paths ---------------------------------------------------
CLEANED_PATH = "BRFSS_Cleaned/BRFSS_2024_cleaned.parquet"
OUTPUT_DIR   = "BRFSS_Cleaned"

# -- Human-readable label for each raw column (from codebook) -
VAR_LABELS = {
    'MENTHLTH': 'Poor mental health days (0-30)',
    'PHYSHLTH': 'Poor physical health days (0-30)',
    'EXERANY2': 'Any physical activity  (1=Yes, 2=No)',
    '_TOTINDA': 'Meets activity guidelines  (1=Yes, 2=No)',
    '_SMOKER3': 'Smoker status  (1=everyday, 2=someday, 3=former, 4=never)',
    '_DRNKWK3': 'Alcoholic drinks per week',
    '_BMI5':    'BMI x100',
    '_BMI5CAT': 'BMI category  (1=under, 2=normal, 3=over, 4=obese)',
    '_AGE80':   'Age (18-80)',
    'SEXVAR':   'Sex  (1=Male, 2=Female)',
    'EDUCA':    'Education  (1..6, 6=college grad)',
    'INCOME3':  'Income bracket  (1=<$10k .. 11=$200k+)',
    'EMPLOY1':  'Employment  (1=wages,2=self,3-4=out of work,5=homemaker,6=student,7=retired,8=unable)',
    '_HLTHPL2': 'Health insurance  (1=Have, 2=None)',
    'PERSDOC3': 'Personal doctor  (1=one, 2=>one, 3=No)',
    'CHECKUP1': 'Last checkup  (1=<1yr, 2=<2yr, 3=<5yr, 4=5yr+, 8=never)',
    'mental_risk': 'TARGET: mental health risk (MENTHLTH>=14)',
}

print("=" * 70)
print("BRFSS 2024 -- STEP 5: INSPECT UNIQUE VALUES + RECODE TO BINARY")
print("=" * 70)

# -- Load: keep the naming the PI's workflow uses ------------
BRFSS_24_cleaned = pd.read_parquet(CLEANED_PATH)
df = BRFSS_24_cleaned.copy()          # working copy; original untouched
print(f"\nLoaded : {df.shape[0]:,} rows x {df.shape[1]} columns")

# ============================================================
# STEP 1 -- how many unique values in each variable
# ============================================================
print("\n" + "-" * 70)
print("STEP 1 -- Number of unique values per variable")
print("-" * 70)
nuniq = df.nunique().sort_values()
for col, k in nuniq.items():
    print(f"  {col:<12} {k:>6} unique   | {VAR_LABELS.get(col, '')}")

# ============================================================
# STEP 2 -- what those unique values are
# ============================================================
print("\n" + "-" * 70)
print("STEP 2 -- The unique values themselves")
print("-" * 70)
for col in df.columns:
    vals = sorted(df[col].dropna().unique().tolist())
    if len(vals) <= 12:
        print(f"  {col:<12} {vals}")
    else:
        print(f"  {col:<12} min={min(vals):g}  max={max(vals):g}  "
              f"({len(vals)} distinct values, continuous)")

# ============================================================
# STEP 3 -- any missing (NaN) values left? drop if so
# ============================================================
print("\n" + "-" * 70)
print("STEP 3 -- Remaining missing (NaN) values")
print("-" * 70)
na = df.isna().sum()
na = na[na > 0]
if na.empty:
    print("  No missing values found -- nothing to drop.")
else:
    print("  Missing values per column:")
    for col, cnt in na.items():
        print(f"    {col:<12} {cnt:,}")
    before = len(df)
    df = df.dropna()
    print(f"  Dropped {before - len(df):,} rows -> {len(df):,} remain.")

# ============================================================
# STEP 4 & 5 -- recode to binary (0/1)
# ============================================================
# Each entry maps a NEW binary column -> (source column, set of
# source codes that become 1). Every other valid code becomes 0.
# The "->1" side matches the codebook "Yes"/positive label.
print("\n" + "-" * 70)
print("STEP 4 & 5 -- Create binary (0/1) variables")
print("-" * 70)

BINARY_RECODES = {
    # new column          source       codes that map to 1   meaning of "1"
    'exercise':          ('EXERANY2', {1},          'did any physical activity'),
    'meets_activity':    ('_TOTINDA', {1},          'meets activity guidelines'),
    'current_smoker':    ('_SMOKER3', {1, 2},       'smokes every/some days'),   # multi -> binary
    'obese':             ('_BMI5CAT', {4},          'BMI category = obese'),     # multi -> binary
    'female':            ('SEXVAR',   {2},          'female'),
    'insured':           ('_HLTHPL2', {1},          'has health insurance'),
    'has_pers_doctor':   ('PERSDOC3', {1, 2},       'has >=1 personal doctor'),  # multi -> binary
    'checkup_within_1yr':('CHECKUP1', {1},          'checkup within past year'), # multi -> binary
    'college_grad':      ('EDUCA',    {6},          'college graduate (4+ yrs)'),# multi -> binary
    'employed':          ('EMPLOY1',  {1, 2},       'employed for wages / self'),# multi -> binary
    'low_income':        ('INCOME3',  {1, 2, 3, 4}, 'household income < $25,000'),# multi -> binary
}

for new_col, (src, ones, meaning) in BINARY_RECODES.items():
    df[new_col] = df[src].isin(ones).astype(int)
    n1 = int(df[new_col].sum())
    pct = n1 / len(df) * 100
    codes = '{' + ','.join(str(c) for c in sorted(ones)) + '}'
    print(f"  {new_col:<20} <- {src:<9} 1 if in {codes:<11} ({meaning})")
    print(f"  {'':<20}    -> 1 = {n1:,} ({pct:.1f}%) | 0 = {len(df)-n1:,} ({100-pct:.1f}%)")

# -- Verify recodes: crosstab source vs new (proves mapping) --
print("\n" + "-" * 70)
print("VERIFY -- source code vs new binary (should be a clean split)")
print("-" * 70)
for new_col, (src, ones, meaning) in BINARY_RECODES.items():
    ct = pd.crosstab(df[src], df[new_col])
    ct.columns = [f'{new_col}=0', f'{new_col}=1']
    print(f"\n  {src} -> {new_col}")
    print(ct.to_string().replace('\n', '\n  '))

# ============================================================
# Save enriched dataset
# ============================================================
print("\n" + "=" * 70)
print("SAVE")
print("=" * 70)
os.makedirs(OUTPUT_DIR, exist_ok=True)
parquet_out = f"{OUTPUT_DIR}/BRFSS_2024_recoded.parquet"
csv_out     = f"{OUTPUT_DIR}/BRFSS_2024_recoded.csv"
df.to_parquet(parquet_out, index=False)
df.to_csv(csv_out, index=False)

print(f"  Rows x cols   : {df.shape[0]:,} x {df.shape[1]}")
print(f"  Binary cols   : {', '.join(BINARY_RECODES.keys())}")
print(f"  Saved parquet : {parquet_out}  ({os.path.getsize(parquet_out)/1e6:.1f} MB)")
print(f"  Saved CSV     : {csv_out}  ({os.path.getsize(csv_out)/1e6:.1f} MB)")
print("\nDONE\n")
print(df.head().to_string())
