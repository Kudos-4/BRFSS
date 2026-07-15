# ============================================================
# BRFSS 2024 -- DATA CLEANING
# ============================================================
# Loads only columns of interest, recodes sentinel "None" values,
# removes rows that are blank or carry a "no valid response" code
# (per the BRFSS 2024 codebook), then creates the binary target.
# SLEPTIM1 is absent from the 2024 core questionnaire.
# INCOME3 / _HLTHPL2 / PERSDOC3 replace prior-year names.
# ============================================================

import pandas as pd
import os

# -- Paths ----------------------------------------------------
# Paths are relative to the repo root (works in PyCharm and Google Colab alike).
PARQUET_PATH = "BRFSS_2024.parquet"
OUTPUT_DIR   = "BRFSS_Cleaned"

# -- Columns of interest + their "no valid response" codes ---
# Codes listed here trigger row removal (respondent gave no usable answer).
# 88 is NOT listed -- it means "None / zero days" and is recoded to 0 first.
INVALID_CODES = {
    'MENTHLTH': [77, 99],    # Days mental health not good  (88=None->0)
    'PHYSHLTH': [77, 99],    # Days physical health not good (88=None->0)
    'EXERANY2': [7, 9],      # Did any physical activity
    '_TOTINDA': [9],          # Meets activity guidelines
    '_SMOKER3': [9],          # Smoking status
    '_DRNKWK3': [99900],     # Drinks per week
    '_BMI5':    [],           # BMI x100 -- blank only, no numeric sentinel
    '_BMI5CAT': [],           # BMI category -- blank only
    '_AGE80':   [],           # Imputed age -- no refusal codes
    'SEXVAR':   [],           # Sex -- no refusal codes recorded
    'EDUCA':    [9],          # Education level
    'INCOME3':  [77, 99],    # Income (was INCOME2 in pre-2024 years)
    'EMPLOY1':  [9],          # Employment status
    '_HLTHPL2': [9],          # Health insurance (was HLTHPLN1 in pre-2024)
    'PERSDOC3': [7, 9],      # Personal doctor (was PERSDOC2 in pre-2024)
    'CHECKUP1': [7, 9],      # Last checkup  (8=Never is a valid answer)
}



COLUMNS = list(INVALID_CODES.keys())

# -- 1. Load --------------------------------------------------
print("=" * 70)
print("BRFSS 2024 -- DATA CLEANING")
print("=" * 70)

print(f"\nSource: {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH, columns=COLUMNS)
n_original = len(df)
print(f"Loaded : {n_original:,} rows x {df.shape[1]} columns")

# -- 2. Recode 88 -> 0 for days-count variables --------------
# 88 means the respondent had zero bad days -- it is valid data.
print("\n" + "-" * 70)
print("STEP 0 -- Recode 88 ('None / zero days') -> 0")
for col in ['MENTHLTH', 'PHYSHLTH']:
    n = (df[col] == 88).sum()
    df[col] = df[col].replace(88, 0)
    print(f"  {col}: {n:,} rows recoded 88 -> 0")

# -- 3. Drop rows with NaN in any column of interest ---------
print("\n" + "-" * 70)
print("STEP 1 -- Remove rows blank / not asked (NaN)")
before = len(df)
df = df.dropna(subset=COLUMNS)
n_after_nan = len(df)
removed_nan = before - n_after_nan
print(f"  Removed  : {removed_nan:,} rows")
print(f"  Remaining: {n_after_nan:,} rows  ({n_after_nan/n_original*100:.1f}% of original)")

# -- 4. Drop rows with "no valid response" codes -------------
print("\n" + "-" * 70)
print("STEP 2 -- Remove rows with 'no valid response' codes (per codebook)")
for col, bad in INVALID_CODES.items():
    if not bad:
        continue
    mask  = df[col].isin(bad)
    count = mask.sum()
    if count:
        df = df[~mask]
        print(f"  {col:<12}  codes {str(bad):<12}  removed {count:,} rows")

n_after_codes = len(df)
print(f"\n  Remaining: {n_after_codes:,} rows  ({n_after_codes/n_original*100:.1f}% of original)")

# -- 5. Create binary target ---------------------------------
df['mental_risk'] = (df['MENTHLTH'] >= 14).astype(int)

# -- 6. Summary ----------------------------------------------
print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)
print(f"  Original rows  : {n_original:,}")
print(f"  Rows dropped   : {n_original - n_after_codes:,}  ({(n_original - n_after_codes)/n_original*100:.1f}%)")
print(f"  Usable rows    : {n_after_codes:,}  ({n_after_codes/n_original*100:.1f}%)")
print(f"  Columns kept   : {df.shape[1]}")
print(f"  Columns        : {', '.join(df.columns.tolist())}")

c0 = (df['mental_risk'] == 0).sum()
c1 = (df['mental_risk'] == 1).sum()
print(f"\n  mental_risk=0  (< 14 bad days) : {c0:,}  ({c0/n_after_codes*100:.1f}%)")
print(f"  mental_risk=1  (>= 14 bad days): {c1:,}  ({c1/n_after_codes*100:.1f}%)")

# -- 7. Per-column value range check -------------------------
print("\n" + "-" * 70)
print("VALUE RANGES IN CLEANED DATA")
print("-" * 70)
for col in df.columns:
    series     = df[col]
    unique_cnt = series.nunique()
    if unique_cnt <= 12:
        print(f"  {col:<12}  unique values : {sorted(series.unique())}")
    else:
        print(f"  {col:<12}  min={series.min():.0f}  max={series.max():.0f}  distinct={unique_cnt}")

# -- 8. Export -----------------------------------------------
print("\n" + "-" * 70)
os.makedirs(OUTPUT_DIR, exist_ok=True)

parquet_out = f"{OUTPUT_DIR}/BRFSS_2024_cleaned.parquet"
csv_out     = f"{OUTPUT_DIR}/BRFSS_2024_cleaned.csv"

df.to_parquet(parquet_out, index=False)
df.to_csv(csv_out, index=False)

print(f"  Saved parquet : {parquet_out}  ({os.path.getsize(parquet_out)/1e6:.1f} MB)")
print(f"  Saved CSV     : {csv_out}  ({os.path.getsize(csv_out)/1e6:.1f} MB)")

print("\nDONE\n")
print(df.head().to_string())

