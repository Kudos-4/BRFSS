# ============================================================
# Test: load the cleaned BRFSS 2024 data and make a working copy
# ============================================================
# BRFSS.py exports the cleaned data to BRFSS_Cleaned/.
# Here we load it back as `BRFSS_24_cleaned`, then make a copy
# into `df` so downstream edits don't touch the original.
# ============================================================

import pandas as pd

# Relative to the repo root (portable across PyCharm and Google Colab).
CLEANED_PATH = "BRFSS_Cleaned/BRFSS_2024_cleaned.parquet"

# Load the cleaned dataset
BRFSS_24_cleaned = pd.read_parquet(CLEANED_PATH)

# Make a working copy (the line you wanted to implement)
df = BRFSS_24_cleaned.copy()

# -- Quick checks --------------------------------------------
print(f"Loaded BRFSS_24_cleaned : {BRFSS_24_cleaned.shape[0]:,} rows x {BRFSS_24_cleaned.shape[1]} cols")
print(f"Working copy `df`        : {df.shape[0]:,} rows x {df.shape[1]} cols")

# Confirm it is a real copy, not a view of the original
assert df is not BRFSS_24_cleaned, "df should be a separate object"
assert df.equals(BRFSS_24_cleaned), "df should hold the same data"
print("OK: df is an independent copy with identical contents")

print("\n" + df.head().to_string())
print("\nShape:", df.shape)