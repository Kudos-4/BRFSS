# PI's `brfss24.ipynb` vs. my `BRFSS_Full_Colab.ipynb`

This branch (`pi-brfss24`) holds Dr. Bhatta's Colab notebook so it can be diffed and reviewed
next to my pipeline without opening Colab.

| File | What it is |
|---|---|
| `brfss24.ipynb` | PI's notebook (126 original cells) + 16 review cells marked `NOTE (Jason)` / `added by Jason`. Outputs stripped. |
| `brfss24.py` | Same content as a `# %%`-cell script for reading/diffing in PyCharm. |
| `PI_vs_MINE_COMPARISON.md` | This document. |

The PI's notebook reads `gdrive/My Drive/BRFSS24_DilliNJason/brfss24.csv` (raw, 457,670 x 301) and
writes `brfss24_clean.csv` to the same Drive folder. It does not use this repo's parquet files.

## Headline

My notebook is a faithful port of the PyCharm pipeline and follows the guideline PDF
(`Research Topic 1- Guideline.pdf`): `MENTHLTH >= 14 -> mental_risk`, the ~16 suggested variables,
four models, ROC/accuracy/confusion-matrix evaluation. The PI's notebook (header: "Analysis completed,
prepare a manuscript", July 2026) is a **revised study design** that has moved well past that PDF.
The important differences are scientific choices, not code style: different variable sets on different samples.

## Side by side

| | PI `brfss24` | Mine `BRFSS_Full_Colab` |
|---|---|---|
| Data in | `brfss24.csv` from Google Drive | Clones this repo, reads `BRFSS_2024.parquet` |
| Target | `_MENT14D` (CDC calculated var) recoded 1,2 -> 0, 3 -> 1 | `MENTHLTH >= 14 -> mental_risk` (equivalent definition) |
| Predictors | **20**: `_AGE80 _SEX _RACEGR3 MARITAL _EDUCAG _INCOMG1 EMPLOY1 GENHLTH PHYSHLTH _BMI5CAT _TOTINDA _SMOKER3 MEDCOST1 PRIMINS2 PERSDOC3 ADDEPEV3 DIABETE4 ASTHMA3 CHCCOPD3 DIFFWALK` | **15**: `PHYSHLTH EXERANY2 _TOTINDA _SMOKER3 _DRNKWK3 _BMI5 _BMI5CAT _AGE80 SEXVAR EDUCA INCOME3 EMPLOY1 _HLTHPL2 PERSDOC3 CHECKUP1` |
| Shared predictors | 7: `_AGE80 PHYSHLTH _BMI5CAT _TOTINDA _SMOKER3 EMPLOY1 PERSDOC3` | |
| Cleaning | `drop_duplicates()` -> recode 7/9/77/99 to NaN -> `dropna()` -> **286,704** rows (229,363 train / 57,341 test) | recode 88 -> 0, drop NaN + invalid codes -> **~307k** rows, no de-dup |
| Categoricals | Collapsed (MARITAL -> 3, EMPLOY1 -> 4, PRIMINS2 -> 3, BMI -> 3); `C()` in statsmodels, `OneHotEncoder` in sklearn | 11 binary dummies built in Stage 2, but Stage 4 reads the *cleaned* file, so models see raw codes as numbers |
| Models | statsmodels logit (OR / CI / p) -> balanced LR -> tuned RF (500 trees, depth 10) -> tuned XGB (depth 3, lr 0.03). No Decision Tree. | LR, Decision Tree, RF (300, depth 12), XGB (depth 6, lr 0.05), all balanced; StandardScaler on continuous |
| Threshold | 0.5, 0.10, and Youden-optimal per model | 0.5 only |
| Evaluation | Accuracy, AUC, sensitivity/specificity; no plots originally | ROC + PR curves, confusion matrices, F1/AP, bar chart, saved PNGs |
| Interpretation | OR table + written paragraph | none yet |
| Results (test AUC) | Std LR ~0.79 (summary text), Balanced LR **0.831** (this run), RF ~0.78, XGB ~0.76 | LR 0.776, DT 0.775, RF 0.787, XGB 0.788 |

## What the PI's run shows (outputs from 2026-10-22 run, cells added by Jason)

Row accounting: 457,670 raw -> 441,358 after `drop_duplicates()` (-16,312) -> 286,704 after `dropna()`
(-154,654) = 62.6% of the raw file retained.

Duplicate check: 0 duplicate records by `(_STATE, SEQNO)`. The 16,312 rows removed by `drop_duplicates()`
are distinct respondents who gave identical answers on the 22 kept variables.

Balanced class weights on the training split: N0 = 196,731 (w = 0.583), N1 = 32,632 (w = 3.514);
a case gets ~6x the weight of a non-case.

Permutation importance (AUC drop, test-set sample of 20,000, 5 repeats):

| Variable | Random Forest | XGBoost |
|---|---|---|
| `ADDEPEV3` (ever diagnosed with depression) | 0.108 | 0.099 |
| `PHYSHLTH` (poor physical health days) | 0.033 | 0.036 |
| `_AGE80` (age) | 0.027 | 0.034 |
| `GENHLTH` (fair/poor self-rated health) | 0.009 | 0.007 |
| `MEDCOST1` (couldn't see doctor due to cost) | 0.008 | 0.007 |
| everything else | < 0.003 each | < 0.003 each |

Chronic conditions (`DIABETE4`, `ASTHMA3`, `CHCCOPD3`), `_BMI5CAT` and `PRIMINS2` add essentially nothing
once depression history and physical health are in the model.

## Issues flagged in the PI's notebook (each has a `NOTE (Jason)` cell in place)

1. `drop_duplicates()` on answer patterns deletes real respondents (confirmed: 0 true duplicate records).
2. `ASTHMA3` recode comments are flipped (1 = Yes in BRFSS); the code itself is correct.
3. `9999` is not a `_BMI5CAT` code; the line is harmless.
4. "RACE and CKD" crosstab comment and the N = 5479 CKD class-weight example are template leftovers.
5. Interpretation paragraph cites "American Indian/Alaska Native" and "Asian" groups, which do not exist in
   `_RACEGR3` (5 levels). Likely from a run with `_IMPRACE` / `_RACE`.
6. SMOTE / SMOTENC results in the final summary have no code behind them; an optional commented-out SMOTENC
   cell was added.
7. The "DO NOT RUN" concept cell raises a PatsyError on purpose and stops **Run all**; use *Runtime > Run after*
   from the next cell.
8. Final-summary AUC numbers (~0.79) are stale: the balanced LR scored 0.831 on this run.

## What my notebook should take from the PI's

- One-hot encode the multi-level categoricals (`EDUCA`, `INCOME3`, `EMPLOY1`, `PERSDOC3`, `CHECKUP1`) in
  Stage 4, or feed it the recoded dummies.
- Add a statsmodels `smf.logit` OR / CI / p-value table (needed for the manuscript).
- Add feature importance (impurity + permutation) for RF and XGB.
- Add Youden-optimal thresholds per model.
- Add a `FEATURES_PI` list with the PI's 20 variables and a `FEATURE_SET` knob so both designs run from one pipeline.

## Decision needed with the PI

Which variable set is the paper? The guideline PDF's list (mine) or the revised 20-variable design (his)?
Everything else above is mechanical once that is settled.
