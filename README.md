# BRFSS 2024 — Predicting Mental-Health Risk

A machine-learning pipeline built on the **2024 Behavioral Risk Factor Surveillance System (BRFSS)** survey.
It cleans the raw survey, engineers binary features, explores the data, trains four classifiers, and
evaluates them on a single binary target:

> **`mental_risk` = 1** when a respondent reported **14 or more** poor-mental-health days in the past
> 30 (`MENTHLTH >= 14`), else **0**.

The project runs two ways from the same code and committed data:
- **Locally in PyCharm** (the primary development environment), and
- **In Google Colab** (for review) — see [Running in Google Colab](#running-in-google-colab).

---

## Pipeline

Each script is a standalone step that reads the previous step's output and writes into its own folder.
Run them **in order** from the repo root.

| Step | Script | Reads | Writes |
|------|--------|-------|--------|
| 1. Clean | `BRFSS.py` | `BRFSS_2024.parquet` (raw, ~16 cols) | `BRFSS_Cleaned/BRFSS_2024_cleaned.parquet` |
| 2. Recode | `BRFSS_Recode.py` | cleaned parquet | `BRFSS_Cleaned/BRFSS_2024_recoded.parquet` |
| 3. EDA | `BRFSS_EDA.py` | cleaned parquet | `BRFSS_EDA/` (plots + summary CSVs) |
| 4. Model | `BRFSS_Model.py` | cleaned parquet | `BRFSS_Models/` (trained `.pkl`, `model_comparison.csv`) |
| 5. Evaluate | `BRFSS_Evaluate.py` | cleaned parquet + models | `BRFSS_Evaluation/` (ROC/PR/confusion plots, metrics) |

**Cleaning** (`BRFSS.py`): loads the columns of interest, recodes `88` ("None / zero days") → `0` for the
health-day counts, drops blanks and per-codebook "no valid response" codes, then builds `mental_risk`
(~307k usable rows).

**Recoding** (`BRFSS_Recode.py`): the PI's five prep steps — inspect uniques, drop any remaining NaN,
and collapse categorical variables into 11 interpretable binary features (`current_smoker`, `obese`,
`insured`, `low_income`, `college_grad`, …).

**Modeling** (`BRFSS_Model.py`): Logistic Regression, Decision Tree, Random Forest, and XGBoost, all with
`class_weight`/scale balancing for the minority class. `MENTHLTH` is excluded from the features to avoid
target leakage.

Supporting exploratory folders (`BRFSS_analysis/`, `BRFSS_Missing_Analysis/`,
`BRFSS_Target_Variables_Analysis/`) hold earlier missingness/target summaries. `USCODE24_LLCP_082125.HTML`
is the official 2024 codebook.

---

## Quick start (local / PyCharm)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python BRFSS.py           # step 1 — clean
python BRFSS_Recode.py    # step 2 — recode
python BRFSS_EDA.py       # step 3 — explore
python BRFSS_Model.py     # step 4 — train
python BRFSS_Evaluate.py  # step 5 — evaluate
```

> On Windows, if the console errors on codebook text, set `PYTHONIOENCODING=utf-8` first.

The cleaned/recoded **parquet** files are committed, so steps 3–5 run without re-running step 1.

---

## Running in Google Colab

Two options, easiest first.

### Option A — open the ready-made notebooks (recommended)
This repo ships Colab notebooks that **clone the repo into Colab and run the pipeline**, so the code and
the committed data come along automatically — nothing to upload.

- **`BRFSS_Colab.ipynb`** — runs the whole pipeline (clean → recode → EDA → model → evaluate).
- **`BRFSS_Recode_Colab.ipynb`** — the recode step (Steps 1–5) on its own, with an upload/Drive option.

To open either one in Colab, prefix its GitHub URL with the Colab loader:

```
https://colab.research.google.com/github/Kudos-4/BRFSS/blob/main/BRFSS_Colab.ipynb
```

Then just **Runtime → Run all**.

### Option B — upload a script and a data file
In a Colab cell:

```python
!git clone https://github.com/Kudos-4/BRFSS.git
%cd BRFSS
!pip install -q xgboost pyarrow      # most others are preinstalled in Colab
!python BRFSS_Recode.py              # or any other step
```

Because paths in every script are **relative to the repo root**, they work unchanged in Colab after `%cd BRFSS`.

---

## What is (and isn't) in the repo

**Committed:** all code, the ready-to-run notebooks, the raw `BRFSS_2024.parquet` (35 MB), the small
cleaned/recoded parquets, all EDA/evaluation plots and summary CSVs, the small trained models, the codebook,
and the guideline PDF.

**Excluded** (see `.gitignore`) — too large for GitHub and fully regenerable by re-running the pipeline:
`BRFSS_2024.pkl` (1.1 GB raw pickle), the ~400 MB per-row missing-value CSV dumps, the large intermediate
CSV twins of the parquets, and `Random_Forest.pkl` (62 MB — rebuild with `python BRFSS_Model.py`).
