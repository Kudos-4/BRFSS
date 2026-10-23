# brfss24.py - script twin of brfss24.ipynb (PI's notebook + Jason's review cells), '# %%' cell format for PyCharm.

# %%
#----------------------------------------------------------------------------------------#
# Created on: 07/3/2026  --------- Modified on 07/5/2026                                 #
# -----Analysis completed -------Results are convincing ----Prepare a manuscript --------#
#----------------------------------------------------------------------------------------#

# %% [markdown]
# # **Machine Learning Approaches to Predict Poor Mental Health Among U.S. Adults: Evidence from BRFSS 2024**

# %% [markdown]
# ## 1. Importing necessary libraries and data

# %% [markdown]
# **1. Importing necessary libraries**

# %%
# Run this block of code everytime
# Importing the basic libraries we will require for the project

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# Libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Importing the Machine Learning models we require from Scikit-Learn
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Importing the other functions we may require from Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# To get diferent metric scores
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,precision_recall_curve,roc_curve,make_scorer

# Code to ignore warnings from function usage
import warnings;
import numpy as np
warnings.filterwarnings('ignore')

# %% [markdown]
# **2. Connect google colab to google drive to permit this notebook to access your Google Drive files?**

# %%
# Run this everytime
from google.colab import drive
drive.mount('/content/gdrive')

# %% [markdown]
# **3. Load data**

# %% [markdown]
# **Note:** You do not need to load this data everytime you start a new session once it is cleaned up. Instead, jump to start loading ***brfss24_clean.csv***.

# %%
# Just read this file one time for cleaning. Later you will be using cleaned file for the analysis
mydata = pd.read_csv('gdrive/My Drive/BRFSS24_DilliNJason/brfss24.csv')

# %% [markdown]
# ## 2. Data Overview

# %%
mydata.head()

# %%
# Dimension of the data
mydata.shape

# %%
#data.info()

# %%
# Make a copy
df = mydata.copy()

# %% [markdown]
# # 3. Data Cleaning

# %% [markdown]
# ## **Recommended Variables**
#
# | No. | Variable   | Description                     | Reason for Inclusion                                          |
# | --- | ---------- | ------------------------------- | ------------------------------------------------------------- |
# | 1   | `_AGE80`   | Age                             | Mental health varies by age                                   |
# | 2   | `_SEX`     | Sex                             | Mental health differs between males and females               |
# | 3   | `_RACEGR3`    | Race/Ethnicity                  | Five-level race/ethnicity category                        |
# | 4   | `MARITAL`  | Marital status                  | Social support influences mental health                       |
# | 5   | `_EDUCAG`    | Education level                 |  Level of education completed                    |
# | 6   | `_INCOMG1`  | Household income                | Income categories                       |
# | 7   | `EMPLOY1`  | Employment status               | Unemployment is linked to depression and anxiety              |
# | 8   | `GENHLTH`  | General health status           | Strong predictor of mental health                             |
# | 9   | `PHYSHLTH` | Poor physical health days       | Physical and mental health are closely related                |
# | 10  | `_BMI5CAT` | BMI category                    | Obesity is associated with poor mental health                 |
# | 11  | `_TOTINDA` | Any physical exercise           | Physical activity improves mental well-being                  |
# | 12  | `_SMOKER3` | Smoking status                  | Smoking is associated with depression                         |
# | 13  | `MEDCOST1` | Couldn't see doctor due to cost | Indicates barriers to healthcare                              |
# | 14  | `PRIMINS2` | Health insurance coverage       | Access to healthcare affects mental health                    |
# | 15  | `PERSDOC3` | Has a personal doctor           | Reflects continuity of healthcare                             |
# | 16  | `ADDEPEV3` | Ever diagnosed with depression  | Strong predictor of future poor mental health                 |
# | 17  | `DIABETE4` | Diabetes status                 | Chronic disease increases psychological burden                |
# | 18  | `ASTHMA3`  | Asthma status                   | Chronic respiratory illness is linked with anxiety/depression |
# | 19  | `CHCCOPD3` | COPD status                     | Chronic illness contributes to poor mental health             |
# | 20  | `DIFFWALK` | Difficulty walking              | Disability is associated with mental distress                 |
#
# **Recommended Target Variable**
# * _MENT14D (Frequent Mental Distress: ≥14 poor mental health days in the past 30 days)
#     * Binary classification (recommended for logistic regression, Random Forest, XGBoost, etc.)
#
# **If Using MENTHLTH as the Outcome**
# If your target is MENTHLTH (number of poor mental health days), exclude: _MENT14D

# %%
# Let us keep the following variables from df based on above recommendation.
keep_var = ['MENTHLTH','_MENT14D','_AGE80','_SEX','_RACEGR3','MARITAL','_EDUCAG','_INCOMG1','EMPLOY1','GENHLTH','PHYSHLTH',
           '_BMI5CAT','_TOTINDA','_SMOKER3','MEDCOST1','PRIMINS2','PERSDOC3','ADDEPEV3','DIABETE4','ASTHMA3','CHCCOPD3','DIFFWALK']
df1 = df[keep_var]

# Recommended Target Variable: _MENT14D (Frequent Mental Distress: ≥14 poor mental health days in the past 30 days)
                            # Binary classification (recommended for logistic regression, Random Forest, XGBoost, etc.)

# %%
df1.shape

# %%
df1.info()

# %%
# Check duplicated rows
duplicated_count=df1.duplicated().sum()
print(duplicated_count)

# %%
# Remove duplicate rows
df1_clean = df1.drop_duplicates()

# %% [markdown]
# **NOTE (Jason):** `drop_duplicates()` here compares respondents on their 22 answers, not on a respondent ID. Because almost all of these variables are categorical, two different people can easily give identical answers, so this step removes real respondents rather than duplicate records. The cell below counts duplicates by respondent ID (`SEQNO`) versus by answer pattern and gives a switch to skip the answer-pattern de-dup so the analysis can be run both ways.

# %%
# --- Duplicate check: by respondent ID vs. by answer pattern (added by Jason) ---
# BRFSS SEQNO is only unique WITHIN a state, so the respondent ID is (_STATE, SEQNO).
id_cols = [c for c in ['_STATE', 'SEQNO'] if c in df.columns]
by_pattern = int(df1.duplicated().sum())
by_id = int(df[id_cols].duplicated().sum()) if len(id_cols) == 2 else None
print(f'Duplicate rows by answer pattern (22 variables): {by_pattern:,}')
print(f'Duplicate rows by respondent ID {id_cols}: ' + (f'{by_id:,}' if by_id is not None else 'ID columns not found'))
if by_id == 0 and by_pattern > 0:
    print(f'-> No true duplicate records: the {by_pattern:,} rows removed by drop_duplicates() are distinct respondents with identical answers.')

# To keep every respondent (skip the answer-pattern de-dup), uncomment the next line:
# df1_clean = df1.copy()

# %%
df2 = df1_clean.copy()

# %%
df2.shape

# %%
# Count of unique values in each column
unique_counts = df2.nunique()
print(unique_counts)

# %%
df3 = df2.drop(columns=['MENTHLTH'])

# %%
df3.info()

# %%
# Count of missing (NaN) values in each columns
df3.isnull().sum()

# %% [markdown]
# **Note:** If your goal is a research-quality analysis, I would not drop true NaNs here. They will be dropped after recoding 7, 9, 77, 99 etc. type values to NaN.

# %%
# Get column names
col_names = df3.columns
print(col_names)

# %% [markdown]
# ### 3.2 Recoding

# %%
# Unique values per column
for col in col_names:
    print(f"Unique values in '{col}': {df3[col].unique()}")

# %% [markdown]
# **Observations:**
#
# *   In many columns 7, 9, 7., or 9. are typically used as codes for missing, refused, or unknown values in survey datasets (like BRFSS or NHANES).
#       - 7 → "Don't know/Not sure"
#       - 9 → "Refused"
#       - 7., 9. → Likely represents a float version of this
#       - In **_INCOMG1**, number 7 and 9 represent valid data value but 77 and 99 represent Don't Know/Not Sure or Refused
#          - Convert 77 and 99 to missing (NaN).
#       - For **DIABETE4**, you can make it binary type by excluding prediabetes because it is neither clearly diabetic nor non-diabetic. So, recode as: 1=1, 3=0 and 2, 4, 7, 9, Missing=NaN
#         - **Use ChatGpt prompt:** How to create binary category for DIABETE4 in BRFSS 2024. You can copy and paste the table and context related to DIABETE4 with the prompt in ChatGpt.

# %%
# Recoding: Make sure to check if these coding matches with BRFSS 2024 codebook. Otherwsie, the results would be reversed.
# We collape categories while recoding to avoid sparsness in the distribution of categorical variables

# _MENT14D: Poor mental health status
df3['_MENT14D'] = df3['_MENT14D'].replace({
    1: 0,        # 0 days when mental health not good
    2: 0,        # 1–13 days when mental health not good
    3: 1,        # 14–30 days when mental health not good
    7: np.nan,   # Don't know
    9: np.nan    # Refused/Missing
})

# X_SEX: Sex of respondent (1=Male, 2=Female)
df3['_SEX'] = df3['_SEX'].replace({
    1: 1,  # Male
    2: 0   # Female
})

# _RACEGR3: Race/Ethnicity (# 1=White NH, 2=Black NH, 3=Other NH, 4=Multiracial NH, 5=Hispanic, 9=Missing)
# Keeping all race categories is generally preferable instead of collapsing them into White vs Non-White.
# If there is a sparseness in the distribution, then it may be necessary to collapse. We do that later.

df3['_RACEGR3'] = df3['_RACEGR3'].replace({
    9: np.nan
})

df3['MARITAL']=df3['MARITAL'].replace({

    1: 1,  # Married
    6: 1,  # Unmarried couple
    2: 2,  # Divorced
    3: 2,  # Widowed
    4: 2,  # Separated
    5: 3,   # Never married
    9: np.nan
})


# X_EDUCAG: For X_EDUCAG (Education Level), I recommend keeping it as an ordinal categorical variable because
# education often has a dose-response relationship with health outcomes.
# (1=Less than high school, 2=High school graduate, 3=Some college/technical school, 4=College graduate, 9=Missing)
df3['_EDUCAG'] = df3['_EDUCAG'].replace({9: np.nan})

# _INCOMG1: Income group
# 1=<$15k, 2=$15k-$24,999, 3=$25k-$34,999,
# 4=$35k-$49,999, 5=$50k-$99,999, 6=$100k-$199,999, 7=$200k+, 9=Missing
# 7 is a valid value

df3['_INCOMG1'] = df3['_INCOMG1'].replace({
    9: np.nan
})

# EMPLOY1: Employment status
df3['EMPLOY1'] = df3['EMPLOY1'].replace({

    1: 1,  # Employed for wages
    2: 1,  # Self-employed
    3: 2,  # Out of work >=1 year
    4: 2,  # Out of work <1 year
    5: 3,  # Homemaker
    6: 3,  # Student
    7: 3,  # Retired
    8: 4,   # Unable to work
    9: np.nan
    })

# GENHLTH: General health status (1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor, 7=Don't know, 9=Refused)
# Following coding is very common public-health recoding. No need to reverse.
# We want to recode 1=Poor health, 0=good health
df3['GENHLTH'] = df3['GENHLTH'].replace({
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
    7: np.nan,
    9: np.nan
    })

df3['PHYSHLTH'] = df3['PHYSHLTH'].replace({
          88: 0,
          77: np.nan,
          99: np.nan
      })

df3['_BMI5CAT'] = df3['_BMI5CAT'].replace({
    1: 1,  # Underweight
    2: 1,  # Normal
    3: 2,  # Overweight
    4: 3,   # Obese
    9999: np.nan

    })

# _TOTINDA: Total physical activity indicator
df3['_TOTINDA'] = df3['_TOTINDA'].replace({
    1: 1,      # Physically active
    2: 0,      # Not physically active
    9: np.nan  # Missing
})

# X_SMOKER3: Smoking status (1=Current smoker every day, 2=Current smoker some days, 3=Former smoker, 4=Never smoked, 9=Missing)
df3['_SMOKER3'] = df3['_SMOKER3'].replace({
    1: 1,
    2: 1,
    3: 0,
    4: 0,
    9: np.nan})

# MEDCOST1: Could not see a doctor in past 12 months because of cost (1=Yes, 2=No, 7=Don't know, 9=Refused)
df3['MEDCOST1'] = df3['MEDCOST1'].replace({
    1: 1,      # Cost prevented medical care
    2: 0,      # Cost did not prevent medical care
    7: np.nan,
    9: np.nan
})

df3['PRIMINS2'] = df3['PRIMINS2'].replace({
    1: 1,   # Private
    2: 1,   # Private
    3: 2,   # Public
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    88: 3,    # Uninsured
    77: np.nan,
    99: np.nan
    })

# PERSDOC3: Personal doctor or health care provider (1=Only one, 2=More than one, 3=None, 7=Don't know, 9=Refused)
df3['PERSDOC3'] = df3['PERSDOC3'].replace({
    1: 1,      # Has personal doctor
    2: 1,      # Has personal doctor
    3: 0,      # No personal doctor
    7: np.nan,
    9: np.nan
})


# ADDEPEV3: Depressive disorder
df3['ADDEPEV3'] = df3['ADDEPEV3'].replace({
    1:1,
    2: 0,
    7: np.nan,
    9: np.nan
})

# DIABETE4: Diabetes status (1=Diabetes, 3=No diabetes, 2=Gestational, 4=Prediabetes, 7=Don't know, 9=Refused)
df3['DIABETE4'] = df3['DIABETE4'].replace({
    1: 1,
    3: 0,
    2: np.nan,
    4: np.nan,
    7: np.nan,
    9: np.nan
    })


# ASTHMA3: Lifetime asthma
df3['ASTHMA3'] = df3['ASTHMA3'].replace({
    1: 1,  # No
    2: 0,  # Yes
    7: np.nan,
    9: np.nan
})

# CHCCOPD3: chronic obstructive pulmonary disease
df3['CHCCOPD3'] = df3['CHCCOPD3'].replace({
    1:1,
    2: 0,
    7: np.nan,
    9: np.nan
})

# DIFFWALK: Difficult walking
df3['DIFFWALK'] = df3['DIFFWALK'].replace({
    1:1,
    2: 0,
    7: np.nan,
    9: np.nan
})

# %% [markdown]
# **NOTE (Jason):** In the `ASTHMA3` block above the comments are flipped - in BRFSS, 1 = Yes and 2 = No, so the code (1 -> 1, 2 -> 0) is correct and only the comments are wrong. Also, `9999` is not a `_BMI5CAT` code (missing is blank), so that line is harmless but does nothing.

# %%
df3.shape

# %%
# Count of missing values after recoding
df3.isnull().sum()

# %%
# Now, drop missing (NaN) values
df3_clean = df3.dropna()

# %%
df3_clean.shape

# %%
# --- Row accounting (added by Jason): where do the rows go between the raw file and the analytic sample? ---
steps = [
    ('Raw file (df)',                              len(df)),
    ('Kept 22 variables (df1)',                    len(df1)),
    ('After drop_duplicates (df1_clean)',          len(df1_clean)),
    ('After recoding 7/9/77/99 -> NaN (df3)',      len(df3)),
    ('After dropna (df3_clean)',                   len(df3_clean)),
]
prev = None
for label, n in steps:
    change = '' if prev is None else f'   ({n - prev:+,})'
    print(f'{label:<45} {n:>9,}{change}')
    prev = n
print(f'\nFinal analytic sample = {len(df3_clean) / len(df):.1%} of the raw file.')

# %%
# Check unique values per column again
for col in col_names:
    print(f"Unique values in '{col}': {df3_clean[col].unique()}")

# %% [markdown]
# **Observations:**
# * All the variables except __AGE80, _RACEGR3, MARITAL, X_EDUCAG, INCOMG1, EMPLOY1, _BMI5CAT and PRIMIN2 are binary type.
# * __AGE80, _RACEGR3, MARITAL, X_EDUCAG, INCOMG1, EMPLOY1, _BMI5CAT and PRIMIN2 are categorical with multiple categories.

# %%
# Save this clean dataset --- Its likely that it is not fully cleaned, may need furthet clean up.
df3_clean.to_csv("gdrive/My Drive/BRFSS24_DilliNJason/brfss24_clean.csv", index=False)

# %%
# Load saved file for further analysis --- If needed we will further clean this dataset.
my_data = pd.read_csv('gdrive/My Drive/BRFSS24_DilliNJason/brfss24_clean.csv')

# %%
my_data.info()

# %%
my_data.shape

# %% [markdown]
# **NOTE (Jason):** Row counts reconcile: the cleaning chain ends at **286,704** rows (same as the shared `brfss24_clean.csv`), and the **229,363** observations reported in the logistic-model summary further down are the 80% *training* split of that (196,731 + 32,632). So the shared CSV is the final cleaned sample; the model summary just reports training N, not total N.

# %%
# Make a copy of the data
mdf = my_data.copy()

# %% [markdown]
# ## **Cross-tabulation of Outcome Vatiable and Categorical Variables**
# This will guide if further collapse in categories is required.

# %% [markdown]
# **NOTE (Jason):** The comment in the next cell says "RACE and CKD" - that is a leftover from a CKD template. The crosstab is actually race/ethnicity (`_RACEGR3`) by frequent mental distress (`_MENT14D`).

# %%
# Cross-tabulation of RACE and CKD ---- Distribution of CKD in Racial groups
pd.crosstab(mdf['_RACEGR3'], mdf['_MENT14D'],margins=True)
# Percentage of row
#pd.crosstab(mdf['_RACEGR3'], mdf['_MENT14D'], normalize='index',margins=True).round(2)*100

# %% [markdown]
# ## **Checking Class Imbalance in Outcome Variable**

# %%
# Distribution of  _MENT14D (count and proportion)
print(mdf['_MENT14D'].value_counts())
print(mdf['_MENT14D'].value_counts(normalize=True))

# %% [markdown]
# **Observation:** Not severaly unbalanced.

# %% [markdown]
# ## **START MODELLING NOW**

# %% [markdown]
# We split the data into training and test set. A standard approach is to split the data into 80% training and 20% testing using a fixed random seed for reproducibility.
#
# We use tarining set to build the model and use test set to evaluate predictive performance of the trained model.

# %% [markdown]
# **1. Dividing data into training and testing set**

# %%
from sklearn.model_selection import train_test_split

# Outcome variable
y = mdf['_MENT14D']

# Predictor variables
X = mdf.drop(columns=['_MENT14D'])

# Random 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y  # preserves outcome proportions
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# %% [markdown]
# ## 1. STANDARD LOGISTIC REGRESSION MODEL

# %% [markdown]
# The primary objective of a standard logistic regression model is to estimate and interpret the associations between explanatory variables and the outcome, rather than to maximize predictive performance unlike predictive modeling approaches.

# %%
import statsmodels.formula.api as smf

# Combine X and y for statsmodels
train_df = X_train.copy()
train_df['_MENT14D'] = y_train

# Logistic regression model
model = smf.logit(
    formula='''
  _MENT14D ~
  _AGE80+
_SEX+
C(_RACEGR3)+
 C(MARITAL)+
 C(_EDUCAG)+
 C(_INCOMG1)+
 EMPLOY1+
 GENHLTH+
 PHYSHLTH+
 C(_BMI5CAT)+
 _TOTINDA+
 _SMOKER3+
 MEDCOST1+
 C(PRIMINS2)+
 PERSDOC3+
 ADDEPEV3+
 DIABETE4+
 ASTHMA3+
 CHCCOPD3+
 DIFFWALK
    ''',
 data=train_df
).fit()

print(model.summary())

# %% [markdown]
# **Odds Ratios and 95% Confidence Intervals**

# %%
import numpy as np
import pandas as pd

or_table = pd.DataFrame({
    'OR': np.exp(model.params),
    'Lower95CI': np.exp(model.conf_int()[0]),
    'Upper95CI': np.exp(model.conf_int()[1]),
    'p_value': model.pvalues
})

print(or_table.round(3))

# %% [markdown]
# **NOTE (Jason):** The interpretation below refers to "American Indian/Alaska Native" and "Asian" respondents, but `_RACEGR3` only has five levels (1 = White NH, 2 = Black NH, 3 = Other race NH, 4 = Multiracial NH, 5 = Hispanic). Those group names would come from `_IMPRACE` or `_RACE`, so this paragraph may describe a different model run. Cross-check the race rows of the OR table above before using this text in the manuscript.

# %% [markdown]
# **Public Health Interpretation**
#
# Multivariable logistic regression identified race/ethnicity, marital status, education, household income, age, sex, self-rated health status, physically unhealthy days, physical activity, smoking status, cost-related barriers to healthcare, having a personal healthcare provider, depression, diabetes, asthma, and difficulty walking as significant predictors of frequent mental distress. Compared with White respondents, Black (OR=1.09, 95% CI: 1.04-1.15, p=0.001), American Indian/Alaska Native (OR=1.10, 95% CI: 1.04-1.17, p=0.001), and Asian respondents (OR=1.18, 95% CI: 1.09-1.28, p < 0.001) had higher odds of frequent mental distress, whereas Hispanic respondents had lower odds (OR=0.83, 95% CI: 0.79-0.87, p < 0.001). Compared with partnered individuals, previously married respondents had 24% higher odds (OR=1.24, 95% CI: 1.20-1.29, p < 0.001), while never-married respondents had 8% higher odds (OR=1.08, 95% CI: 1.04-1.12, p < 0.001). Higher household income was associated with progressively lower odds of frequent mental distress, with respondents in the highest income category having 37% lower odds than those in the lowest income category (OR=0.64, 95% CI: 0.58-0.69, p < 0.001). Increasing age was associated with lower odds of frequent mental distress (OR=0.97, 95% CI: 0.97-0.98, p < 0.001). Poorer self-rated health (OR=1.84, 95% CI: 1.77-1.91, p < 0.001), a greater number of physically unhealthy days (OR=1.05, 95% CI: 1.05-1.05, p < 0.001), current smoking (OR=1.38, 95% CI: 1.33-1.43, p < 0.001), inability to afford medical care (OR=1.93, 95% CI: 1.86-2.01, p < 0.001), asthma (OR=1.08, 95% CI: 1.04-1.12, p < 0.001), and difficulty walking or climbing stairs (OR=1.05, 95% CI: 1.01-1.10, p=0.017) were associated with increased odds of frequent mental distress. In contrast, meeting physical activity recommendations (OR=0.85, 95% CI: 0.82-0.88, p < 0.001), having a personal healthcare provider (OR=0.83, 95% CI: 0.79-0.87, p < 0.001), and diabetes (OR=0.92, 95% CI: 0.89-0.96, p < 0.001) were associated with lower odds. Respondents with a history of depression had more than fivefold higher odds of frequent mental distress compared with those without depression (OR=5.29, 95% CI: 5.14-5.44, p < 0.001). No statistically significant associations were observed for body mass index, health insurance coverage, employment status, or chronic obstructive pulmonary disease (COPD).
#
# Finally, The overall model was statistically significant (Likelihood Ratio χ², p < 0.001) and explained approximately 24.2% of the variability in frequent mental distress (Pseudo R² = 0.242). The model converged successfully and included 229,363 participants.

# %% [markdown]
# ### **1.1 Evaluate predictive performance of the model in test data**

# %%
# -------- Predict probabilities in test data --------------------------------------------#
# In statsmodels.formula.api, after fitting a logistic regression model,
# you can use the .predict() method to obtain predicted probabilities.
y_prob_test_stand = model.predict(X_test)

# %%
# Print the first 5 elements of the predicted probabilities array/Series.
print(y_prob_test_stand[:5])

# %%
# Check the data type (class) of the object stored in y_prob_stand ---- its a panda series
type(y_prob_test_stand)

# %%
# Summary of predicted probabilities ---- use describe() ----for panda series object
print(y_prob_test_stand.describe())

# %%
# Predict class label in test data using default threshold cutoff value of 0.50
y_pred_test_stand50 = (y_prob_test_stand >= 0.5).astype(int)

# %%
# Calcuate accuracy, AUC, confusion matrix & classification report
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

print("***** Predictive Performance ***** Test Data *****Standard Logistic********")
print("\n Accuracy at default(50%) threshold:", accuracy_score(y_test, y_pred_test_stand50))
print("AUC:", roc_auc_score(y_test, y_prob_test_stand))

print("\n Confusion Matrix:\n",confusion_matrix(y_test, y_pred_test_stand50))
print("\n Classification Report:\n", classification_report(y_test, y_pred_test_stand50))

# %% [markdown]
# **Definitions:**
#
# **1. Accuracy:** Accuracy is a measure of how often a classification model makes the correct prediction.
#
# The formula is:
# $$
# Accuracy=\frac{Number ~of ~Correct ~Predictions}{Total ~Number ~of ~Predictions}
# $$
# \
# **2. Area Under the ROC Curve (AUC):**
#
# It measures how well a classification model can distinguish between the two classes (0 and 1).
# * If I randomly pick one positive case and one negative case, what is the probability that the model assigns a higher score to the positive case?

# %% [markdown]
# **Arbitrarily lower the cutoff from default (50%) to 10% and calculate sensitivity.**

# %%
y_pred_test_stand10 = (y_prob_test_stand >= 0.10).astype(int)

# %%
# Calcuate sensitivity and other measures
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

print("***** Predictive Performance ***** Test Data *****Standard Logistic********")
print("Accuracy at 10% threshold:", accuracy_score(y_test, y_pred_test_stand10))
print("AUC:", roc_auc_score(y_test, y_prob_test_stand))

print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_stand10))
print("\n Classification Report:\n",classification_report(y_test, y_pred_test_stand10))

# %%
from sklearn.metrics import roc_curve
import numpy as np

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_test_stand)

# Youden Index
youden_index = tpr - fpr

# Optimal threshold
optimal_idx = np.argmax(youden_index)
optimal_threshold_stand = thresholds[optimal_idx]

optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

print("Optimal threshold:", optimal_threshold_stand)
print("Youden Index:", youden_index[optimal_idx])
print("Sensitivity:", optimal_sensitivity)
print("Specificity:", optimal_specificity)

# %% [markdown]
# **Calculate predictive performance measures at the optimal threshold**

# %%
# Predict class labels in test data using optimal threshold
y_pred_test_standOpt = (y_prob_test_stand >= optimal_threshold_stand).astype(int)

# %%
# Calcuate performance measures: accuracy, AUC, confusion matrix and classification report
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

print("Accuracy at optimal threshold:", accuracy_score(y_test, y_pred_test_standOpt))
print("AUC:", roc_auc_score(y_test, y_prob_test_stand))

print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_test_standOpt))
print("\nClassification report:\n", classification_report(y_test, y_pred_test_standOpt))

# %% [markdown]
# **Predictive Models:**
#
# We will explore following models:
# 1. Class-weighted Logistic Regression (also known as Balanced Logistic Regression).
# 2. Tuned Random Forest with balanced classes
# 3. Tuned XGBoost
# 4. SMOTE
#
# We then compare AUC, Sensitivity and Specificity for all the models
# - Standard Logistic Regression
# - Balanced Logestic Regression
# - Tuned Random Forest
# - Tuned XGBoost
# - SMOTE

# %% [markdown]
# ## 2. Class Weighted Logistic Regression (Balanced Logistic Regression)

# %% [markdown]
# Balanced logistic regression is appropriate when the outcome variable is imbalanced and the minority class is clinically or practically important. In such situations, a standard logistic regression model may focus primarily on maximizing overall accuracy, which can lead to poor detection of the minority class.
#
# Balanced logistic regression addresses this issue by assigning a weight to each observation in the loss function:
#
# $L=-\sum_{i=1}^{N} w_i \left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$
#
# where:
# * $y_i$ is the observed class label,
# * $p_i$ is the predicted probability of the positive class
# * $w_i$ is the weight assigned to observation $i$.
#
# The weights depend on the class membership of each observation. A commonly used weighting scheme is:
# $$
# w_c=\frac{N}{KN_c}
# $$
#
# where:
#
# - N = total number of observations
# - K = number of classes
# - $N_c$ = number of observations in class $c$
#
# Under this approach, classes with fewer observations receive larger weights, causing misclassification of minority-class observations to incur a greater penalty during model training. As a result, the model places more emphasis on correctly identifying minority-class cases, often improving sensitivity at the expense of some reduction in overall accuracy and specificity.
#
# **Example (you need to change following numbers)**
#
# For your CKD dataset:
#
# * Total observations: N=5479
# * Classes: K=2
# * No CKD: $N_0$ =5218
# * CKD: $N_1$=261
#
# The corresponding class weights are:
# $$
# w_0=\frac{5479}{2\times 5218}\approx 0.525
# $$
# $$
# w_1=\frac{5479}{2\times 261}\approx 10.50
# $$
#
# Thus, a CKD observation receives approximately 20 times more weight than a non-CKD observation during model fitting, encouraging the model to pay greater attention to identifying CKD cases.

# %% [markdown]
# **NOTE (Jason):** The worked example above (N = 5479, CKD vs no CKD) is left over from a different project's template. The cell below computes the actual balanced class weights for this BRFSS training set.

# %%
# --- Balanced class weights for THIS dataset (added by Jason) ---
N = len(y_train); K = y_train.nunique()
counts = y_train.value_counts().sort_index()
w = {}
for c, n_c in counts.items():
    w[int(c)] = N / (K * n_c)
    print(f'class {int(c)}: N_c = {n_c:>9,}   w_c = N / (K * N_c) = {w[int(c)]:.3f}')
print(f'\nA frequent-mental-distress observation gets ~{w[1] / w[0]:.1f}x the weight of a non-case.')

# %%
###------------ IMPORTANT NOTE ----------------------------------------------------------------------------------#####
#  You can fit a balanced logistic using "statsmodels.formula.api" but not with smf.logit() directly
#  instead use smf.glm() with a Binomial family and supply observation weights.
# Rather, we fit balanced logistic regression in scikit-learn (= "sklearn.linear_model")
# statsmodels.formula.api.logit() does not support a class_weight='balanced' parameter like scikit-learn.
### -------------------------------------------------------------------------------------------------------------####

# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

clf.fit(X_train, y_train)

# %% [markdown]
# **Explanation:**
#
# The **'balanced'** option sets class weights inversely proportional to class frequencies:
# $$
# w_c=\frac{n_{sample}}{n_{classes}\times n_c}
# $$
# where $n_c$ is the number of samples in class $c$.
#
# You can also pass custom weights Or supply per-sample weights during fitting.

# %%
# Passing custom weights
#clf = LogisticRegression(
#    class_weight={0: 1, 1: 5}
#)

#clf.fit(X_train, y_train)

# %%
# Supply per-sample weights during fitting
#clf.fit(X_train, y_train, sample_weight=weights)

# %% [markdown]
# **Note:**
# * In **scikit-learn**, clf.fit() does not produce a statistical summary like statsmodels (such as smf.logit.fit()) does.
# * After fitting, you can inspect the fitted model through its attributes as displayed below.
# * scikit-learn LogisticRegression → prediction-focused, no built-in statistical summary.
# * statsmodels Logit → inference-focused, includes a full summary report.

# %%
# 1. Coefficient
# print("Coefficients:", clf.coef_)

# %%
# 2. Intercept
#print("Intercept:", clf.intercept_)

# %%
# 3. Coefficients with variable names

import pandas as pd

coef_df = pd.DataFrame({
    'Variable': X_train.columns,
    'Coefficient': clf.coef_[0]
})

print(coef_df.sort_values('Coefficient', ascending=False))

# %% [markdown]
# Scikit-learn does not automatically recognize integer-coded variables as categorical. Instead, it treats them as continuous numeric variables. In contrast, statsmodels automatically generated dummy variables for these predictors in the standard logistic regression model. Therefore, one-hot encoding is required in scikit-learn to properly represent these categorical variables.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Categorical variables
categorical_cols = [
    '_RACEGR3',
    'MARITAL',
    '_EDUCAG',
    '_INCOMG1',
    '_BMI5CAT',
    'PRIMINS2'
]

# Binary + continuous variables
numeric_cols = [
'_AGE80',
'_SEX',
'EMPLOY1',
 'GENHLTH',
 'PHYSHLTH',
 '_TOTINDA',
 '_SMOKER3',
 'MEDCOST1',
 'PERSDOC3',
 'ADDEPEV3',
 'DIABETE4',
 'ASTHMA3',
 'CHCCOPD3',
 'DIFFWALK'
]

# One-hot encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(drop='first', handle_unknown='ignore'),
            categorical_cols
        ),
        (
            'num',
            'passthrough',
            numeric_cols
        )
    ]
)

# Balanced logistic regression
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('logit', LogisticRegression(
        class_weight='balanced',
        max_iter=5000,
        random_state=42
    ))
])

# Fit model
clf.fit(X_train, y_train)

# %% [markdown]
# **Important Point to Know:**
# * In scikit-learn, you do not pass a DataFrame in the model definition. The workflow is different from Statsmodels.

# %% [markdown]
# **NOTE (Jason):** The next cell is a concept-only example and errors on purpose (`x1, x2, x3` don't exist). **"Run all" stops here.** To execute the rest of the notebook, click the cell *after* it and use **Runtime > Run after** (Ctrl+F10).

# %%
#### ------------------ DO NOT RUN THIS BLOCK OF CODE ------------------#####
#### ------------------ THIS IS JUST FOR THE CONCEPT -------------#####

# In Statsmodels, You specify the formula and dataset when creating the model:
import statsmodels.formula.api as smf

model = smf.logit(
    formula='y ~ x1 + x2 + x3',
    data=df
)

result = model.fit()

# Scikit-learn: You create an empty estimator first, then pass the data to .fit():
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)

model.fit(X_train, y_train)

# %% [markdown]
# **Evaluate predictive performance in test data**

# %%
### --------------------------- IMPORTANT NOTE ----------------------------------------###
# In scikit-learn, the probabilities are predicted using model.predict_proba(X_test) unlike model.predict(X_test) in
# statsmodels.formula.api.
### -----------------------------------------------------------------------------------###

# %%
### ------- If you predict the probability this way, it returns a matrix ------- ###
# y_prob_bal=clf.predict_proba(X_test)
# print(y_prob_bal.shape)
# print(y_prob_bal[:5])
# y_prob_bal here is a matrix of dimension 1096x2, where Column 0 = P(Y=0), Column 1 = P(Y=1)
# Usually for ROC, AUC, and threshold selection, you want only the probability of the positive class which is a one dimensional vector.

# %%
# Predicted probabilities for positive class (Y=1)
y_prob_test_bal = clf.predict_proba(X_test)[:, 1]

# %%
# Using default threshold
y_pred_test_bal50 = (y_prob_test_bal >= 0.5).astype(int)
#y_pred_test_bal10 = (y_prob_test_bal >= 0.10).astype(int)

# %%
# Evaluation
print("******Predictive Performance ********* Test Data ******** Balanced Logistic *******")
print("Accuracy at default(50%) threshold:", accuracy_score(y_test,y_pred_test_bal50))
print("AUC:", roc_auc_score(y_test, y_prob_test_bal))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_bal50))
print("Classification Report:\n", classification_report(y_test, y_pred_test_bal50))

# %% [markdown]
# **Find Optimal Threshold Using Youden Index**

# %%
from sklearn.metrics import roc_curve
import numpy as np

fpr, tpr, thresholds = roc_curve(y_test, y_prob_test_bal)

youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)

optimal_threshold_bal = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

print("Optimal threshold in Balanced Logistic:", optimal_threshold_bal)
print("Youden Index:", youden_index[optimal_idx])
print("Sensitivity:", optimal_sensitivity)
print("Specificity:", optimal_specificity)

# %%
# Calculate accuracy, AUC, confusion matrix and classification report at the optimal threshold
y_pred_test_balOpt = (y_prob_test_bal >= optimal_threshold_bal).astype(int)

print("***** Predictive Performance ******* Test Data ********** Balanced Logistic")
print("\n Accuracy at optimal threshold:", accuracy_score(y_test,y_pred_test_balOpt))
print("AUC:", roc_auc_score(y_test, y_prob_test_bal))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_balOpt))
print("\n Classification Report:\n", classification_report(y_test, y_pred_test_balOpt))

# %% [markdown]
# **Important**
#
# You should find the optimal threshold separately for each model. Do not use the same threshold because each model produces probabilities on a different scale.

# %% [markdown]
# **NOTE (Jason):** The final summary mentions SMOTE / SMOTENC, but there is no SMOTE code in this notebook. The optional cell below reproduces that check. SMOTENC is used (not plain SMOTE) because most predictors are categorical and plain SMOTE would interpolate between category codes. It is commented out by default.

# %%
# --- OPTIONAL: SMOTENC + logistic regression (added by Jason; commented out by default) ---
# !pip install -q imbalanced-learn
# from imblearn.over_sampling import SMOTENC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
#
# cat_idx = [X_train.columns.get_loc(c) for c in categorical_cols]   # categorical_cols defined in the one-hot cell above
# sm = SMOTENC(categorical_features=cat_idx, random_state=42)
# X_res, y_res = sm.fit_resample(X_train, y_train)
# print('Resampled class counts:', y_res.value_counts().to_dict())
#
# clf_smote = Pipeline([('preprocessor', preprocessor),
#                       ('logit', LogisticRegression(max_iter=5000, random_state=42))])
# clf_smote.fit(X_res, y_res)
# y_prob_test_smote = clf_smote.predict_proba(X_test)[:, 1]
# print('AUC (SMOTENC + logistic):', roc_auc_score(y_test, y_prob_test_smote))

# %% [markdown]
# ## 3. Building a Random Forest model

# %% [markdown]
# ### 3.1 Tuned Random Forest Model ---- Better Version

# %%
# Parameter tuning
from sklearn.ensemble import RandomForestClassifier

tun_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

tun_rf.fit(X_train, y_train)

# %% [markdown]
# ### 3.1.1 Predictive Performance in Test Data

# %%
# Predict probabilities in test data using fitted tuned random forest
y_prob_test_tun_rf = tun_rf.predict_proba(X_test)[:,1]

# %%
# Predict class label in test data using default threshold cutoff value of 0.50
y_pred_test_tun_rf50 = (y_prob_test_tun_rf >= 0.5).astype(int)

# %%
# Calcuate predictive performance measures
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

print("Predictive performance ---- in test data -------using tuned Random Forest Model")
print("\n Accuracy at default(50%) threshold:", accuracy_score(y_test, y_pred_test_tun_rf50))
print("AUC:", roc_auc_score(y_test, y_prob_test_tun_rf))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test_tun_rf50))
print("\n Classification Report:")
print(classification_report(y_test, y_pred_test_tun_rf50))

# %% [markdown]
# ### 3.2.2 Optimized Threshold in Tuned Random Forest

# %%
from sklearn.metrics import roc_curve
import numpy as np

fpr, tpr, thresholds = roc_curve(y_test, y_prob_test_tun_rf)

youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)

optimal_threshold_tunRF = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

print("Optimal threshold in tuned random forest:", optimal_threshold_tunRF)
print("Sensitivity:", optimal_sensitivity)
print("Specificity:", optimal_specificity)
print("Youden Index:", youden_index[optimal_idx])

# %% [markdown]
# ### 3.1.3 Predictive Performance in Test Data at Optimal Threshold

# %%
y_pred_test_tun_rfOpt = (y_prob_test_tun_rf >= optimal_threshold_tunRF).astype(int)

# %%
# Calculate accuracy, AUC, confusion matrix and classification report using optimal threshold
print("Predictive performance ---- in test data -------using tuned Random Forest Model")
print("\n Accuracy at optimal threshold:", accuracy_score(y_test, y_pred_test_tun_rfOpt))
print("AUC:", roc_auc_score(y_test, y_prob_test_tun_rf))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test_tun_rfOpt))
print("\n Classification Report:")
print(classification_report(y_test, y_pred_test_tun_rfOpt))

# %%
# --- Feature importance: tuned Random Forest (added by Jason) ---
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
OUT_DIR = 'gdrive/My Drive/BRFSS24_DilliNJason'

imp_rf = pd.Series(tun_rf.feature_importances_, index=X_train.columns).sort_values()
ax = imp_rf.plot(kind='barh', figsize=(8, 7), color='steelblue')
ax.set_title('Tuned Random Forest - impurity-based feature importance')
ax.set_xlabel('Mean decrease in impurity')
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/feature_importance_rf.png', dpi=150); plt.show()

# Permutation importance on a 20,000-row sample of the test set (AUC drop when a feature is shuffled)
Xp = X_test.sample(n=min(20000, len(X_test)), random_state=42); yp = y_test.loc[Xp.index]
perm_rf = permutation_importance(tun_rf, Xp, yp, scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=-1)
perm_rf_df = pd.DataFrame({'mean_auc_drop': perm_rf.importances_mean, 'std': perm_rf.importances_std},
                          index=X_train.columns).sort_values('mean_auc_drop', ascending=False)
print(perm_rf_df.round(4))
perm_rf_df.to_csv(f'{OUT_DIR}/permutation_importance_rf.csv')

# %% [markdown]
# ## **4. XGBoosting (eXtreme Gradient Boosting)**

# %% [markdown]
# ### 4.1 Tuned XGBoost --- Balanced

# %%
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Compute balanced class weights
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

# Tuned XGBoost
tun_xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',

    # Tree parameters
    max_depth=3,
    min_child_weight=5,

    # Boosting parameters
    learning_rate=0.03,
    n_estimators=500,

    # Sampling
    subsample=0.8,
    colsample_bytree=0.8,

    # Regularization
    reg_alpha=0.1,
    reg_lambda=1.0,

    random_state=42
)

# Fit with balanced weights
tun_xgb.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

# %% [markdown]
# ### 4.1.1 Predictive Performance

# %%
y_prob_test_tun_xgb = tun_xgb.predict_proba(X_test)[:,1]

# %%
# Predict class label in test data using default threshold cutoff value of 0.50
y_pred_test_tun_xgb50 = (y_prob_test_tun_xgb >= 0.5).astype(int)

# %%
# Compute AUC, accuracy, confusion matrix and classification report at default threshold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

print("Predictive performance ---- in test data -------using tuned XGBoost Model")
print("Accuracy at default(50%) threshold:", accuracy_score(y_test, y_pred_test_tun_xgb50))
print("AUC:", roc_auc_score(y_test, y_prob_test_tun_xgb))

print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_tun_xgb50))
print("\n Classification Report:\n", classification_report(y_test, y_pred_test_tun_xgb50))

# %% [markdown]
# ### 4.2.2 Optimal Threshold

# %%
from sklearn.metrics import roc_curve
import numpy as np

fpr, tpr, thresholds = roc_curve(y_test, y_prob_test_tun_xgb)

youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)

optimal_threshold = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

print("Optimal threshold:", optimal_threshold)
print("Sensitivity:", optimal_sensitivity)
print("Specificity:", optimal_specificity)
print("Youden Index:", youden_index[optimal_idx])

# %%
from sklearn.metrics import roc_curve
import numpy as np

fpr, tpr, thresholds = roc_curve(y_test, y_prob_test_tun_xgb)

youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)

optimal_threshold_tunXGB = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

print("Optimal threshold in tuned  XGBoost:", optimal_threshold_tunXGB)
print("Sensitivity:", optimal_sensitivity)
print("Specificity:", optimal_specificity)
print("Youden Index:", youden_index[optimal_idx])

# %%
y_pred_test_tun_xgbOpt = (y_prob_test_tun_xgb >= optimal_threshold_tunXGB).astype(int)

# %%
# Calculate accuracy, AUC, confusion matrix and classification report using optimal threshold
print("Predictive performance ---- in test data -------using tuned XGBoost Model")
print("Accuracy at optimal threshold:", accuracy_score(y_test, y_pred_test_tun_xgbOpt))
print("AUC:", roc_auc_score(y_test, y_prob_test_tun_xgb))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test_tun_xgbOpt))
print("\n Classification Report:")
print(classification_report(y_test, y_pred_test_tun_xgbOpt))

# %%
# --- Feature importance: tuned XGBoost (added by Jason) ---
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
OUT_DIR = 'gdrive/My Drive/BRFSS24_DilliNJason'

imp_xgb = pd.Series(tun_xgb.feature_importances_, index=X_train.columns).sort_values()
ax = imp_xgb.plot(kind='barh', figsize=(8, 7), color='darkorange')
ax.set_title('Tuned XGBoost - gain-based feature importance')
ax.set_xlabel('Importance (gain share)')
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/feature_importance_xgb.png', dpi=150); plt.show()

# Permutation importance on a 20,000-row sample of the test set (AUC drop when a feature is shuffled)
Xp = X_test.sample(n=min(20000, len(X_test)), random_state=42); yp = y_test.loc[Xp.index]
perm_xgb = permutation_importance(tun_xgb, Xp, yp, scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=-1)
perm_xgb_df = pd.DataFrame({'mean_auc_drop': perm_xgb.importances_mean, 'std': perm_xgb.importances_std},
                           index=X_train.columns).sort_values('mean_auc_drop', ascending=False)
print(perm_xgb_df.round(4))
perm_xgb_df.to_csv(f'{OUT_DIR}/permutation_importance_xgb.csv')

# %%
# --- ROC curves: all models on the same test set (added by Jason) ---
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
OUT_DIR = 'gdrive/My Drive/BRFSS24_DilliNJason'

curves = {
    'Standard Logistic':   y_prob_test_stand,
    'Balanced Logistic':   y_prob_test_bal,
    'Tuned Random Forest': y_prob_test_tun_rf,
    'Tuned XGBoost':       y_prob_test_tun_xgb,
}
fig, ax = plt.subplots(figsize=(7, 6))
for name, prob in curves.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc_score(y_test, prob):.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance (0.500)')
ax.set_xlabel('False positive rate (1 - specificity)')
ax.set_ylabel('True positive rate (sensitivity)')
ax.set_title('ROC curves - frequent mental distress (_MENT14D), test set')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/roc_all_models.png', dpi=150); plt.show()

# %% [markdown]
# ## **Final Comparison of all the models**
#
# - Logistic Regression      AUC ≈ 0.79
# - Balanced Logistic        AUC ≈ 0.79
# - Tuned Random Forest      AUC ≈ 0.779
# - Tuned XGBoost            AUC ≈ 0.755
# - SMOTE                    No improvement
# - SMOTENC                  No improvement
#
# This is actually a very important finding.
#
# **What does this tell us?**
#
# The limitation is not class imbalance
#
# Many analysts assume that imbalanced data
# - → Apply SMOTE
# - → Big improvement
#
# But that only works when the minority class contains useful hidden structure that the model is failing to learn.
#
# **Your results suggest:**
#
# The models are already extracting most of the available signal from the predictors.
#
# SMOTE cannot create new information.
#
# It only creates synthetic observations based on existing ones.

# %% [markdown]
# **NOTE (Jason):** The SMOTE / SMOTENC lines in this summary have no code behind them in this notebook. An optional, commented-out SMOTENC cell was added under the Balanced Logistic section (just before "3. Building a Random Forest model") so that result can be reproduced if needed.
