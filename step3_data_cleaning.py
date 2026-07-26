"""
Step 3 - Data Cleaning
=======================
Goal: fix the issues found in Step 2 (missing first_shooter, date as text)
and derive two columns we'll need for our business question.
"""

import pandas as pd

df = pd.read_csv("shootouts.csv")

# 1. Drop rows where first_shooter is missing (~62% of the data). This is
#    the exact information our business question depends on, so we cannot
#    guess or impute it - we only keep matches where it was recorded.
df_clean = df.dropna(subset=['first_shooter']).copy()

# 2. Convert date from text to a proper datetime type
df_clean['date'] = pd.to_datetime(df_clean['date'])

# 3. Derive a clean home/away signal: did the home team shoot first?
#    (first_shooter originally stores a team NAME, not "home"/"away")
df_clean['home_shot_first'] = (df_clean['first_shooter'] == df_clean['home_team']).astype(int)

# 4. Derive our target variable: did the home team win the shootout?
df_clean['home_team_won'] = (df_clean['winner'] == df_clean['home_team']).astype(int)

print("Rows before cleaning:", len(df))
print("Rows after cleaning:", len(df_clean))
print()
print(df_clean[['date', 'home_team', 'away_team', 'home_shot_first', 'home_team_won']].head())
print()
print("Missing values after cleaning:")
print(df_clean.isnull().sum())

df_clean.to_csv("shootouts_cleaned.csv", index=False)
