"""
Step 3 - Data Cleaning
=======================
Goal: fix the issues found in Step 2 (missing first_shooter, date as text)
and derive two columns we'll need for our business question.
"""

import pandas as pd

df = pd.read_csv("shootouts.csv")

df_clean = df.dropna(subset=['first_shooter']).copy()

df_clean['date'] = pd.to_datetime(df_clean['date'])

df_clean['home_shot_first'] = (df_clean['first_shooter'] == df_clean['home_team']).astype(int)

df_clean['home_team_won'] = (df_clean['winner'] == df_clean['home_team']).astype(int)

print("Rows before cleaning:", len(df))
print("Rows after cleaning:", len(df_clean))
print()
print(df_clean[['date', 'home_team', 'away_team', 'home_shot_first', 'home_team_won']].head())
print()
print("Missing values after cleaning:")
print(df_clean.isnull().sum())

df_clean.to_csv("shootouts_cleaned.csv", index=False)
