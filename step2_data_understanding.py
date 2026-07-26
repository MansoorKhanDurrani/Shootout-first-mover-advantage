"""
Step 2 - Data Understanding
============================
Goal: explore the raw data before making any changes, so we know exactly
what cleaning is needed (missing values, wrong types, inconsistent data).
"""

import pandas as pd
pd.set_option('display.max_columns', None)

df = pd.read_csv("shootouts.csv")

print("SHAPE:", df.shape)
print()
print("DTYPES:")
print(df.dtypes)
print()
print("HEAD:")
print(df.head())
print()
print("MISSING VALUES:")
print(df.isnull().sum())
print()
print("Missing % of first_shooter:", round(df['first_shooter'].isnull().mean() * 100, 1))
print()
print("Date range:", df['date'].min(), "to", df['date'].max())

bad_rows = df[~((df['winner'] == df['home_team']) | (df['winner'] == df['away_team']))]
print("\nRows where winner is neither home nor away team:", len(bad_rows))

# ------------------------------------------------------------------
# Key findings from this exploration:
# - 683 rows, 5 columns, all currently stored as text (including date)
# - first_shooter is missing in 61.9% of rows (423 out of 683) - this is
# - No inconsistent/invalid values found in winner (always matches
#   home_team or away_team)
# - date needs to be converted to a proper datetime type
# ------------------------------------------------------------------