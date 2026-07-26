"""
Step 4 - Exploratory Data Analysis (EDA)
==========================================
Goal: visualize and quantify whether shooting first relates to winning -
this directly answers our business question before we even build a model.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("shootouts_cleaned.csv", parse_dates=['date'])
sns.set_style("whitegrid")

# Core question: does home_shot_first relate to home_team_won?
win_rate_by_shooter = df.groupby('home_shot_first')['home_team_won'].mean()
print("Win rate when home team did NOT shoot first:", win_rate_by_shooter[0])
print("Win rate when home team DID shoot first:", win_rate_by_shooter[1])
print()
print("How often did the home team shoot first?")
print(df['home_shot_first'].value_counts(normalize=True))
print()
print("Overall home team win rate:", df['home_team_won'].mean())

# Plot 1: Win rate comparison - the key chart for our business question
plt.figure(figsize=(6, 4))
sns.barplot(x=win_rate_by_shooter.index, y=win_rate_by_shooter.values)
plt.xticks([0, 1], ['Home did NOT\nshoot first', 'Home DID\nshoot first'])
plt.ylabel('Home team win rate')
plt.title('Win Rate: First Shooter vs Not')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('images/win_rate_by_first_shooter.png', dpi=120)
plt.close()

# Plot 2: How many shootouts we have data for, per year
df['year'] = df['date'].dt.year
plt.figure(figsize=(8, 4))
df.groupby('year').size().plot(kind='bar')
plt.title('Number of Shootouts per Year (cleaned data)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/shootouts_over_time.png', dpi=120)
plt.close()

# ------------------------------------------------------------------
# Key finding:
# - Home team wins 49.5% of the time when they did NOT shoot first
# - Home team wins 53.5% of the time when they DID shoot first
# - This is a modest ~4 percentage point "first-mover advantage",
#   consistent with real football analytics debates - though with only
#   260 matches, this is a trend worth modeling, not a statistically
#   airtight conclusion on its own.
# ------------------------------------------------------------------
