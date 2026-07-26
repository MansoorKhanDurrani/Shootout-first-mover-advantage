"""
Step 5 - Feature Engineering
=============================
Goal: create additional features from the existing columns that might
help a model learn patterns beyond just home_shot_first alone.
"""

import pandas as pd

df = pd.read_csv("shootouts_cleaned.csv", parse_dates=['date'])

# 1. year: extracted from date - lets us check if shootout dynamics have
#    changed over time
df['year'] = df['date'].dt.year

# 2. is_recent: matches from 2000 onwards (modern era, with more sports
#    psychology/coaching applied to shootouts) vs older matches
df['is_recent'] = (df['year'] >= 2000).astype(int)

# 3. prior_matchup_count: how many times has this exact pair of teams
#    already met in a shootout before this match? Captures rivalry/history,
#    computed sequentially so no future information leaks in.
df = df.sort_values('date').reset_index(drop=True)
matchup_count = {}
counts = []
for _, row in df.iterrows():
    key = tuple(sorted([row['home_team'], row['away_team']]))
    counts.append(matchup_count.get(key, 0))
    matchup_count[key] = matchup_count.get(key, 0) + 1
df['prior_matchup_count'] = counts

print(df[['date', 'home_team', 'away_team', 'home_shot_first', 'year',
          'is_recent', 'prior_matchup_count', 'home_team_won']].head(10))
print()
print("Win rate - recent matches (2000+):", df[df['is_recent'] == 1]['home_team_won'].mean())
print("Win rate - older matches:", df[df['is_recent'] == 0]['home_team_won'].mean())

df.to_csv("shootouts_features.csv", index=False)

# ------------------------------------------------------------------
# Note: is_recent shows almost no difference in win rate (51.9% vs 51.8%),
# suggesting the era of the match doesn't meaningfully change outcomes.
# We keep it anyway and let the model / feature importance step decide
# whether it adds real predictive value.
# ------------------------------------------------------------------
