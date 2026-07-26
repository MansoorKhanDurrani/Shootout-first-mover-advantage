# Step 6 — Model Selection

## Problem Recap
- **Target:** `home_team_won` (0 or 1) → Binary Classification
- **Dataset size:** 260 rows (small)
- **Key feature:** `home_shot_first`, plus `year`, `is_recent`, `prior_matchup_count`

## Why Classification (not Regression)?
Our target is a category (win/lose), not a continuous number, so a
regression model (like Linear Regression) is the wrong tool here — it's
built to predict continuous values (e.g. price, temperature), not discrete
outcomes.

## Models Chosen for Comparison

| Model | Reasoning |
|---|---|
| **Logistic Regression** | A simple, interpretable baseline. With only 260 rows, a simpler model is less likely to overfit and often generalizes better than something more complex. |
| **Decision Tree** | Can capture non-linear splits (e.g. "if home_shot_first AND is_recent, then...") and produces rules that are easy to explain to a non-technical audience. |
| **Random Forest** | An ensemble of decision trees, usually more robust than a single tree - though with a small dataset like this one, it also carries a higher risk of overfitting, which we'll check for in the evaluation step. |

We deliberately did **not** jump straight to the most complex model
(Random Forest) as the only choice - comparing all three lets us see
which one actually performs best on this specific, small dataset, rather
than assuming a fancier model is automatically better.
