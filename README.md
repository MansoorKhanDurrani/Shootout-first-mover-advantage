#  Penalty Shootout: Does Shooting First Actually Help?

An end-to-end mini ML project investigating a long-standing football debate:
**does shooting first in a penalty shootout give a real advantage?** Built as
a complete pipeline — from raw data to an honestly evaluated classification
model.

---

##  Business Question

In a penalty shootout, does the team that shoots **first** have a higher
chance of winning?

This isn't just trivia — since 2003, football's rule-making body (IFAB) has
experimented with alternate shootout orders (e.g. "ABBA") specifically
because of concerns that shooting first creates a psychological edge. If the
data confirms a real advantage, it has direct implications for how captains
approach the coin toss.

**Problem type:** Binary classification (`home_team_won`: 1 = home team won, 0 = away team won)

---

##  Dataset

**International Football Shootouts** — 683 recorded penalty shootouts from
international football (1967–2026).

Original columns: `date`, `home_team`, `away_team`, `winner`, `first_shooter`

---

##  Data Understanding

- 683 rows, all columns originally stored as text (including `date`)
- **`first_shooter` was missing in 61.9% of rows (423 of 683)** — the single
  biggest data quality issue, and unfortunately the exact column our
  business question depends on
- `winner` was always consistent with either `home_team` or `away_team` (no invalid values)

---

##  Data Cleaning

| Issue | Action | Why |
|---|---|---|
| `first_shooter` missing (62%) | Dropped those rows entirely | Can't guess who shot first — imputing this would fabricate the exact signal we're trying to measure |
| `date` stored as text | Converted to proper datetime | Needed for time-based features |
| `first_shooter` stored as a team name | Derived `home_shot_first` (1/0) | Converts it into a clean signal that matches our home/away framing |
| No target column | Derived `home_team_won` (1/0) from `winner` | This is our prediction target |

**Result:** 683 rows → **260 clean, usable rows**.

---

##  EDA — Answering the Business Question Directly

| Scenario | Home team win rate |
|---|---|
| Home team did **not** shoot first | 49.5% |
| Home team **did** shoot first | **53.5%** |

There's a modest **~4 percentage point first-mover advantage** — consistent
with the real football analytics debate. However, with only 260 matches,
this is a meaningful *descriptive* trend rather than a statistically bulletproof
conclusion on its own.

Additional finding: home teams shot first in 60.4% of shootouts in this
dataset (likely tied to coin-toss dynamics and possibly a small home-field
psychological effect).

*(See `images/win_rate_by_first_shooter.png` and `images/shootouts_over_time.png`.)*

---

##  Feature Engineering

- **`year`** — extracted from date
- **`is_recent`** — 1 if the match was 2000 or later (modern era, more sports psychology/coaching applied to shootouts)
- **`prior_matchup_count`** — how many times these two teams had already met in a shootout before this match (rivalry/history), computed sequentially so no future data leaks in

`is_recent` showed almost no difference in win rate (51.9% vs 51.8%),
suggesting era alone doesn't explain outcomes — kept anyway to let the
models judge its value directly.

---

##  Model Selection

Since this is a **classification** problem (not regression — the target is
a category, not a continuous number), three classifiers were compared
rather than defaulting to the most complex option:

- **Logistic Regression** — simple, interpretable baseline; less prone to overfitting on a small dataset
- **Decision Tree** — captures non-linear splits, easy to explain
- **Random Forest** — ensemble of trees, usually more robust, though riskier to overfit with only 260 rows

---

##  Evaluation — Honest Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.500 | 0.514 | 0.667 | 0.581 |
| Decision Tree | 0.442 | 0.479 | 0.852 | 0.613 |
| Random Forest | 0.442 | 0.479 | 0.852 | 0.613 |
| **Baseline** (always predict "home wins") | **0.519** | — | — | — |

###  The Honest Finding
**None of the three trained models beat the naive baseline** of simply
always predicting "home team wins." That baseline alone scores 51.9%
accuracy - better than every model we trained.

### What the Confusion Matrix Actually Shows (Decision Tree / Random Forest)

|  | Predicted: Away Won | Predicted: Home Won |
|---|---|---|
| **Actual: Away Won** | 0 (True Negative) | 25 (False Positive) |
| **Actual: Home Won** | 4 (False Negative) | 23 (True Positive) |

- **True Negatives = 0:** these models **never once** correctly predicted an away-team win
- **False Positives (25):** the model predicted "home wins" but the away team actually won — the model is essentially guessing the majority class every time, not learning a real pattern
- This tells us the models collapsed into always guessing the more common outcome, rather than genuinely learning from `home_shot_first`

### Why did this happen?
1. **Small dataset** — only 260 rows, 208 for training, is not much for a model to learn a reliable pattern from
2. **Weak signal** — the real win-rate difference we found in EDA (53.5% vs 49.5%) is only ~4 points, which is easily lost in noise with this few examples

### Conclusion
The EDA finding — a modest first-mover advantage — still stands as a
genuine descriptive pattern in the data. But a reliable *predictive* model
would need substantially more shootouts with recorded `first_shooter`
values to confirm and reliably exploit that pattern. This is a good example
of an honest, non-inflated evaluation: a clean pipeline doesn't guarantee
a model that beats a trivial baseline, and it's important to report that
rather than only reporting accuracy at face value.

---

## 📁 Project Structure

```
shootout-first-mover-advantage/
│
├── shootouts.csv                    # Raw dataset
├── shootouts_cleaned.csv            # After cleaning
├── shootouts_features.csv           # After feature engineering
├── X_train.csv / X_test.csv         # Train/test feature splits
├── y_train.csv / y_test.csv         # Train/test target splits
│
├── step1_business_question.md
├── step2_data_understanding.py
├── step3_data_cleaning.py
├── step4_eda.py
├── step5_feature_engineering.py
├── step6_model_selection.md
├── step7_model_training.py
├── step8_model_evaluation.py
│
├── images/
│   ├── win_rate_by_first_shooter.png
│   ├── shootouts_over_time.png
│   └── confusion_matrices.png
│
└── README.md
```

---


## 🌿 Git Workflow

This project followed a strict branching model, with each pipeline stage
developed and merged independently:

- **`main`** — stable, fully verified code
- **`dev`** — integration branch for completed stages
- **`feature/01-business-question` → `feature/08-model-evaluation`** — one branch per pipeline stage, each merged into `dev` via its own reviewed pull request

---

##  Author

**Mansoor Khan Durrani**
