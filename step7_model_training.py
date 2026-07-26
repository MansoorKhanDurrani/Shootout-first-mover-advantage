"""
Step 7 - Model Training
=========================
Goal: split the data into train/test sets and train all three chosen
models. We hold out 20% of the data as a test set so we can honestly
evaluate performance on matches the models have never seen.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("shootouts_features.csv")

feature_cols = ['home_shot_first', 'is_recent', 'prior_matchup_count']
X = df[feature_cols]
y = df['home_team_won']

# stratify=y keeps the win/loss ratio consistent between train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained.")

# Save the splits so the evaluation step (Step 8) can reuse the exact
# same train/test data without retraining from scratch
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
