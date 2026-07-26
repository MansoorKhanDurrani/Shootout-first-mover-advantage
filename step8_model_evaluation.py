"""
Step 8 - Model Evaluation (Honest Evaluation)
================================================
Goal: evaluate all three trained models on the held-out test set, and
critically interpret the confusion matrix rather than just reporting
accuracy at face value.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "preds": preds,
    }
    print(f"--- {name} ---")
    print(f"Accuracy:  {results[name]['accuracy']:.3f}")
    print(f"Precision: {results[name]['precision']:.3f}")
    print(f"Recall:    {results[name]['recall']:.3f}")
    print(f"F1 Score:  {results[name]['f1']:.3f}\n")

# Baseline comparison: what if we just always predicted "home wins"?
# This is a critical honesty check - a model only proves its worth if it
# beats this trivial baseline.
baseline_acc = (y_test == 1).mean()
print("Baseline (always predict home wins) accuracy:", baseline_acc)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Away Won', 'Home Won'], yticklabels=['Away Won', 'Home Won'])
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('images/confusion_matrices.png', dpi=120)
plt.close()

print("\nConfusion matrix breakdown (TN, FP, FN, TP) for each model:")
for name, res in results.items():
    cm = confusion_matrix(y_test, res['preds'])
    tn, fp, fn, tp = cm.ravel()
    print(f"{name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ------------------------------------------------------------------
# Honest interpretation:
# - None of the three models beat the naive baseline (51.9% accuracy from
#   always predicting "home wins").
# - Decision Tree and Random Forest collapsed to predicting "Home Won"
#   almost every time (TN=0 - they never correctly identified an away win).
#   This means they didn't learn a genuine pattern; they just learned to
#   guess the more common class.
# - This is likely because: (1) the dataset is small (260 rows total,
#   208 for training), and (2) the win-rate difference we found in EDA
#   (53.5% vs 49.5%) is real but too small and noisy for these models to
#   reliably exploit with so few examples.
# - Conclusion: the EDA finding (a modest first-mover advantage) still
#   stands as a descriptive pattern in the data, but a reliable predictive
#   model would need substantially more shootout data with recorded
#   first_shooter values to confirm and exploit it.
# ------------------------------------------------------------------
