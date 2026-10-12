"""
Predictive Modeling Using Machine Learning
Task: train Logistic Regression, Decision Tree and Random Forest classifiers,
evaluate them, and visualise performance (confusion matrices + ROC curves).

Dataset: Breast Cancer Wisconsin (built into scikit-learn).
To use your own data, replace the "1. LOAD DATA" section with pd.read_csv(...)
and set X (features) and y (target).
"""
import matplotlib
matplotlib.use("Agg")  # remove this line if you want plots to pop up in a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------- 1. LOAD DATA
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print("Class balance:\n", y.value_counts(normalize=True).round(3), "\n")
print("Missing values:", int(X.isna().sum().sum()), "\n")

# ------------------------------------------------------- 2. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------------ 3. MODELS
models = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# ------------------------------------------------------- 4. TRAIN + EVALUATE
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV Acc (5-fold)": cv.mean(),
    })
    print(f"=== {name} ===")
    print(classification_report(y_test, pred, target_names=data.target_names))

summary = pd.DataFrame(results).set_index("Model").round(4)
print("Model comparison:\n", summary, "\n")
summary.to_csv("model_comparison.csv")

# ------------------------------------------------------------ 5. VISUALISE
# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models.items()):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=data.target_names,
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)

# ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_title("ROC Curves")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)

# Feature importance (Random Forest)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
fig, ax = plt.subplots(figsize=(7, 5))
imp.sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_title("Top 10 Features - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)

best = summary["ROC-AUC"].idxmax()
print(f"Best model by ROC-AUC: {best}")
print("Saved: confusion_matrices.png, roc_curves.png, feature_importance.png, model_comparison.csv")
