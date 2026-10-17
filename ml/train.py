"""Train, evaluate, and serialize the Week 4 wine-classification model.

Run from the repository root with: python -m ml.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ROOT / "figures"
for folder in (DATA_DIR, ARTIFACT_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    # The dataset is packaged with scikit-learn, so training does not need a network download.
    dataset = load_wine(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()
    class_names = [str(name) for name in dataset.target_names]
    feature_names = [str(name) for name in X.columns]

    data_export = X.copy()
    data_export["target"] = y
    data_export.to_csv(DATA_DIR / "wine.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ]),
    }

    comparison = []
    fitted = {}
    for name, estimator in candidates.items():
        cv_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="accuracy")
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        metrics = {
            "model": name,
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        }
        comparison.append(metrics)
        fitted[name] = (estimator, y_pred)

    # Model selection is based on mean training-fold CV accuracy, not the held-out test set.
    # If CV means tie, prefer Logistic Regression as the simpler baseline; the held-out test set is not used to choose the model.
    best_record = max(comparison, key=lambda row: (round(row["cv_accuracy_mean"], 10), row["model"] == "Logistic Regression"))
    best_name = best_record["model"]
    best_model, best_pred = fitted[best_name]

    joblib.dump(best_model, ARTIFACT_DIR / "wine_classifier.joblib")
    metadata = {
        "project_name": "Wine Classifier Prediction API",
        "dataset_name": "Wine recognition dataset (scikit-learn bundled dataset)",
        "best_model": best_name,
        "feature_names": feature_names,
        "class_names": class_names,
        "target_description": "Wine cultivar class identifier; class labels are dataset categories, not quality grades.",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(class_names)),
        "model_file": "artifacts/wine_classifier.joblib",
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps({"models": comparison, "selected_model": best_name}, indent=2), encoding="utf-8")

    report = classification_report(y_test, best_pred, target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(ARTIFACT_DIR / "classification_report.csv", index=True)
    cm = confusion_matrix(y_test, best_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(ARTIFACT_DIR / "confusion_matrix.csv")

    # Figure 1: compare held-out metrics for candidate models.
    plot_df = pd.DataFrame(comparison).set_index("model")
    ax = plot_df[["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1.05), rot=0
    )
    ax.set_title("Held-out Test Performance by Model")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png", dpi=180)
    plt.close()

    # Figure 2: selected-model confusion matrix.
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, class_names)
    ax.set_yticks(ticks, class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    # Sample input copied from a real held-out observation so the documented API example is valid.
    sample = X_test.iloc[0].to_dict()
    api_name_map = {"od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"}
    sample_payload = {api_name_map.get(key, key): float(value) for key, value in sample.items()}
    (ARTIFACT_DIR / "sample_request.json").write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")

    print("Training and evaluation completed.")
    print(f"Dataset rows: {len(X)} | features: {X.shape[1]} | classes: {len(class_names)}")
    print(f"Train rows: {len(X_train)} | test rows: {len(X_test)}")
    print("Model comparison:")
    for row in comparison:
        print(
            f"- {row['model']}: CV accuracy={row['cv_accuracy_mean']:.4f} +/- {row['cv_accuracy_std']:.4f}; "
            f"test accuracy={row['test_accuracy']:.4f}; macro F1={row['test_f1_macro']:.4f}"
        )
    print(f"Selected model: {best_name}")
    print(f"Saved model: {ARTIFACT_DIR / 'wine_classifier.joblib'}")


if __name__ == "__main__":
    main()
