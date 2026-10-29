#!/usr/bin/env python3
"""
Illustrate stacking on the beer consumption classification task.

Steps:
1. Build multi-class labels from the beer dataset.
2. Compare a single logistic regression classifier with a stacked ensemble via a holdout split.
3. Report cross-validated metrics for both models.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data, derive multi-class labels, and return features/target."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["Class1"] = (df["Litres"] > 29000).astype(int)
    df["Class2"] = (df["Litres"] > 25000).astype(int)
    df["Class3"] = (df["Litres"] > 22000).astype(int)
    df["Class4"] = (df["Litres"] > 10000).astype(int)
    df["Class"] = df[["Class1", "Class2", "Class3", "Class4"]].sum(axis=1)

    # Month cyclic encoding
    df["Month"] = df["Date"].dt.month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    feature_cols = [
        "AvgTemp",
        "MinTemp",
        "MaxTemp",
        "Rainfall_mm",
        "Weekend",
        "Month_sin",
        "Month_cos",
    ]
    X = df[feature_cols]
    y = df["Class"]
    return X, y


def evaluate_holdout(model, X_train, X_test, y_train, y_test) -> Metrics:
    """Train a model on train split and return metrics on test split."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return Metrics(
        accuracy=accuracy_score(y_test, predictions),
        macro_f1=f1_score(y_test, predictions, average="macro"),
    )


def cross_validate(model, X, y, n_splits: int = 5, random_state: int = 42) -> Metrics:
    """Return mean cross-validated accuracy/macro-F1."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    macro_f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    return Metrics(float(np.mean(accuracy_scores)), float(np.mean(macro_f1_scores)))


def build_stacking_classifier(random_state: int = 42) -> StackingClassifier:
    """Create a stacking ensemble with diverse base learners."""
    logreg_base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=10000, C=1.0)),
        ]
    )
    svc_base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=random_state)),
        ]
    )
    rf_base = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )

    stacking = StackingClassifier(
        estimators=[
            ("logreg", logreg_base),
            ("svc", svc_base),
            ("rf", rf_base),
        ],
        final_estimator=LogisticRegression(max_iter=10000),
        cv=5,
        n_jobs=-1,
        passthrough=True,
    )
    return stacking


def main(csv_path: str) -> None:
    X, y = load_dataset(csv_path)
    print(f"Loaded {len(X)} rows with {X.shape[1]} features from {csv_path}")
    print("Class counts:")
    for cls, count in y.value_counts().sort_index().items():
        print(f"  Class {cls}: {count}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    baseline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=10000)),
        ]
    )
    stacking = build_stacking_classifier(random_state=42)

    print("\n=== Holdout evaluation (25% test split) ===")
    baseline_metrics = evaluate_holdout(baseline, X_train, X_test, y_train, y_test)
    print(
        f"Logistic regression -> Accuracy: {baseline_metrics.accuracy:.3f} | Macro-F1: {baseline_metrics.macro_f1:.3f}"
    )
    stacking_metrics = evaluate_holdout(stacking, X_train, X_test, y_train, y_test)
    print(
        f"Stacking ensemble -> Accuracy: {stacking_metrics.accuracy:.3f} | Macro-F1: {stacking_metrics.macro_f1:.3f}"
    )

    print("\n=== Stratified 5-fold cross-validation ===")
    baseline_cv = cross_validate(baseline, X, y)
    print(
        f"Logistic regression -> Mean Accuracy: {baseline_cv.accuracy:.3f} | Mean Macro-F1: {baseline_cv.macro_f1:.3f}"
    )
    stacking_cv = cross_validate(stacking, X, y)
    print(
        f"Stacking ensemble -> Mean Accuracy: {stacking_cv.accuracy:.3f} | Mean Macro-F1: {stacking_cv.macro_f1:.3f}"
    )

    print(
        "\nStacking blends heterogeneous learners by feeding their predictions into a meta-model. "
        "Toggle `passthrough`, adjust the base estimators, or change the meta-classifier to "
        "experiment with different ensemble behaviours."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacking demo on the beer dataset.")
    parser.add_argument(
        "--csv",
        default="./Beerconsumption.csv",
        help="Path to the beer consumption CSV file.",
    )
    args = parser.parse_args()
    main(args.csv)
