#!/usr/bin/env python3
"""
Illustrate boosting on the beer consumption classification task.

Steps:
1. Build multi-class labels from the beer dataset.
2. Compare a baseline decision tree with Gradient Boosting via a holdout split.
3. Report cross-validated metrics for both models.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


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


def main(csv_path: str) -> None:
    X, y = load_dataset(csv_path)
    print(f"Loaded {len(X)} rows with {X.shape[1]} features from {csv_path}")
    print("Class counts:")
    for cls, count in y.value_counts().sort_index().items():
        print(f"  Class {cls}: {count}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    boosting = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=3,
        random_state=42,
    )

    print("\n=== Holdout evaluation (25% test split) ===")
    tree_metrics = evaluate_holdout(base_tree, X_train, X_test, y_train, y_test)
    print(
        f"Decision tree -> Accuracy: {tree_metrics.accuracy:.3f} | Macro-F1: {tree_metrics.macro_f1:.3f}"
    )
    boosting_metrics = evaluate_holdout(boosting, X_train, X_test, y_train, y_test)
    print(
        f"Gradient Boosting -> Accuracy: {boosting_metrics.accuracy:.3f} | Macro-F1: {boosting_metrics.macro_f1:.3f}"
    )

    print("\n=== Stratified 5-fold cross-validation ===")
    tree_cv = cross_validate(base_tree, X, y)
    print(
        f"Decision tree -> Mean Accuracy: {tree_cv.accuracy:.3f} | Mean Macro-F1: {tree_cv.macro_f1:.3f}"
    )
    boosting_cv = cross_validate(boosting, X, y)
    print(
        f"Gradient Boosting -> Mean Accuracy: {boosting_cv.accuracy:.3f} | Mean Macro-F1: {boosting_cv.macro_f1:.3f}"
    )

    print(
        "\nBoosting builds an additive ensemble of shallow trees trained on residual errors. "
        "Experiment with `learning_rate`, `n_estimators`, or `max_depth` to balance bias and variance."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boosting demo on the beer dataset.")
    parser.add_argument(
        "--csv",
        default="./Beerconsumption.csv",
        help="Path to the beer consumption CSV file.",
    )
    args = parser.parse_args()
    main(args.csv)
