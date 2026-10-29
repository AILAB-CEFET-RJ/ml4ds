#!/usr/bin/env python3
"""
Illustrate bagging on the beer consumption classification task.

Steps:
1. Build multi-class labels from the beer dataset.
2. Compare a single decision tree against a bagging ensemble via a holdout split.
3. Report cross-validated metrics for both models.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
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

    tree = DecisionTreeClassifier(random_state=42)
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=200,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )

    print("\n=== Holdout evaluation (25% test split) ===")
    tree_metrics = evaluate_holdout(tree, X_train, X_test, y_train, y_test)
    print(
        f"Single tree -> Accuracy: {tree_metrics.accuracy:.3f} | Macro-F1: {tree_metrics.macro_f1:.3f}"
    )
    bagging_metrics = evaluate_holdout(bagging, X_train, X_test, y_train, y_test)
    print(
        f"Bagging ensemble -> Accuracy: {bagging_metrics.accuracy:.3f} | Macro-F1: {bagging_metrics.macro_f1:.3f}"
    )

    print("\n=== Stratified 5-fold cross-validation ===")
    tree_cv = cross_validate(tree, X, y)
    print(
        f"Single tree -> Mean Accuracy: {tree_cv.accuracy:.3f} | Mean Macro-F1: {tree_cv.macro_f1:.3f}"
    )
    bagging_cv = cross_validate(bagging, X, y)
    print(
        f"Bagging ensemble -> Mean Accuracy: {bagging_cv.accuracy:.3f} | Mean Macro-F1: {bagging_cv.macro_f1:.3f}"
    )

    print(
        "\nBagging reduces variance by averaging many bootstrapped trees; here we see its effect "
        "compared to a single tree baseline. Try adjusting `n_estimators`, `max_samples`, or the "
        "base tree depth to explore bias-variance trade-offs."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bagging demo on the beer dataset.")
    parser.add_argument(
        "--csv",
        default="./Beerconsumption.csv",
        help="Path to the beer consumption CSV file.",
    )
    args = parser.parse_args()
    main(args.csv)
