"""
modelling.py (MLProject entry point)
Training model Random Forest - digunakan oleh MLflow Project.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "heart_disease_preprocessing.csv"
TARGET_COL = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    print("=" * 60)
    print("MLflow Project - Heart Disease Classification")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_experiment("Heart Disease - CI")
    mlflow.sklearn.autolog(log_input_examples=True)

    with mlflow.start_run(run_name="CI-Run"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"], ax=ax)
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, dpi=100)
        plt.close()
        mlflow.log_artifact(cm_path, "plots")

        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")
        print("\n" + classification_report(y_test, y_pred))
        print("=" * 60)


if __name__ == "__main__":
    main()
