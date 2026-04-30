import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # ── Tracking URI dari environment variable (DagsHub / lokal) ────────
    # Jika MLFLOW_TRACKING_URI sudah di-set via env (GitHub Actions / DagsHub),
    # mlflow akan otomatis membacanya — tidak perlu set_tracking_uri() manual.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        print(f"MLflow Tracking URI : {tracking_uri}")
    else:
        print("MLflow Tracking URI : local (mlruns/)")

    # ── Dynamic file path dari MLProject parameter ───────────────────────
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "credit_card_fraud_preprocessing.csv"
    )

    print(f"Loading dataset from : {file_path}")
    data = pd.read_csv(file_path)

    # ── Split features & target ──────────────────────────────────────────
    TARGET_COL = "IsFraud"
    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]

    fraud_count     = int(y.sum())
    non_fraud_count = int((y == 0).sum())
    imbalance_ratio = round(non_fraud_count / fraud_count, 2)

    print(f"Class distribution — Fraud: {fraud_count} | Non-fraud: {non_fraud_count}")
    print(f"Imbalance ratio     : {imbalance_ratio}:1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=42,
        test_size=0.2,
        stratify=y
    )

    input_example = X_train.iloc[0:5]

    # ── Dynamic hyperparameters dari MLProject entry_points ─────────────
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth    = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"\nn_estimators : {n_estimators}")
    print(f"max_depth    : {max_depth}")
    print(f"Train size   : {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # ── MLflow run ───────────────────────────────────────────────────────
    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("n_estimators",    n_estimators)
        mlflow.log_param("max_depth",       max_depth)
        mlflow.log_param("random_state",    42)
        mlflow.log_param("class_weight",    "balanced")
        mlflow.log_param("dataset",         os.path.basename(file_path))
        mlflow.log_param("imbalance_ratio", imbalance_ratio)

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)

        # Log metrics
        mlflow.log_metric("accuracy",       accuracy)
        mlflow.log_metric("precision",      precision)
        mlflow.log_metric("recall",         recall)
        mlflow.log_metric("f1_score",       f1)
        mlflow.log_metric("roc_auc",        roc_auc)
        mlflow.log_metric("fraud_count",    fraud_count)
        mlflow.log_metric("nonfraud_count", non_fraud_count)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"\n── Evaluation Results ──────────────────")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"────────────────────────────────────────")
