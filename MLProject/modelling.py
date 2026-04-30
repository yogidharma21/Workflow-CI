import os
import sys
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# ============================================================
# modelling.py — MLProject Entry Point
# Credit Card Fraud Detection — CI Workflow
# Author: Yogi-Dharma
# CATATAN: Tidak ada set_tracking_uri() — mlruns disimpan lokal
#          lalu di-push ke GitHub oleh CI
# ============================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Path dataset — bisa dari sys.argv atau default
    file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "credit_card_fraud_dataset_preprocessing",
        "credit_card_fraud_preprocessing.csv"
    )

    print(f"[INFO] Membaca dataset dari: {file_path}")
    df = pd.read_csv(file_path)

    X = df.drop(columns=["IsFraud"])
    y = df["IsFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    input_example = X_train[0:5]

    # Best params dari eksperimen Kriteria 2
    n_estimators = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    max_depth    = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    with mlflow.start_run():
        mlflow.autolog()

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        print(f"[DONE] Training selesai! Accuracy: {accuracy:.4f}")
